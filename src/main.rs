use std::{
    collections::{BTreeSet, HashMap},
    fs,
    str::FromStr,
};

const LOWERCASE_A: u8 = 97;

#[derive(Default, PartialEq, Eq, PartialOrd, Ord, Copy, Clone, Debug, Hash)]
struct Word1(pub u32);

impl Word1 {
    pub fn get_chars(&self) -> BTreeSet<char> {
        let mut set = BTreeSet::new();

        for x in 0..32 {
            if self.0 & (1 << x) != 0 {
                set.insert((x + LOWERCASE_A) as char);
            }
        }

        set
    }

    pub fn len(&self) -> usize {
        let mut total = 0;
        let mut i = self.0;

        while i != 0 {
            total += i & 1;
            i >>= 1;
        }

        total as usize
    }

    pub fn intersects(&self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    pub fn is_disjoint(&self, other: Self) -> bool {
        (self.0 & other.0) == 0
    }

    pub fn union(&self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl FromStr for Word1 {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(
            s.to_ascii_lowercase()
                .as_bytes()
                .iter()
                .fold(0, |i, b| i | (1 << (b - LOWERCASE_A))),
        ))
    }
}

type Word = BTreeSet<char>;

fn words_pretty(
    ixs: &BTreeSet<usize>,
    words: &[Word],
    original_word: &HashMap<Word, BTreeSet<&str>>,
) -> String {
    format!(
        "{:?}",
        ixs.iter()
            .map(|ix| original_word
                .get(&words[*ix])
                .unwrap()
                .iter()
                .next()
                .unwrap())
            .collect::<Vec<_>>()
    )
}

fn word_pretty(ix: usize, words: &[Word], original_word: &HashMap<Word, BTreeSet<&str>>) -> String {
    let ow = original_word
        .get(&words[ix])
        .unwrap()
        .iter()
        // .take(5)
        .collect::<Vec<_>>();
    format!("word[{ix}]: {ow:?}")
}

// fn word_to_string(word: &Word) -> String {
//     word.iter().collect::<String>()
// }
//

fn printify_words(
    words: &BTreeSet<Word1>,
    original_word: &HashMap<Word1, BTreeSet<&str>>,
) -> String {
    format!(
        "{:?}",
        words
            .iter()
            .map(|w| { original_word.get(w).unwrap() })
            .collect::<Vec<_>>()
    )
}

fn main() {
    //let path = "/Users/jacob/Downloads/wordle-answers-alphabetical.txt";
    let path = "/Users/jacob/Downloads/wordle-allowed-guesses.txt";
    let word_length = 5;

    let contents = fs::read_to_string(path).unwrap();

    let mut original_word: HashMap<Word1, BTreeSet<&str>> = HashMap::new();

    let lines = contents.lines();

    println!("Input file contains {} words", lines.clone().count());

    let words = lines
        .filter_map(|s| {
            let s = s.trim();
            let w = Word1::from_str(s).unwrap();
            if w.len() == word_length {
                if let Some(word_set) = original_word.get_mut(&w) {
                    word_set.insert(s);
                } else {
                    let mut word_set = BTreeSet::new();
                    word_set.insert(s);
                    original_word.insert(w, word_set);
                }
                Some(w)
            } else {
                None
            }
        })
        //.take(500)
        .collect::<BTreeSet<Word1>>();

    let words = words.into_iter().collect::<Vec<_>>();

    fn mds(
        already_used: Word1,
        words: &[Word1],
        monitor: Option<&dyn Fn(usize, &BTreeSet<Word1>)>,
    ) -> BTreeSet<Word1> {
        let mut set = BTreeSet::new();

        for (i, word) in words.iter().enumerate() {
            if word.is_disjoint(already_used) {
                let mut rec = mds(already_used.union(*word), &words[i..], None);
                rec.insert(*word);
                if rec.len() > set.len() {
                    set = rec;
                }
            }

            if let Some(monitor) = monitor {
                monitor(i, &set);
            }
        }

        set
    }

    let longest = mds(
        Default::default(),
        &words,
        Some(&|i, set| {
            let pc = i as f32 / words.len() as f32 * 100.0;
            println!(
                "Progress: {i} / {} {pc:.2}% {}",
                words.len(),
                printify_words(set, &original_word),
            );
        }),
    );

    println!("Solution: {}", printify_words(&longest, &original_word));

    // println!("Words matching criteria: {}", words.len());

    // println!("Building graph...");

    // let mut graph = words.iter().enumerate().fold(
    //     HashMap::<usize, BTreeSet<usize>>::new(),
    //     |mut acc, (i, main_word)| {
    //         let disjoint_set = words.iter().enumerate().take(i).fold(
    //             BTreeSet::new(),
    //             |mut disjoint_set, (j, cmp_word)| {
    //                 if main_word.is_disjoint(cmp_word) {
    //                     disjoint_set.insert(j);
    //                 }
    //                 disjoint_set
    //             },
    //         );

    //         // update foreign references
    //         disjoint_set.iter().for_each(|word_ix| {
    //             acc.get_mut(word_ix).unwrap().insert(i);
    //         });

    //         acc.insert(i, disjoint_set);

    //         if i % 100 == 0 {
    //             let pc = (i as f64 + 1.0) / words.len() as f64 * 100.0;

    //             println!("Graph {pc:.1}% built");
    //         }

    //         acc
    //     },
    // );

    // println!("Done building graph");

    // Print out graph
    //    graph.iter().take(5).for_each(|(word_ix, disjoint_set)| {
    //        println!("{}", word_pretty(*word_ix, &words, &original_word));
    //        println!("Disjoint set:");
    //        for foreign_ix in disjoint_set.iter().take(5) {
    //            println!("\t{}", word_pretty(*foreign_ix, &words, &original_word));
    //        }
    //    });

    fn traverse(
        level: usize,
        pool: BTreeSet<usize>,
        graph: &mut HashMap<usize, BTreeSet<usize>>,
        words: &[Word],
        original_word: &HashMap<Word, BTreeSet<&str>>,
    ) -> (usize, BTreeSet<BTreeSet<usize>>) {
        pool.iter()
            .fold((0, BTreeSet::new()), |(largest_size, mut acc), ix| {
                let disjoint_set = graph.get(ix).unwrap();

                //println!("{disjoint_set:?}");

                let new_pool: BTreeSet<usize> = disjoint_set.intersection(&pool).copied().collect();

                //println!("{}new pool: {}", "\t".repeat(level), new_pool.len());

                let (mut subsets_size, mut largest_subsets) =
                    traverse(level + 1, new_pool, graph, words, original_word);

                //println!("{}largest subsets: {}", "\t".repeat(level), largest_subsets.len());

                if subsets_size == 0 {
                    largest_subsets.insert(BTreeSet::from([*ix]));
                } else {
                    largest_subsets = largest_subsets
                        .into_iter()
                        .map(|mut set| {
                            set.insert(*ix);
                            set
                        })
                        .collect();
                }

                subsets_size += 1;

                //largest_subsets.iter().insert(*ix);

                //            if level == 0 {
                //                // remove index and foreign entries
                //                let foreign_ixs = graph.remove(ix).unwrap();
                //                foreign_ixs.iter().for_each(|foreign_ix| {
                //                    graph.get_mut(foreign_ix).unwrap().remove(ix);
                //                });
                //            }
                //

                let r = if subsets_size == largest_size {
                    acc.extend(largest_subsets);
                    (largest_size, acc)
                } else if subsets_size > largest_size {
                    (subsets_size, largest_subsets)
                } else {
                    (largest_size, acc)
                };

                if level == 0 {
                    let pc = (*ix as f64 + 1.0) / words.len() as f64 * 100.0;
                    let solutions = r.1.len();
                    if solutions > 0 {
                        println!(
                            "Progress: {ix} / {} ({pc:.2}%) [solutions={solutions}] {}",
                            words.len(),
                            words_pretty(r.1.iter().next().unwrap(), words, original_word),
                        );
                    }
                }

                r
            })
    }

    // let (_, largest_subsets) = traverse(
    //     0,
    //     (0..words.len()).collect(),
    //     &mut graph,
    //     &words,
    //     &original_word,
    // );

    // println!("Largest subset");
    // for (i, subset) in largest_subsets.iter().enumerate() {
    //     println!("solution {i}");
    //     for word_ix in subset.iter() {
    //         println!("\t{}", word_pretty(*word_ix, &words, &original_word));
    //     }
    // }
}
