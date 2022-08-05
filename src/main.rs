use std::{
    cmp::Ordering,
    collections::{BTreeSet, HashMap},
    fs,
    str::FromStr,
};

const LOWERCASE_A: u8 = 97;

#[derive(Default, PartialEq, Eq, PartialOrd, Ord, Copy, Clone, Debug, Hash)]
struct Word(pub u32);

impl Word {
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

impl FromStr for Word {
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

fn printify_words(words: &BTreeSet<Word>, original_word: &HashMap<Word, BTreeSet<&str>>) -> String {
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
    //let path = "/Users/jacob/Downloads/wordle-allowed-guesses.txt";
    let path = "/Users/jacob/Downloads/words_alpha.txt";
    let word_length = 5;

    let contents = fs::read_to_string(path).unwrap();

    let mut original_word: HashMap<Word, BTreeSet<&str>> = HashMap::new();

    let lines = contents.lines();

    println!("Input file contains {} words", lines.clone().count());

    let words = lines
        .filter_map(|s| {
            let s = s.trim();
            let w = Word::from_str(s).unwrap();
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
        .collect::<BTreeSet<Word>>();

    let words = words.into_iter().collect::<Vec<_>>();

    println!(
        "Found {} unique {word_length}-letter combinations",
        words.len(),
    );

    fn mds(
        already_used: Word,
        words: &[Word],
        monitor: Option<&dyn Fn(usize, &BTreeSet<BTreeSet<Word>>)>,
    ) -> (usize, BTreeSet<BTreeSet<Word>>) {
        let mut solutions = BTreeSet::new();
        let mut solution_len = 0;

        for (i, word) in words
            .iter()
            .enumerate()
            .filter(|(_, word)| word.is_disjoint(already_used))
        {
            let (mut sub_solution_len, mut sub_solutions) =
                mds(already_used.union(*word), &words[i..], None);

            sub_solutions = sub_solutions
                .into_iter()
                .map(|mut set| {
                    set.insert(*word);
                    set
                })
                .collect::<BTreeSet<_>>();

            if sub_solutions.is_empty() {
                sub_solutions.insert(BTreeSet::from([*word]));
            }

            sub_solution_len += 1;

            match sub_solution_len.cmp(&solution_len) {
                Ordering::Greater => {
                    solution_len = sub_solution_len;
                    solutions = sub_solutions;
                }
                Ordering::Equal => {
                    solutions.extend(sub_solutions);
                }
                _ => {}
            }

            if let Some(monitor) = monitor {
                monitor(i, &solutions);
            }
        }

        (solution_len, solutions)
    }

    let (solution_len, solutions) = mds(
        Default::default(),
        &words,
        Some(&|i, set| {
            let pc = i as f32 / words.len() as f32 * 100.0;
            let single_words_opt = set.iter().next();
            if let Some(sample) = single_words_opt {
                println!(
                    "Progress: {i} / {} {pc:.2}% {}",
                    words.len(),
                    printify_words(sample, &original_word),
                );
            }
        }),
    );

    println!(
        "Found {} solutions that are {solution_len} words long",
        solutions.len(),
    );

    for (i, solution) in solutions.iter().enumerate() {
        println!("\t[{i}]: {}", printify_words(solution, &original_word));
    }

    println!(
        "Found {} solutions that are {solution_len} words long",
        solutions.len(),
    );
}
