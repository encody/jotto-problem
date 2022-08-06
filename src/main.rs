// 33m 48s

use std::{
    cmp::Ordering,
    collections::{BTreeSet, HashMap},
    fs,
    str::FromStr,
};

struct PathList<'a, T> {
    length: usize,
    value: T,
    next: Option<&'a Self>,
}

impl<'a, T> PathList<'a, T> {
    pub fn new(value: T) -> Self {
        Self {
            length: 1,
            value,
            next: None,
        }
    }

    pub fn prefix(&'a self, value: T) -> PathList<'a, T> {
        Self {
            length: self.length + 1,
            value,
            next: Some(self),
        }
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn iter(&self) -> PathListIter<T> {
        PathListIter(Some(self))
    }
}

struct PathListIter<'a, T>(pub Option<&'a PathList<'a, T>>);

impl<'a, T> Iterator for PathListIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.map(|list| {
            let retval = &list.value;
            self.0 = list.next;
            retval
        })
    }
}

struct Graph<T> {
    pub nodes: Vec<Node<T>>,
    pub edges: Vec<Edge>,
}

impl<T> Graph<T> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn node(&self, node_index: &NodeIndex) -> &Node<T> {
        &self.nodes[node_index.0]
    }

    pub fn edge(&self, edge_index: &EdgeIndex) -> &Edge {
        &self.edges[edge_index.0]
    }

    pub fn iter_edges(&self, node_index: &NodeIndex) -> EdgeIter<'_, T> {
        EdgeIter {
            graph: self,
            edge_index: self.nodes[node_index.0].edges.as_ref(),
        }
    }

    pub fn add_node(&mut self, value: T) -> NodeIndex {
        let ix = self.nodes.len();
        self.nodes.push(Node { value, maximum_further_depth: 0, edges: None });
        NodeIndex(ix)
    }

    pub fn add_edge(&mut self, source: &NodeIndex, target: NodeIndex) {
        let ix = EdgeIndex(self.edges.len());
        let target_depth = {
            let target_node = &self.nodes[target.0];
            target_node.maximum_further_depth
        };

        let node = &mut self.nodes[source.0];
        self.edges.push(Edge {
            target,
            next: node.edges.clone(),
        });
        node.edges = Some(ix);
        node.maximum_further_depth = node.maximum_further_depth.max(target_depth + 1);
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Debug)]
struct NodeIndex(pub usize);

struct Node<T> {
    pub value: T,
    pub maximum_further_depth: usize,
    pub edges: Option<EdgeIndex>,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Debug)]
struct EdgeIndex(pub usize);

struct Edge {
    pub target: NodeIndex,
    pub next: Option<EdgeIndex>,
}

struct EdgeIter<'a, T> {
    pub graph: &'a Graph<T>,
    pub edge_index: Option<&'a EdgeIndex>,
}

impl<'a, T> Iterator for EdgeIter<'a, T> {
    type Item = &'a EdgeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        self.edge_index.map(|edge_index| {
            let edge = &self.graph.edges[edge_index.0];

            self.edge_index = edge.next.as_ref();

            edge_index
        })
    }
}

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
        self.0.count_ones() as usize
    }

    pub fn intersects(&self, other: &Self) -> bool {
        (self.0 & other.0) != 0
    }

    pub fn is_disjoint(&self, other: &Self) -> bool {
        (self.0 & other.0) == 0
    }

    pub fn union(&self, other: &Self) -> Self {
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

fn printify_words(words: &[Word], original_word: &HashMap<Word, Vec<&str>>) -> String {
    format!(
        "{:?}",
        words
            .iter()
            .map(|w| { original_word.get(w).unwrap() })
            .collect::<Vec<_>>()
    )
}

fn scaled_progress(completed_items: usize, total_items: usize) -> f64 {
    let total_work = total_items * total_items;

    let remaining_items = total_items - completed_items;
    let completed_work = total_work - (remaining_items * remaining_items);

    completed_work as f64 / total_work as f64
}

fn main() {
    //let path = "/Users/jacob/Downloads/wordle-answers-alphabetical.txt";
    //let path = "/Users/jacob/Downloads/wordle-allowed-guesses.txt";
    let path = "/Users/jacob/Downloads/words_alpha.txt";
    let word_length = 5; // # of letters
    let solution_length = 5; // # of words

    let contents = fs::read_to_string(path).unwrap();

    let mut original_word: HashMap<Word, Vec<&str>> = HashMap::new();

    let lines = contents.lines();

    println!("Input file contains {} words", lines.clone().count());

    let words = lines
        .filter_map(|s| {
            let s = s.trim();

            if s.len() != word_length {
                return None;
            }

            let w = Word::from_str(s).unwrap();
            if w.len() == word_length {
                if let Some(word_set) = original_word.get_mut(&w) {
                    word_set.push(s);
                } else {
                    let word_set = vec![s];
                    original_word.insert(w, word_set);
                }
                Some(w)
            } else {
                None
            }
        })
        // .take(10)
        .collect::<BTreeSet<Word>>();

    let words = words.into_iter().collect::<Vec<_>>();

    println!(
        "Found {} unique {word_length}-letter combinations",
        words.len(),
    );

    let mut node_indices = HashMap::new();

    println!("Constructing graph");

    let graph = words
        .iter()
        .enumerate()
        // deepest_paths will go through the words in order, so we only need to set up the
        // graph from back to front. Each word will only have links to words that appear
        // later in the words list.
        .rev()
        .fold(Graph::new(), |mut graph, (i, word)| {
            let node_ix = graph.add_node(*word);
            node_indices.insert(word, node_ix.clone());

            words[(i + 1)..]
                .iter()
                .filter(|w| word.is_disjoint(*w))
                .for_each(|w| {
                    graph.add_edge(&node_ix, node_indices[w].clone());
                });

            graph
        });

    println!("Done constructing graph");

    let (solution_len, solutions) = words.iter().map(|w| &node_indices[w]).enumerate().fold(
        (0, Vec::new()),
        |(final_len, mut final_set), (i, n)| {
            let (len, set) = deepest_paths(
                &graph,
                n,
                &PathList::new(&graph.node(n).value),
                solution_length,
            );

            let (len, set) = if len < solution_length {
                (final_len, final_set)
            } else {
                match len.cmp(&final_len) {
                    Ordering::Less => (final_len, final_set),
                    Ordering::Equal => {
                        final_set.extend(set);
                        (final_len, final_set)
                    }
                    Ordering::Greater => (len, set),
                }
            };

            let spc = scaled_progress(i + 1, words.len()) * 100.0;
            println!(
                "Finished {i} / {} ({spc:.2}%)\t{solution_length}-word solutions found: {}",
                words.len(),
                set.len(),
            );

            // if !set.is_empty() {
            //     panic!("quick return for testing");
            // }

            (len, set)
        },
    );

    fn deepest_paths<'a, 'g>(
        graph: &'g Graph<Word>,
        node_ix: &'g NodeIndex,
        current_path: &'a PathList<&Word>,
        target_length: usize,
    ) -> (usize, Vec<Vec<&'g NodeIndex>>) {
        let minimum_required_length_remaining = target_length.saturating_sub(current_path.len());

        let filtered_edges = graph
            .iter_edges(node_ix)
            .filter_map(|edge_ix| {
                let next_node_ix = &graph.edge(edge_ix).target;

                let next_node = graph.node(next_node_ix);

                if next_node.maximum_further_depth + 1 < minimum_required_length_remaining {
                    return None;
                }

                current_path
                    .iter()
                    .skip(1) // node_ix is in current_path, so edges were already checked
                    .all(|w| next_node.value.is_disjoint(w))
                    .then(|| {
                        let next_path = current_path.prefix(&next_node.value);
                        (next_node_ix, next_path)
                    })
            })
            .collect::<Vec<_>>();

        if filtered_edges.len() < minimum_required_length_remaining {
            return Default::default();
        }

        let (mut path_len, mut paths) = filtered_edges
            .into_iter()
            .map(|(next_node_ix, next_path)| {
                deepest_paths(graph, next_node_ix, &next_path, target_length)
            })
            .fold(
                (0, Vec::new()),
                |(path_len, mut paths), (node_path_len, node_paths)| match node_path_len
                    .cmp(&path_len)
                {
                    Ordering::Less => (path_len, paths),
                    Ordering::Equal => {
                        paths.extend(node_paths);
                        (path_len, paths)
                    }
                    Ordering::Greater => (node_path_len, node_paths),
                },
            );

        if path_len == 0 {
            (1, vec![vec![node_ix]])
        } else {
            paths = paths
                .into_iter()
                .map(|mut s| {
                    s.push(node_ix);
                    s
                })
                .collect();

            path_len += 1;

            (path_len, paths)
        }
    }

    println!(
        "Found {} solutions that are {solution_len} words long",
        solutions.len(),
    );

    for (i, solution) in solutions.iter().enumerate() {
        println!(
            "\t[{i}]: {}",
            printify_words(
                &solution
                    .iter()
                    .map(|n| graph.node(n).value)
                    .collect::<Vec<_>>(),
                &original_word
            )
        );
    }

    println!(
        "Found {} solutions that are {solution_len} words long",
        solutions.len(),
    );
}
