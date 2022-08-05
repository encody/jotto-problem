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
        self.nodes.push(Node { value, edges: None });
        NodeIndex(ix)
    }

    pub fn add_edge(&mut self, source: &NodeIndex, target: NodeIndex) {
        let ix = EdgeIndex(self.edges.len());
        let node = &mut self.nodes[source.0];
        self.edges.push(Edge {
            target,
            next: node.edges.clone(),
        });
        node.edges = Some(ix);
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Debug)]
struct NodeIndex(pub usize);

struct Node<T> {
    pub value: T,
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
        let mut total = 0;
        let mut i = self.0;

        while i != 0 {
            total += i & 1;
            i >>= 1;
        }

        total as usize
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
    let target_length = 5;

    let contents = fs::read_to_string(path).unwrap();

    let mut original_word: HashMap<Word, BTreeSet<&str>> = HashMap::new();

    let lines = contents.lines();

    println!("Input file contains {} words", lines.clone().count());

    let words = lines
        .filter_map(|s| {
            let s = s.trim();

            if s.len() != 5 {
                return None;
            }

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
                target_length,
            );

            let (len, set) = if len < target_length {
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

            let pc = (i as f32) / (words.len() as f32) * 100.0;
            println!(
                "Finished {i} / {} ({pc:.2}%)\t{} {}-word solutions found",
                words.len(),
                set.len(),
                target_length,
            );

            (len, set)
        },
    );

    fn deepest_paths<'a, 'g>(
        graph: &'g Graph<Word>,
        node_ix: &'g NodeIndex,
        current_path: &'a PathList<&Word>,
        target_length: usize,
    ) -> (usize, Vec<Vec<&'g NodeIndex>>) {
        let filtered_edges = graph
            .iter_edges(node_ix)
            .filter_map(|edge_ix| {
                let next_node_ix = &graph.edge(edge_ix).target;

                let next_node = graph.node(next_node_ix);

                current_path
                    .iter()
                    .all(|w| next_node.value.is_disjoint(w))
                    .then(|| {
                        let next_path = current_path.prefix(&next_node.value);
                        (next_node_ix, next_path)
                    })
            })
            .collect::<Vec<_>>();

        if filtered_edges.len() < (target_length - current_path.len()) {
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
                &solution.iter().map(|n| graph.node(n).value).collect(),
                &original_word
            )
        );
    }

    println!(
        "Found {} solutions that are {solution_len} words long",
        solutions.len(),
    );
}
