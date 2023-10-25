// Some of the if branches might be more readable if not collapsed
#![allow(clippy::collapsible_if)]
// This lint never produces correct suggestions in our case.
#![allow(clippy::mutex_atomic)]

use genome_graph::bigraph::algo::eulerian::{
    compute_eulerian_superfluous_out_biedges, find_non_eulerian_binodes_with_differences,
};
use genome_graph::bigraph::interface::dynamic_bigraph::DynamicEdgeCentricBigraph;
use genome_graph::bigraph::interface::static_bigraph::StaticBigraph;
use genome_graph::bigraph::interface::static_bigraph::StaticEdgeCentricBigraph;
use genome_graph::bigraph::interface::BidirectedData;
use genome_graph::bigraph::traitgraph::index::{GraphIndex, OptionalGraphIndex};
use genome_graph::bigraph::traitgraph::interface::{GraphBase, ImmutableGraphContainer};
use genome_graph::bigraph::traitgraph::traitsequence::interface::Sequence;
use genome_graph::bigraph::traitgraph::walks::{EdgeWalk, VecEdgeWalk};
use itertools::Itertools;
use log::{info, warn};
use simplelog::{ColorChoice, CombinedLogger, Config, LevelFilter, TermLogger, TerminalMode};
use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io::BufWriter;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use traitgraph_algo::dijkstra::{DijkstraTargetMap, DijkstraWeightedEdgeData};

pub mod eulertigs;
pub mod greedytigs;
pub mod matchtigs;
pub mod pathtigs;

const TARGET_DIJKSTRA_BLOCK_TIME: f32 = 5.0; // seconds

pub fn initialise_logging(log_level: LevelFilter) {
    CombinedLogger::init(vec![TermLogger::new(
        log_level,
        Config::default(),
        TerminalMode::Mixed,
        ColorChoice::Auto,
    )])
    .unwrap();

    info!("Logging initialised successfully");
}

/// An algorithm to compute tigs for a graph.
pub trait TigAlgorithm<Graph: GraphBase>: Default {
    /// The configuration of the algorithm.
    type Configuration;

    /// Compute the tigs given a graph and configuration.
    fn compute_tigs(
        graph: &mut Graph,
        configuration: &Self::Configuration,
    ) -> Vec<VecEdgeWalk<Graph>>;
}

/// The type of the data structure to store the weight of visited nodes in Dijkstra's algorithm.
#[derive(Eq, PartialEq, Debug, Clone, Copy)]
pub enum NodeWeightArrayType {
    /// Use the [EpochNodeWeightArray].
    EpochNodeWeightArray,

    /// Use the [hashbrown::HashMap] to store node weights.
    HashbrownHashMap,
}

impl FromStr for NodeWeightArrayType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "EpochNodeWeightArray" => Self::EpochNodeWeightArray,
            "HashbrownHashMap" => Self::HashbrownHashMap,
            other => {
                return Err(format!("Unknown node weight array type: {other}"));
            }
        })
    }
}

/// The heap data structure used by Dijkstra's algorithm.
#[derive(Eq, PartialEq, Debug, Clone, Copy)]
pub enum HeapType {
    /// Use the [BinaryHeap](std::collections::BinaryHeap).
    StdBinaryHeap,
}

impl FromStr for HeapType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "StdBinaryHeap" => Self::StdBinaryHeap,
            other => {
                return Err(format!("Unknown heap type: {other}"));
            }
        })
    }
}

/// The performance data collector used by Dijkstra's algorithm.
#[derive(Eq, PartialEq, Debug, Clone, Copy)]
pub enum PerformanceDataType {
    /// Collect no performance data.
    None,
    /// Collect all possible performance data.
    Complete,
}

impl FromStr for PerformanceDataType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "None" => Self::None,
            "Complete" => Self::Complete,
            other => {
                return Err(format!("Unknown performance data type: {other}"));
            }
        })
    }
}

pub struct RelaxedAtomicBoolVec {
    map: Vec<AtomicBool>,
}

impl RelaxedAtomicBoolVec {
    pub fn new(len: usize) -> Self {
        Self {
            map: std::iter::repeat(false).map(Into::into).take(len).collect(),
        }
    }

    pub fn set(&self, index: usize, value: bool) {
        self.map[index].store(value, Ordering::Relaxed);
    }

    pub fn get(&self, index: usize) -> bool {
        self.map[index].load(Ordering::Relaxed)
    }

    /*pub fn swap(&self, index: usize, value: bool) -> bool {
        self.map[index].swap(value, Ordering::Relaxed)
    }*/

    pub fn reinitialise(&mut self, len: usize) {
        self.map.clear();
        self.map
            .extend(std::iter::repeat(false).map(Into::into).take(len));
    }

    pub fn slice(&self, range: Range<usize>) -> RelaxedAtomicBoolSlice {
        RelaxedAtomicBoolSlice {
            map: &self.map[range],
        }
    }

    pub fn iter(&self) -> impl '_ + Iterator<Item = bool> {
        self.map.iter().map(|b| b.load(Ordering::Relaxed))
    }
}

pub struct RelaxedAtomicBoolSlice<'a> {
    map: &'a [AtomicBool],
}

impl<'a> RelaxedAtomicBoolSlice<'a> {
    pub fn set(&self, index: usize, value: bool) {
        self.map[index].store(value, Ordering::Relaxed);
    }

    /*pub fn get(&self, index: usize) -> bool {
        self.map[index].load(Ordering::Relaxed)
    }*/
}

impl<Graph: GraphBase> DijkstraTargetMap<Graph> for RelaxedAtomicBoolVec {
    fn is_target(&self, node_index: Graph::NodeIndex) -> bool {
        self.get(node_index.as_usize())
    }
}

pub struct GraphMatchingNodeMap {
    node_id_map: Vec<Vec<usize>>,
    current_node_id: usize,
}

impl GraphMatchingNodeMap {
    pub fn new<Graph: StaticBigraph>(graph: &Graph) -> Self {
        Self {
            node_id_map: vec![Vec::new(); graph.node_count()],
            current_node_id: 0,
        }
    }

    pub fn get_or_create_node_indexes<
        EdgeData: BidirectedData + Eq,
        Graph: StaticEdgeCentricBigraph<EdgeData = EdgeData>,
    >(
        &mut self,
        graph: &Graph,
        node_index: Graph::NodeIndex,
    ) -> &[usize] {
        let result = &mut self.node_id_map[node_index.as_usize()];
        if result.is_empty() {
            for _ in 0..compute_eulerian_superfluous_out_biedges(graph, node_index).abs() {
                result.push(self.current_node_id);
                self.current_node_id += 1;
            }

            self.node_id_map[graph.mirror_node(node_index).unwrap().as_usize()] = result.clone();
        }

        // Retrieve the value again for lifetime reasons.
        let result = &self.node_id_map[node_index.as_usize()];
        debug_assert!(!result.is_empty());
        result
    }

    pub fn get_node_indexes<
        OptionalNodeIndex: OptionalGraphIndex<NodeIndex>,
        NodeIndex: GraphIndex<OptionalNodeIndex>,
    >(
        &self,
        node_index: NodeIndex,
    ) -> &[usize] {
        let result = &self.node_id_map[node_index.as_usize()];
        debug_assert!(!result.is_empty());
        result
    }

    pub fn get_node_indexes_unchecked<
        OptionalNodeIndex: OptionalGraphIndex<NodeIndex>,
        NodeIndex: GraphIndex<OptionalNodeIndex>,
    >(
        &self,
        node_index: NodeIndex,
    ) -> &[usize] {
        &self.node_id_map[node_index.as_usize()]
    }

    pub fn node_count(&self) -> usize {
        self.current_node_id
    }
}

pub fn choose_in_node_from_iterator<
    EdgeData: BidirectedData + Eq,
    Graph: StaticEdgeCentricBigraph<EdgeData = EdgeData>, //, EdgeData = BidirectedGfaNodeData<EdgeData>>,
    InNodeIterator: Iterator<Item = Graph::NodeIndex>,
>(
    graph: &Graph,
    mut in_node_iterator: InNodeIterator,
    out_node_difference: isize,
    out_node: Graph::NodeIndex,
) -> Option<Graph::NodeIndex> {
    let mut in_node = Some(in_node_iterator.next().unwrap());
    if (in_node == Some(graph.mirror_node(out_node).unwrap()) && out_node_difference > -2)
        || in_node == Some(out_node)
    {
        // If the in_node is the mirror of the out_node, adding an edge would create a self loop that does not resolve anything regarding Eulerianess.
        // And if the in_node is the out_node, then we have a self-mirror, and again adding a self loop does not resolve anything regarding Eulerianess.
        if in_node == Some(graph.mirror_node(out_node).unwrap()) && out_node_difference > -2 {
            warn!("Adding expensive edges in a different order because chosen nodes {} and {} are mirrors of each other", in_node.unwrap().as_usize(), out_node.as_usize());
        } else if in_node == Some(out_node) {
            warn!("Adding expensive edges in a different order because chosen nodes are a self-mirror");
        }
        in_node = if let Some(in_node) = in_node_iterator.next() {
            Some(in_node)
        } else {
            debug_assert_ne!(compute_eulerian_superfluous_out_biedges(graph, out_node), 0);
            debug_assert_ne!(
                compute_eulerian_superfluous_out_biedges(graph, in_node.unwrap()),
                0
            );
            None
        };
    }
    in_node
}

/// The edge data of the graph to compute matchtigs on has to implement this.
pub trait MatchtigEdgeData<SequenceHandle>:
    DijkstraWeightedEdgeData<usize> + BidirectedData
{
    /// Returns true if the edge is a dummy edge.
    /// This is the case if the `dummy_id` given in [Self::new] is non-zero.
    fn is_dummy(&self) -> bool;

    /// Return true if the edge is an original edge.
    fn is_original(&self) -> bool {
        !self.is_dummy()
    }

    /// Returns true if this is the forwards variant of this edge,
    /// i.e. if the `sequence_handle` given in [Self::new] is the sequence belonging to this edge.
    /// This is the case if the `forwards` given in [Self::new] is `true`.
    fn is_forwards(&self) -> bool;

    /// Returns true if this is the backwards variant of this edge,
    /// i.e. if the `sequence_handle` given in [Self::new] is the reverse complement of the sequence belonging to this edge.
    /// This is the case if the `forwards` given in [Self::new] is `false`.
    fn is_backwards(&self) -> bool {
        !self.is_forwards()
    }

    /// Creates a new edge with the given properties.
    /// The given values must be returned by the respective functions of this trait,
    /// and the `weight` by the implementation of [DijkstraWeightedEdgeData](traitgraph_algo::dijkstra::DijkstraWeightedEdgeData) of this type.
    fn new(sequence_handle: SequenceHandle, forwards: bool, weight: usize, dummy_id: usize)
        -> Self;
}

pub fn debug_assert_graph_has_no_consecutive_dummy_edges<
    SequenceHandle,
    EdgeData: MatchtigEdgeData<SequenceHandle> + Eq,
    Graph: StaticEdgeCentricBigraph<EdgeData = EdgeData>,
>(
    graph: &Graph,
    k: usize,
) {
    if !cfg!(debug_assertions) {
        return;
    }

    let mut dummy_in_edges = Vec::new();
    let mut dummy_cheap_in_edges = Vec::new();
    let mut dummy_expensive_in_edges = Vec::new();
    let mut dummy_out_edges = Vec::new();
    let mut dummy_cheap_out_edges = Vec::new();
    let mut dummy_expensive_out_edges = Vec::new();
    for node_index in graph.node_indices() {
        dummy_in_edges.clear();
        dummy_cheap_in_edges.clear();
        dummy_expensive_in_edges.clear();
        for in_neighbor in graph.in_neighbors(node_index) {
            let edge_data = graph.edge_data(in_neighbor.edge_id);
            if edge_data.is_dummy() {
                dummy_in_edges.push(in_neighbor.edge_id);
                if edge_data.weight() >= k {
                    dummy_expensive_in_edges.push(in_neighbor.edge_id);
                } else {
                    dummy_cheap_in_edges.push((in_neighbor.edge_id, edge_data.weight()));
                }
            }
        }

        if !dummy_in_edges.is_empty() {
            dummy_out_edges.clear();
            dummy_cheap_out_edges.clear();
            dummy_expensive_out_edges.clear();
            for out_neighbor in graph.out_neighbors(node_index) {
                let edge_data = graph.edge_data(out_neighbor.edge_id);
                if edge_data.is_dummy() {
                    dummy_out_edges.push(out_neighbor.edge_id);
                    if edge_data.weight() >= k {
                        dummy_expensive_out_edges.push(out_neighbor.edge_id);
                    } else {
                        dummy_cheap_out_edges.push((out_neighbor.edge_id, edge_data.weight()));
                    }
                }
            }

            if !dummy_out_edges.is_empty() {
                if dummy_in_edges.len() == 1 && dummy_out_edges.len() == 1 {
                    if graph
                        .mirror_edge_edge_centric(*dummy_in_edges.first().unwrap())
                        .expect("Edge has no mirror")
                        == *dummy_out_edges.first().unwrap()
                    {
                        debug_assert_ne!(
                            graph.edge_data(*dummy_in_edges.first().unwrap()).weight(),
                            0
                        );
                        continue;
                    }
                }

                panic!("Found node with both an incoming and an outgoing dummy edge. This node is {}a self-mirror. Dummy in-edges: cheap: {:?} expensive: {:?}, dummy out-edges: cheap: {:?} expensive: {:?}",
                           if graph.is_self_mirror_node(node_index) { "" } else { "NOT " },
                           dummy_cheap_in_edges, dummy_expensive_in_edges, dummy_cheap_out_edges, dummy_expensive_out_edges);
            }
        }
    }
}

pub fn make_graph_eulerian_with_breaking_edges<
    NodeIndex: GraphIndex<OptionalNodeIndex>,
    OptionalNodeIndex: OptionalGraphIndex<NodeIndex>,
    SequenceHandle: Default + Clone,
    EdgeData: BidirectedData + Eq + MatchtigEdgeData<SequenceHandle> + Clone,
    Graph: DynamicEdgeCentricBigraph<
        NodeIndex = NodeIndex,
        OptionalNodeIndex = OptionalNodeIndex,
        EdgeData = EdgeData,
    >,
>(
    graph: &mut Graph,
    dummy_sequence: SequenceHandle,
    dummy_edge_id: &mut usize,
    k: usize,
) {
    let nodes_and_differences = find_non_eulerian_binodes_with_differences(graph);
    let mut out_node_differences = BTreeMap::new();
    let mut in_node_differences = BTreeMap::new();
    out_node_differences.extend(
        nodes_and_differences
            .iter()
            .filter(|(_, difference)| *difference < 0)
            .copied()
            .map(|(e, d)| (std::cmp::Reverse(e), d)),
    );
    in_node_differences.extend(
        nodes_and_differences
            .iter()
            .filter(|(_, difference)| *difference > 0)
            .copied(),
    );
    let self_mirror_node_differences: Vec<_> = nodes_and_differences
        .iter()
        .filter_map(|&(node, difference)| if difference == 0 { Some(node) } else { None })
        .collect();
    info!(
        "Adding edges for {} unmatched in_nodes, {} unmatched out_nodes and {} unmatched self_mirror_nodes",
        in_node_differences.len(),
        out_node_differences.len(),
        self_mirror_node_differences.len(),
    );

    let total_out_node_difference: isize = out_node_differences.values().copied().sum();
    let total_in_node_difference: isize = in_node_differences.values().copied().sum();
    let out_node_difference_one_count = out_node_differences.values().filter(|d| **d == -1).count();
    let in_node_difference_one_count = in_node_differences.values().filter(|d| **d == 1).count();
    let out_node_difference_two_count = out_node_differences.values().filter(|d| **d == -2).count();
    let in_node_difference_two_count = in_node_differences.values().filter(|d| **d == 2).count();
    let out_node_difference_three_count =
        out_node_differences.values().filter(|d| **d == -3).count();
    let in_node_difference_three_count = in_node_differences.values().filter(|d| **d == 3).count();
    let out_node_difference_four_count =
        out_node_differences.values().filter(|d| **d == -4).count();
    let in_node_difference_four_count = in_node_differences.values().filter(|d| **d == 4).count();
    let out_node_difference_more_count = out_node_differences.values().filter(|d| **d < -4).count();
    let in_node_difference_more_count = in_node_differences.values().filter(|d| **d > 4).count();
    debug_assert_eq!(-total_out_node_difference, total_in_node_difference);
    debug_assert_eq!(
        (total_in_node_difference + self_mirror_node_differences.len() as isize) % 2,
        0
    );
    info!(
        "{} edges need to be added in total",
        (total_in_node_difference + self_mirror_node_differences.len() as isize) / 2
    );
    debug_assert_eq!(out_node_difference_one_count, in_node_difference_one_count);
    debug_assert_eq!(out_node_difference_two_count, in_node_difference_two_count);
    debug_assert_eq!(
        out_node_difference_three_count,
        in_node_difference_three_count
    );
    debug_assert_eq!(
        out_node_difference_four_count,
        in_node_difference_four_count
    );
    debug_assert_eq!(
        out_node_difference_more_count,
        in_node_difference_more_count
    );
    debug_assert_eq!(out_node_difference_more_count, 0);
    info!(
        "{}/{}/{}/{} binodes that are not self-mirrors have difference 1/2/3/4",
        out_node_difference_one_count,
        out_node_difference_two_count,
        out_node_difference_three_count,
        out_node_difference_four_count
    );

    for pair in &self_mirror_node_differences.into_iter().chunks(2) {
        let pair: Vec<_> = pair.collect();
        if pair.len() == 2 {
            let out_node = pair[0];
            let in_node = pair[1];
            let mirror_out_node = graph.mirror_node(in_node).unwrap();
            let mirror_in_node = graph.mirror_node(out_node).unwrap();

            let sequence = dummy_sequence.clone();
            *dummy_edge_id += 1;
            let edge_data = EdgeData::new(sequence, true, k, *dummy_edge_id);
            graph.add_edge(out_node, in_node, edge_data.clone());
            graph.add_edge(mirror_out_node, mirror_in_node, edge_data.mirror());
        } else {
            debug_assert_eq!(pair.len(), 1);
            let (&in_node, difference) = in_node_differences.iter_mut().next().expect(
                "Have an uneven number of self-mirrors, but no other nodes with missing in edges.",
            );
            debug_assert_ne!(in_node, graph.mirror_node(in_node).unwrap());
            debug_assert!(*difference > 0);

            let out_node = pair[0];
            let mirror_out_node = graph.mirror_node(in_node).unwrap();
            let mirror_in_node = graph.mirror_node(out_node).unwrap();

            let sequence = dummy_sequence.clone();
            *dummy_edge_id += 1;
            let edge_data = EdgeData::new(sequence, true, k, *dummy_edge_id);
            graph.add_edge(out_node, in_node, edge_data.clone());
            graph.add_edge(mirror_out_node, mirror_in_node, edge_data.mirror());

            *difference -= 1;
            if *difference == 0 {
                in_node_differences.remove(&in_node);
                out_node_differences
                    .remove(&std::cmp::Reverse(graph.mirror_node(in_node).unwrap()))
                    .expect("Mirror of in_node not found");
            } else {
                *out_node_differences
                    .get_mut(&std::cmp::Reverse(graph.mirror_node(in_node).unwrap()))
                    .expect("Mirror of in_node not found") += 1;
            }
        }
    }

    while let Some(out_node) = out_node_differences.keys().copied().next() {
        /*let current_out_node_difference: isize = out_node_differences.iter().map(|(_, d)| *d).sum();
        let current_in_node_difference: isize = in_node_differences.iter().map(|(_, d)| *d).sum();
        debug_assert_eq!(-current_out_node_difference, current_in_node_difference);*/

        let out_node = out_node.0;
        let out_node_difference = out_node_differences
            .get_mut(&std::cmp::Reverse(out_node))
            .unwrap();
        let in_node = if let Some(in_node) = choose_in_node_from_iterator(
            graph,
            in_node_differences.keys().copied(),
            *out_node_difference,
            out_node,
        ) {
            in_node
        } else {
            let current_out_node_difference: isize = out_node_differences.values().copied().sum();
            let current_in_node_difference: isize = in_node_differences.values().copied().sum();
            debug_assert_eq!(-current_out_node_difference, current_in_node_difference);
            debug_assert_eq!(
                (
                    out_node_differences.values().copied().sum::<isize>(),
                    in_node_differences.values().copied().sum::<isize>()
                ),
                (-1, 1)
            );
            panic!("No further in_nodes left");
        };

        /*info!(
            "in_node {} and out_node {}",
            in_node.as_usize(),
            out_node.as_usize()
        );*/
        let is_mirror = in_node == graph.mirror_node(out_node).unwrap();
        debug_assert_ne!(
            in_node, out_node,
            "This part of the algorithm was not designed for self mirrors"
        );
        debug_assert!(compute_eulerian_superfluous_out_biedges(graph, out_node) < 0);
        debug_assert!(compute_eulerian_superfluous_out_biedges(graph, in_node) > 0);

        let mirror_out_node = graph.mirror_node(in_node).unwrap();
        let mirror_in_node = graph.mirror_node(out_node).unwrap();

        let sequence = dummy_sequence.clone();
        *dummy_edge_id += 1;
        let edge_data = EdgeData::new(sequence, true, k, *dummy_edge_id);

        graph.add_edge(out_node, in_node, edge_data.clone());
        graph.add_edge(mirror_out_node, mirror_in_node, edge_data.mirror());

        let in_node_difference = in_node_differences.get_mut(&in_node).unwrap();
        debug_assert_ne!(*out_node_difference, 0);
        debug_assert_ne!(*in_node_difference, 0);
        *out_node_difference += 1;
        *in_node_difference -= 1;

        if is_mirror {
            debug_assert_ne!(*out_node_difference, 0);
            debug_assert_ne!(*in_node_difference, 0);
        }

        let remove_out_node = *out_node_difference == 0;
        let remove_in_node = *in_node_difference == 0;
        if remove_out_node {
            debug_assert_eq!(compute_eulerian_superfluous_out_biedges(graph, out_node), 0);
            out_node_differences.remove(&std::cmp::Reverse(out_node));
        } else {
            debug_assert!(
                compute_eulerian_superfluous_out_biedges(graph, out_node) != 0 || is_mirror
            );
        }
        if remove_in_node {
            debug_assert_eq!(compute_eulerian_superfluous_out_biedges(graph, in_node), 0);
            in_node_differences.remove(&in_node);
        } else {
            debug_assert!(
                compute_eulerian_superfluous_out_biedges(graph, in_node) != 0 || is_mirror
            );
        }

        if let Some(mirror_out_node_difference) =
            out_node_differences.get_mut(&std::cmp::Reverse(mirror_out_node))
        {
            *mirror_out_node_difference += 1;
            if *mirror_out_node_difference == 0 {
                debug_assert_eq!(
                    compute_eulerian_superfluous_out_biedges(graph, mirror_out_node),
                    0
                );
                debug_assert!(remove_in_node || is_mirror);
                out_node_differences.remove(&std::cmp::Reverse(mirror_out_node));
            } else {
                debug_assert_ne!(
                    compute_eulerian_superfluous_out_biedges(graph, mirror_out_node),
                    0
                );
                debug_assert!(!remove_in_node);
            }
        }
        if let Some(mirror_in_node_difference) = in_node_differences.get_mut(&mirror_in_node) {
            *mirror_in_node_difference -= 1;
            if *mirror_in_node_difference == 0 {
                debug_assert_eq!(
                    compute_eulerian_superfluous_out_biedges(graph, mirror_in_node),
                    0
                );
                debug_assert!(remove_out_node || is_mirror);
                in_node_differences.remove(&mirror_in_node);
            } else {
                debug_assert_ne!(
                    compute_eulerian_superfluous_out_biedges(graph, mirror_in_node),
                    0
                );
                debug_assert!(!remove_out_node);
            }
        }
    }

    debug_assert!(out_node_differences.is_empty());
    debug_assert!(in_node_differences.is_empty());
}

/// Convenience method for [write_duplication_bitvector].
pub fn write_duplication_bitvector_to_file<
    'ws,
    SequenceHandle,
    EdgeData: MatchtigEdgeData<SequenceHandle>,
    Graph: ImmutableGraphContainer<EdgeData = EdgeData>,
    Walk: 'ws + EdgeWalk<Graph, Subwalk>,
    Subwalk: EdgeWalk<Graph, Subwalk> + ?Sized,
    WalkSource: 'ws + IntoIterator<Item = &'ws Walk>,
>(
    graph: &Graph,
    walks: WalkSource,
    path: impl AsRef<Path>,
) -> Result<(), std::io::Error> {
    write_duplication_bitvector(graph, walks, &mut BufWriter::new(File::create(path)?))
}

/// Write a bitvector in ASCII format for each walk.
/// The bitvector contains a 1 for each original kmer, and a 0 for each duplicate kmer.
/// The bitvectors of different walks are separated by newlines.
pub fn write_duplication_bitvector<
    'ws,
    SequenceHandle,
    EdgeData: MatchtigEdgeData<SequenceHandle>,
    Graph: ImmutableGraphContainer<EdgeData = EdgeData>,
    Walk: 'ws + EdgeWalk<Graph, Subwalk>,
    Subwalk: EdgeWalk<Graph, Subwalk> + ?Sized,
    WalkSource: 'ws + IntoIterator<Item = &'ws Walk>,
>(
    graph: &Graph,
    walks: WalkSource,
    writer: &mut impl std::io::Write,
) -> Result<(), std::io::Error> {
    for walk in walks.into_iter() {
        if walk.is_empty() {
            panic!("Found empty walk when writing duplication bitvector");
        }

        for edge in walk.iter() {
            let edge_data = graph.edge_data(*edge);
            let character = if edge_data.is_original() { '1' } else { '0' };

            for _ in 0..edge_data.weight() {
                write!(writer, "{character}")?;
            }
        }

        writeln!(writer)?;
    }

    Ok(())
}

fn append_to_filename(path: PathBuf, ext: impl AsRef<OsStr>) -> PathBuf {
    let mut os_string: OsString = path.into();
    os_string.push(ext.as_ref());
    os_string.into()
}

#[cfg(test)]
mod tests {
    use crate::implementation::{make_graph_eulerian_with_breaking_edges, MatchtigEdgeData};
    use genome_graph::bigraph::implementation::node_bigraph_wrapper::NodeBigraphWrapper;
    use genome_graph::bigraph::interface::dynamic_bigraph::DynamicBigraph;
    use genome_graph::bigraph::interface::static_bigraph::StaticBigraphFromDigraph;
    use genome_graph::bigraph::interface::BidirectedData;
    use genome_graph::bigraph::traitgraph::implementation::petgraph_impl::PetGraph;
    use genome_graph::bigraph::traitgraph::interface::MutableGraphContainer;
    use traitgraph_algo::dijkstra::DijkstraWeightedEdgeData;

    #[derive(Debug, Clone, Eq, PartialEq)]
    struct TestEdgeData {
        sequence_handle: usize,
        forwards: bool,
        weight: usize,
        dummy_id: usize,
    }

    impl DijkstraWeightedEdgeData<usize> for TestEdgeData {
        fn weight(&self) -> usize {
            self.weight
        }
    }

    impl BidirectedData for TestEdgeData {
        fn mirror(&self) -> Self {
            let mut result = self.clone();
            result.forwards = !result.forwards;
            result
        }
    }

    impl MatchtigEdgeData<usize> for TestEdgeData {
        fn is_dummy(&self) -> bool {
            self.dummy_id > 0
        }

        fn is_forwards(&self) -> bool {
            self.forwards
        }

        fn new(sequence_handle: usize, forwards: bool, weight: usize, dummy_id: usize) -> Self {
            Self {
                sequence_handle,
                forwards,
                weight,
                dummy_id,
            }
        }
    }

    #[test]
    fn test_make_graph_eulerian_with_breaking_edges_mirror_nodes() {
        let mut graph = NodeBigraphWrapper::new(PetGraph::new());

        let nodes: Vec<_> = (0..8).map(|_| graph.add_node(())).collect();
        graph.set_mirror_nodes(nodes[0], nodes[1]);
        graph.set_mirror_nodes(nodes[2], nodes[2]);
        graph.set_mirror_nodes(nodes[3], nodes[3]);
        graph.set_mirror_nodes(nodes[4], nodes[5]);
        graph.set_mirror_nodes(nodes[6], nodes[6]);
        graph.set_mirror_nodes(nodes[7], nodes[7]);

        graph.add_edge(nodes[0], nodes[3], TestEdgeData::new(1, true, 0, 1));
        graph.add_edge(nodes[3], nodes[1], TestEdgeData::new(1, false, 0, 1));
        graph.add_edge(nodes[2], nodes[0], TestEdgeData::new(2, true, 0, 2));
        graph.add_edge(nodes[1], nodes[2], TestEdgeData::new(2, false, 0, 2));
        graph.add_edge(nodes[6], nodes[4], TestEdgeData::new(3, true, 0, 3));
        graph.add_edge(nodes[5], nodes[6], TestEdgeData::new(3, false, 0, 3));
        graph.add_edge(nodes[7], nodes[4], TestEdgeData::new(4, true, 0, 4));
        graph.add_edge(nodes[5], nodes[7], TestEdgeData::new(4, false, 0, 4));

        make_graph_eulerian_with_breaking_edges(&mut graph, 0, &mut 5, 4);
        println!("{graph:?}");
    }
}
