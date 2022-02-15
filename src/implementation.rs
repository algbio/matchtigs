// Some of the if branches might be more readable if not collapsed
#![allow(clippy::collapsible_if)]
// This lint never produces correct suggestions in our case.
#![allow(clippy::mutex_atomic)]

use atomic_counter::{AtomicCounter, RelaxedCounter};
use genome_graph::bigraph::algo::eulerian::{
    compute_eulerian_superfluous_out_biedges,
    compute_minimum_bidirected_eulerian_cycle_decomposition, decomposes_into_eulerian_bicycles,
    find_non_eulerian_binodes_with_differences,
};
use genome_graph::bigraph::algo::walk_cover::arbitrary_biwalk_cover;
use genome_graph::bigraph::interface::dynamic_bigraph::DynamicEdgeCentricBigraph;
use genome_graph::bigraph::interface::static_bigraph::StaticBigraph;
use genome_graph::bigraph::interface::static_bigraph::StaticEdgeCentricBigraph;
use genome_graph::bigraph::interface::BidirectedData;
use genome_graph::bigraph::traitgraph::index::{GraphIndex, OptionalGraphIndex};
use genome_graph::bigraph::traitgraph::interface::{GraphBase, StaticGraph};
use genome_graph::bigraph::traitgraph::traitsequence::interface::Sequence;
use genome_graph::bigraph::traitgraph::walks::{EdgeWalk, VecEdgeWalk};
use itertools::Itertools;
use simplelog::{ColorChoice, CombinedLogger, Config, LevelFilter, TermLogger, TerminalMode};
use std::collections::{BTreeMap, HashMap};
use std::convert::TryInto;
use std::fs::File;
use std::io::Write;
use std::io::{BufRead, BufReader, BufWriter};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;
use traitgraph_algo::dijkstra::{
    DefaultDijkstra, Dijkstra, DijkstraHeap, DijkstraTargetMap, DijkstraWeightedEdgeData,
    NodeWeightArray,
};

const TARGET_DIJKSTRA_BLOCK_TIME: f32 = 5.0; // seconds

pub fn initialise_logging() {
    CombinedLogger::init(vec![TermLogger::new(
        if cfg!(debug_assertions) {
            LevelFilter::Trace
        } else {
            LevelFilter::Info
        },
        Config::default(),
        TerminalMode::Mixed,
        ColorChoice::Auto,
    )])
    .unwrap();

    info!("Logging initialised successfully");
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

/// Compute pathtigs for the given graph.
/// This is a heuristically small set of edge-disjoint paths, similar to simplitigs and UST-tigs.
pub fn compute_pathtigs<
    EdgeData: BidirectedData + Eq,
    Graph: StaticEdgeCentricBigraph<EdgeData = EdgeData>,
>(
    graph: &Graph,
) -> Vec<VecEdgeWalk<Graph>> {
    info!("Computing pathtigs");
    let pathtigs = arbitrary_biwalk_cover(graph);
    info!("Found {} pathtigs", pathtigs.len());
    pathtigs
}

pub trait MatchtigEdgeData<SequenceHandle>:
    DijkstraWeightedEdgeData<usize> + BidirectedData
{
    fn is_dummy(&self) -> bool;

    fn is_original(&self) -> bool {
        !self.is_dummy()
    }

    fn is_forwards(&self) -> bool;

    fn is_backwards(&self) -> bool {
        !self.is_forwards()
    }

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

    let total_out_node_difference: isize = out_node_differences.iter().map(|(_, d)| *d).sum();
    let total_in_node_difference: isize = in_node_differences.iter().map(|(_, d)| *d).sum();
    let out_node_difference_one_count = out_node_differences
        .iter()
        .filter(|(_, d)| **d == -1)
        .count();
    let in_node_difference_one_count = in_node_differences.iter().filter(|(_, d)| **d == 1).count();
    let out_node_difference_two_count = out_node_differences
        .iter()
        .filter(|(_, d)| **d == -2)
        .count();
    let in_node_difference_two_count = in_node_differences.iter().filter(|(_, d)| **d == 2).count();
    let out_node_difference_three_count = out_node_differences
        .iter()
        .filter(|(_, d)| **d == -3)
        .count();
    let in_node_difference_three_count =
        in_node_differences.iter().filter(|(_, d)| **d == 3).count();
    let out_node_difference_four_count = out_node_differences
        .iter()
        .filter(|(_, d)| **d == -4)
        .count();
    let in_node_difference_four_count =
        in_node_differences.iter().filter(|(_, d)| **d == 4).count();
    let out_node_difference_more_count = out_node_differences
        .iter()
        .filter(|(_, d)| **d < -4)
        .count();
    let in_node_difference_more_count = in_node_differences.iter().filter(|(_, d)| **d > 4).count();
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
            let current_out_node_difference: isize =
                out_node_differences.iter().map(|(_, d)| *d).sum();
            let current_in_node_difference: isize =
                in_node_differences.iter().map(|(_, d)| *d).sum();
            debug_assert_eq!(-current_out_node_difference, current_in_node_difference);
            debug_assert_eq!(
                (
                    out_node_differences.iter().map(|(_, d)| *d).sum::<isize>(),
                    in_node_differences.iter().map(|(_, d)| *d).sum::<isize>()
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

/// Computes greedy matchtigs in the given graph.
pub fn compute_greedytigs<
    NodeIndex: GraphIndex<OptionalNodeIndex> + Send + Sync,
    OptionalNodeIndex: OptionalGraphIndex<NodeIndex>,
    SequenceHandle: Default + Clone,
    EdgeData: BidirectedData + Eq + MatchtigEdgeData<SequenceHandle> + Clone,
    Graph: DynamicEdgeCentricBigraph<
            NodeIndex = NodeIndex,
            OptionalNodeIndex = OptionalNodeIndex,
            EdgeData = EdgeData,
        > + Send
        + Sync,
    DijkstraHeapType: DijkstraHeap<usize, Graph::NodeIndex>,
    DijkstraNodeWeightArray: NodeWeightArray<usize>,
>(
    graph: &mut Graph,
    threads: usize,
    k: usize,
) -> Vec<VecEdgeWalk<Graph>> {
    info!("Collecting nodes with missing incoming or outgoing edges");
    let mut out_nodes = Vec::new(); // Misses outgoing edges
    let mut in_node_count = 0;
    let in_node_map = RelaxedAtomicBoolVec::new(graph.node_count());
    let mut node_multiplicities = vec![0; graph.node_count()];
    let mut unbalanced_self_mirror_count = 0;

    for node_index in graph.node_indices() {
        let diff = compute_eulerian_superfluous_out_biedges(graph, node_index);
        if graph.is_self_mirror_node(node_index) && diff != 0 {
            in_node_count += 1;
            in_node_map.set(node_index.as_usize(), true);
            node_multiplicities[node_index.as_usize()] = diff;
            out_nodes.push(node_index);
            unbalanced_self_mirror_count += 1;
        } else if diff > 0 {
            in_node_count += 1;
            in_node_map.set(node_index.as_usize(), true);
            node_multiplicities[node_index.as_usize()] = diff;
        } else if diff < 0 {
            out_nodes.push(node_index);
            node_multiplicities[node_index.as_usize()] = diff;
        }
    }

    info!(
        "Found {} nodes with missing outgoing edges",
        out_nodes.len()
    );
    info!("Found {} nodes with missing incoming edges", in_node_count);
    info!(
        "Of those there are {} self-mirrors",
        unbalanced_self_mirror_count
    );

    info!("Computing shortest paths between nodes with missing outgoing and nodes with missing incoming edges");

    let dummy_sequence = SequenceHandle::default();
    let mut dummy_edge_id = {
        info!(
            "Using {} threads to run ~{} dijkstras",
            threads,
            out_nodes.len()
        );
        let start = Instant::now();

        let locked_node_multiplicities: Vec<_> = node_multiplicities
            .iter()
            .copied()
            .map(Mutex::new)
            .collect();
        let executed_dijkstras = RelaxedCounter::default();

        #[allow(clippy::too_many_arguments)]
        fn compute_dijkstras<
            EdgeData: DijkstraWeightedEdgeData<usize>,
            Graph: StaticBigraph<EdgeData = EdgeData>,
            DijkstraNodeWeightArray: NodeWeightArray<usize>,
            DijkstraHeapType: DijkstraHeap<usize, Graph::NodeIndex>,
        >(
            graph: &Graph,
            dijkstra: &mut Dijkstra<Graph, usize, DijkstraNodeWeightArray, DijkstraHeapType>,
            distances: &mut Vec<(Graph::NodeIndex, usize)>,
            shortest_paths: &mut Vec<(Graph::NodeIndex, Graph::NodeIndex, usize)>,
            out_nodes: &[Graph::NodeIndex],
            out_nodes_offset: usize,
            out_nodes_limit: usize,
            in_node_map: &RelaxedAtomicBoolVec,
            locked_node_multiplicities: &[Mutex<isize>],
            k: usize,
            executed_dijkstras: &RelaxedCounter,
            output: bool,
            output_step: usize,
            total_dijkstras: usize,
        ) {
            let mut last_output = 0;
            let local_out_nodes = &out_nodes[out_nodes_offset..out_nodes_limit];

            for (i, &out_node) in local_out_nodes.iter().enumerate() {
                let out_node_is_self_mirror = graph.is_self_mirror_node(out_node);
                let out_node_mirror = graph.mirror_node(out_node).unwrap();
                // The linter suggestion to use atomic values for locked_node_multiplicities here is spurious,
                // as we explicitly want to update a set of up to four values atomically later on.
                let out_node_multiplicity_lock = locked_node_multiplicities
                    [out_node_mirror.as_usize()]
                .lock()
                .unwrap();
                let mut out_node_multiplicity: isize = *out_node_multiplicity_lock;
                drop(out_node_multiplicity_lock);

                debug_assert!(
                    (0..=4).contains(&out_node_multiplicity),
                    "out_node_multiplicity = {}, out_node_is_self_mirror = {}",
                    out_node_multiplicity,
                    out_node_is_self_mirror
                );

                if out_node_multiplicity == 0 {
                    continue;
                }

                while out_node_multiplicity > 0 {
                    let target_amount = (out_node_multiplicity + 1).try_into().unwrap();
                    dijkstra.shortest_path_lens(
                        graph,
                        out_node,
                        in_node_map,
                        target_amount,
                        k - 1,
                        true,
                        distances,
                    );

                    if distances.is_empty() {
                        // If no in_nodes are reachable, fix this with breaking dummy arcs in post-processing
                        break;
                    }

                    let abort_after_this = distances.len() < target_amount;

                    for &(in_node, distance) in distances.iter() {
                        let mut is_self_mirror_edge = false;
                        if in_node == out_node_mirror {
                            if out_node_multiplicity < 2 {
                                continue;
                            } else {
                                is_self_mirror_edge = true;
                            }
                        };
                        let is_self_mirror_edge = is_self_mirror_edge;
                        debug_assert!(!is_self_mirror_edge || !out_node_is_self_mirror);

                        let in_node_mirror = graph.mirror_node(in_node).unwrap();
                        let in_node_is_self_mirror = graph.is_self_mirror_node(in_node);
                        debug_assert!(!is_self_mirror_edge || !in_node_is_self_mirror);

                        let mut lock_indices = vec![out_node.as_usize()];
                        if !out_node_is_self_mirror {
                            lock_indices.push(out_node_mirror.as_usize());
                        }
                        if !is_self_mirror_edge {
                            lock_indices.push(in_node.as_usize());
                            if !in_node_is_self_mirror {
                                lock_indices.push(in_node_mirror.as_usize());
                            }
                        }
                        let lock_indices = lock_indices;

                        let lock_permutation = permutation::sort(lock_indices.as_slice());
                        let mut locks = vec![None, None, None, None];

                        for lock_rank in 0..lock_indices.len() {
                            let lock_index = lock_permutation.apply_inv_idx(lock_rank);

                            // The linter suggestion to use atomic values for locked_node_multiplicities here is spurious,
                            // as we explicitly want to update a set of up to four values atomically.
                            locks[lock_index] = Some(
                                locked_node_multiplicities[lock_indices[lock_index]]
                                    .lock()
                                    .unwrap(),
                            );
                        }

                        let mut locks: Vec<_> = locks
                            .into_iter()
                            .take(lock_indices.len())
                            .map(|o| o.unwrap())
                            .collect();
                        let in_node_lock_offset = if out_node_is_self_mirror { 1 } else { 2 };
                        let multiplicity_reduction = if is_self_mirror_edge { 2 } else { 1 };

                        if out_node_is_self_mirror {
                            debug_assert!(*locks[0] >= 0);
                            debug_assert!(*locks[0] <= 1);
                            out_node_multiplicity = *locks[0];
                        } else {
                            debug_assert!(*locks[0] <= 0);
                            debug_assert!(*locks[0] >= -4);
                            debug_assert_eq!(*locks[0], -*locks[1]);
                            out_node_multiplicity = -*locks[0];
                        }

                        if out_node_multiplicity == 0 {
                            break;
                        }

                        if !is_self_mirror_edge {
                            if in_node_is_self_mirror {
                                debug_assert!(*locks[in_node_lock_offset] >= 0,
                                        "*locks[in_node_lock_offset] = {}, in_node_lock_offset = {}, out_node_is_self_mirror = {}, in_node_is_self_mirror = {}, is_self_mirror_edge = {}, lock_indices = {:?}, lock_permutation = {:?}, locks = {:?}",
                                        *locks[in_node_lock_offset],
                                        in_node_lock_offset,
                                        out_node_is_self_mirror,
                                        in_node_is_self_mirror,
                                        is_self_mirror_edge,
                                        lock_indices,
                                        lock_permutation,
                                        locks.iter().map(|l| **l).collect::<Vec<_>>());
                                debug_assert!(*locks[in_node_lock_offset] <= 1,
                                        "*locks[in_node_lock_offset] = {}, in_node_lock_offset = {}, out_node_is_self_mirror = {}, in_node_is_self_mirror = {}, is_self_mirror_edge = {}, lock_indices = {:?}, lock_permutation = {:?}, locks = {:?}",
                                        *locks[in_node_lock_offset],
                                        in_node_lock_offset,
                                        out_node_is_self_mirror,
                                        in_node_is_self_mirror,
                                        is_self_mirror_edge,
                                        lock_indices,
                                        lock_permutation,
                                        locks.iter().map(|l| **l).collect::<Vec<_>>());
                            } else {
                                debug_assert!(*locks[in_node_lock_offset] >= 0,
                                        "*locks[in_node_lock_offset] = {}, in_node_lock_offset = {}, in_node_is_self_mirror = {}, is_self_mirror_edge = {}, lock_permutation = {:?}, locks = {:?}",
                                        *locks[in_node_lock_offset],
                                        in_node_lock_offset,
                                        in_node_is_self_mirror,
                                        is_self_mirror_edge,
                                        lock_permutation,
                                        locks.iter().map(|l| **l).collect::<Vec<_>>());
                                debug_assert!(*locks[in_node_lock_offset] <= 4);
                                debug_assert_eq!(
                                    *locks[in_node_lock_offset],
                                    -*locks[in_node_lock_offset + 1]
                                );
                            }

                            let in_node_multiplicity = *locks[in_node_lock_offset];
                            if in_node_multiplicity == 0 {
                                in_node_map.set(in_node.as_usize(), false);
                                continue;
                            }
                        }

                        shortest_paths.push((out_node, in_node, distance));

                        if out_node_is_self_mirror {
                            *locks[0] -= 1;
                            debug_assert!(*locks[0] >= 0);
                            debug_assert!(*locks[0] <= 1);
                        } else {
                            *locks[0] += multiplicity_reduction;
                            *locks[1] -= multiplicity_reduction;
                            debug_assert!(*locks[0] <= 0);
                            debug_assert!(*locks[0] >= -4);
                            debug_assert_eq!(*locks[0], -*locks[1]);
                        }
                        out_node_multiplicity = -(*locks[0]);

                        if !is_self_mirror_edge {
                            *locks[in_node_lock_offset] -= 1;

                            if in_node_is_self_mirror {
                                debug_assert!(*locks[in_node_lock_offset] >= 0);
                                debug_assert!(*locks[in_node_lock_offset] <= 1);
                            } else {
                                *locks[in_node_lock_offset + 1] += 1;
                                debug_assert!(*locks[in_node_lock_offset] >= 0);
                                debug_assert!(*locks[in_node_lock_offset] <= 4);
                                debug_assert_eq!(
                                    *locks[in_node_lock_offset],
                                    -*locks[in_node_lock_offset + 1]
                                );
                            }
                        }

                        if out_node_multiplicity == 0 {
                            in_node_map.set(out_node_mirror.as_usize(), false);
                        }

                        if !is_self_mirror_edge {
                            if *locks[in_node_lock_offset] == 0 {
                                in_node_map.set(in_node.as_usize(), false);
                            }
                        }
                    }

                    if abort_after_this {
                        break;
                    }
                }

                executed_dijkstras.inc();
                if output && executed_dijkstras.get() - last_output > output_step {
                    last_output += output_step;
                    println!(
                        "{}%, ~{} total shortest paths",
                        last_output * 100 / total_dijkstras,
                        (shortest_paths.len() as f64 * total_dijkstras as f64 / i as f64) as u64,
                    );
                }
            }
        }

        let dijkstra_start = Instant::now();

        let results = Mutex::new(Vec::new());
        // The linter suggestion to use an atomic value instead of the mutex lock here does not work,
        // as we increment the value based on its current value.
        let offset = Mutex::new(0);
        let shared_graph = &*graph;
        crossbeam::scope(|scope| {
            let mut thread_handles = Vec::new();

            info!("Creating dijkstra threads");

            for _ in 0..threads {
                thread_handles.push(scope.spawn(|_| {
                    let graph = shared_graph;
                    let mut dijkstra =
                        Dijkstra::<_, _, DijkstraNodeWeightArray, DijkstraHeapType>::new(graph);
                    let mut distances = Vec::new();
                    let mut shortest_paths = Vec::new();
                    let mut chunk_size = 1024;

                    loop {
                        let (current_offset, current_limit) = {
                            let mut offset = offset.lock().unwrap();
                            let current_offset = *offset;
                            let remaining_nodes = out_nodes.len() - *offset;

                            if remaining_nodes == 0 {
                                break;
                            }

                            chunk_size = chunk_size
                                .min(remaining_nodes / threads)
                                .max(10)
                                .max(chunk_size / 10)
                                .min(remaining_nodes);
                            let current_limit = current_offset + chunk_size;
                            *offset = current_limit;
                            (current_offset, current_limit)
                        };

                        let dijkstra_start = Instant::now();
                        compute_dijkstras(
                            graph,
                            &mut dijkstra,
                            &mut distances,
                            &mut shortest_paths,
                            &out_nodes,
                            current_offset,
                            current_limit,
                            &in_node_map,
                            &locked_node_multiplicities,
                            k,
                            &executed_dijkstras,
                            false,
                            out_nodes.len() / 100,
                            out_nodes.len(),
                        );
                        let dijkstra_end = Instant::now();

                        let duration = (dijkstra_end - dijkstra_start).as_secs_f32();
                        chunk_size = ((chunk_size as f32) * (TARGET_DIJKSTRA_BLOCK_TIME / duration))
                            as usize;
                        chunk_size = chunk_size.max(10);
                    }

                    results.lock().unwrap().append(&mut shortest_paths);
                }));
            }

            info!("Waiting for dijkstra threads to finish");
        })
        .unwrap();
        let results = results.into_inner().unwrap();

        info!("Found {} shortest paths", results.len());
        let dijkstra_time = (Instant::now() - dijkstra_start).as_nanos();

        let mut dummy_edge_id = 0;
        for (out_node, in_node, distance) in results {
            let sequence = dummy_sequence.clone();
            dummy_edge_id += 1;
            let edge_data = EdgeData::new(sequence, true, distance, dummy_edge_id);
            graph.add_edge(out_node, in_node, edge_data.clone());
            graph.add_edge(
                graph.mirror_node(in_node).unwrap(),
                graph.mirror_node(out_node).unwrap(),
                edge_data.mirror(),
            );
        }

        let end = Instant::now();
        info!(
            "Took {:.6}s for computing paths and getting edges, of this {:.6}s are from dijkstra",
            (end - start).as_secs_f64(),
            dijkstra_time as f64 / 1e9
        );
        dummy_edge_id
    };

    debug_assert!(graph.verify_node_pairing());
    debug_assert!(graph.verify_edge_mirror_property());
    debug_assert_graph_has_no_consecutive_dummy_edges(graph, k);

    info!("Making graph Eulerian by adding breaking dummy edges");
    make_graph_eulerian_with_breaking_edges(graph, dummy_sequence, &mut dummy_edge_id, k);

    // Check if the graph now really is Eulerian, and if not, output some debug information
    if !decomposes_into_eulerian_bicycles(graph) {
        let non_eulerian_nodes_and_differences = find_non_eulerian_binodes_with_differences(graph);
        error!(
            "Failed to make the graph Eulerian. Non-Eulerian nodes and differences:\n{:?}",
            non_eulerian_nodes_and_differences
        );
        panic!("Failed to make the graph Eulerian.");
    }
    debug_assert!(graph.verify_node_pairing());
    debug_assert!(graph.verify_edge_mirror_property());
    debug_assert_graph_has_no_consecutive_dummy_edges(graph, k);

    info!("Finding Eulerian bicycle");
    //debug!("{:?}", graph);
    let mut eulerian_cycles = compute_minimum_bidirected_eulerian_cycle_decomposition(graph);
    info!("Found {} Eulerian bicycles", eulerian_cycles.len());

    info!("Breaking Eulerian bicycles at expensive temporary edges");
    let mut greedytigs = Vec::new();

    let mut removed_edges = 0;
    for eulerian_cycle in &mut eulerian_cycles {
        info!(
            "Processing Eulerian bicycle with {} biedges",
            eulerian_cycle.len()
        );
        debug_assert!(eulerian_cycle.is_circular_walk(graph));

        // Rotate cycle such that longest dummy is first edge
        let mut longest_dummy_weight = 0;
        let mut longest_dummy_index = 0;
        for (index, &edge) in eulerian_cycle.iter().enumerate() {
            let edge_data = graph.edge_data(edge);
            if edge_data.is_dummy() && edge_data.weight() > longest_dummy_weight {
                longest_dummy_weight = edge_data.weight();
                longest_dummy_index = index;
            }
        }
        if longest_dummy_weight > 0 {
            eulerian_cycle.rotate_left(longest_dummy_index);
        }

        let mut offset = 0;
        let mut last_edge_is_dummy = false;
        for (current_cycle_index, &current_edge_index) in eulerian_cycle.iter().enumerate() {
            let edge_data = graph.edge_data(current_edge_index);

            if edge_data.is_original() {
                last_edge_is_dummy = false;
            } else {
                if last_edge_is_dummy {
                    warn!(
                        "Found consecutive dummy edges at {}",
                        current_edge_index.as_usize()
                    );
                }
                last_edge_is_dummy = true;
            }

            if (edge_data.weight() >= k && edge_data.is_dummy())
                || (edge_data.is_dummy() && current_cycle_index == 0)
            {
                if offset < current_cycle_index {
                    greedytigs.push(eulerian_cycle[offset..current_cycle_index].to_owned());
                } else if current_cycle_index > 0 {
                    warn!("Found consecutive breaking edges");
                }
                offset = current_cycle_index + 1;
                removed_edges += 1;
            }
        }
        if offset < eulerian_cycle.len() {
            if graph
                .edge_data(*eulerian_cycle.last().unwrap())
                .is_original()
            {
                greedytigs.push(eulerian_cycle[offset..eulerian_cycle.len()].to_owned());
            } else if offset < eulerian_cycle.len() - 1 {
                greedytigs.push(eulerian_cycle[offset..eulerian_cycle.len() - 1].to_owned());
            }
        }
    }

    info!("Found {} expensive temporary edges", removed_edges);
    info!("Found {} greedytigs", greedytigs.len());

    for greedytig in &greedytigs {
        debug_assert!(!greedytig.is_empty());
        debug_assert!(graph.edge_data(*greedytig.first().unwrap()).is_original());
        debug_assert!(graph.edge_data(*greedytig.last().unwrap()).is_original());
    }

    greedytigs
}

/// Computes matchtigs in the given graph.
/// The `matcher_path` should point to a [blossom5](https://pub.ist.ac.at/~vnk/software.html) binary.
/// The `matching_file_prefix` is the name-prefix of the file used to store the matching instance and its result.
pub fn compute_matchtigs<
    NodeIndex: GraphIndex<OptionalNodeIndex> + Send + Sync,
    OptionalNodeIndex: OptionalGraphIndex<NodeIndex>,
    SequenceHandle: Default + Clone,
    EdgeData: BidirectedData + Eq + MatchtigEdgeData<SequenceHandle> + Clone,
    Graph: DynamicEdgeCentricBigraph<
            NodeIndex = NodeIndex,
            OptionalNodeIndex = OptionalNodeIndex,
            EdgeData = EdgeData,
        > + Send
        + Sync,
    DijkstraHeapType: DijkstraHeap<usize, Graph::NodeIndex>,
    DijkstraNodeWeightArray: NodeWeightArray<usize>,
>(
    graph: &mut Graph,
    threads: usize,
    k: usize,
    matching_file_prefix: &str,
    matcher_path: &str,
) -> Vec<VecEdgeWalk<Graph>> {
    let mut dummy_edge_id = 0;
    let dummy_sequence = SequenceHandle::default();
    /*info!("Fixing self-mirrors");
    for node_index in graph.node_indices() {
        if graph.is_self_mirror_node(node_index) {
            dummy_edge_id += 1;
            let edge_data = EdgeData::new(dummy_sequence.clone(), false, 0, dummy_edge_id);
            graph.add_edge(node_index, node_index, edge_data.mirror());
            graph.add_edge(node_index, node_index, edge_data);
        }
    }*/

    // Find nodes with missing incoming or outgoing edges
    info!("Collecting nodes with missing incoming or outgoing edges");
    let mut out_nodes = Vec::new(); // Misses outgoing edges
    let mut in_nodes = Vec::new(); // Misses incoming edges
    let mut out_node_multiplicities = Vec::new();
    let mut in_node_multiplicities = Vec::new();
    let mut unbalanced_self_mirror_count = 0;

    for node_index in graph.node_indices() {
        let diff = compute_eulerian_superfluous_out_biedges(graph, node_index);
        if graph.is_self_mirror_node(node_index) && diff != 0 {
            in_nodes.push(node_index);
            in_node_multiplicities.push(1);
            out_nodes.push(node_index);
            out_node_multiplicities.push(1);
            unbalanced_self_mirror_count += 1;
        } else if diff > 0 {
            in_nodes.push(node_index);
            in_node_multiplicities.push(diff);
        } else if diff < 0 {
            out_nodes.push(node_index);
            out_node_multiplicities.push(-diff);
        }
    }
    info!(
        "Found {} nodes with missing outgoing edges",
        out_nodes.len()
    );
    info!("Found {} nodes with missing incoming edges", in_nodes.len());
    info!(
        "Of those there are {} self-mirrors",
        unbalanced_self_mirror_count
    );
    //debug!("Not Eulerian nodes:\n{:?}\n{:?}", out_nodes, in_nodes);
    //debug!("Multiplicities:\n{:?}", out_node_multiplicities);
    debug_assert_eq!(out_nodes.len(), in_nodes.len());

    info!("Computing shortest paths between nodes with missing outgoing and nodes with missing incoming edges");
    let (edges, node_id_map, matching_mirror_biedge_count, matching_mirror_expanded_biedge_count) =
        if threads == 1 {
            info!("Using 1 thread to run ~{} dijkstras", out_nodes.len());
            let mut dijkstra =
                Dijkstra::<_, _, DijkstraNodeWeightArray, DijkstraHeapType>::new(graph);
            let mut distances = Vec::new();
            let in_node_map = RelaxedAtomicBoolVec::new(graph.node_count());
            for node in &in_nodes {
                in_node_map.set(node.as_usize(), true);
            }

            // A collapsed variant of the edges where node ids are consecutive and binodes have been merged
            let mut edges = HashMap::new();
            let mut node_id_map = GraphMatchingNodeMap::new(graph);

            let mut matching_mirror_biedge_count = 0;
            let mut matching_mirror_expanded_biedge_count = 0;
            let mut dijkstra_time = 0;
            let print_dot_step = (out_nodes.len() / 100).max(1);
            let start = Instant::now();
            for (out_node_index, &out_node) in out_nodes.iter().enumerate() {
                if out_node_index % print_dot_step == 0 {
                    print!(".");
                    std::io::stdout().flush().unwrap();
                } else if out_node_index == out_nodes.len() - 1 {
                    println!();
                }

                let dijkstra_start = Instant::now();
                dijkstra.shortest_path_lens(
                    graph,
                    out_node,
                    &in_node_map,
                    in_nodes.len(),
                    k - 1,
                    true,
                    &mut distances,
                );
                dijkstra_time += (Instant::now() - dijkstra_start).as_nanos();
                for &(target_node, weight) in &distances {
                    // Self-complemental nodes technically have a path of length 0 starting and ending at them,
                    // but these paths do not change their imbalance.
                    // So they are useless, and therefore we ignore them in the dijkstras.
                    debug_assert_ne!(
                        out_node, target_node,
                        "Found shortest path with same start and end"
                    );

                    if weight == 0 {
                        error!(
                            "Found zero weight path from {} to {}",
                            out_node.as_usize(),
                            target_node.as_usize()
                        );
                        error!("original_weight = {}; k = {}", weight, k);
                        debug_assert_ne!(weight, 0);
                    }

                    let is_mirror_biedge = (out_node == graph.mirror_node(target_node).unwrap())
                        && out_node != target_node;
                    if is_mirror_biedge {
                        matching_mirror_biedge_count += 1;
                    }

                    node_id_map.get_or_create_node_indexes(graph, out_node);
                    node_id_map.get_or_create_node_indexes(graph, target_node);
                    let collapsed_n1s = node_id_map.get_node_indexes(out_node);
                    let collapsed_n2s = node_id_map.get_node_indexes(target_node);

                    for &collapsed_n1 in collapsed_n1s {
                        for &collapsed_n2 in collapsed_n2s {
                            if collapsed_n1 == collapsed_n2 {
                                debug_assert!(
                                    is_mirror_biedge,
                                    "Found self-loop not caused by a mirror biedge"
                                );
                                continue;
                            }

                            let previous = edges.insert(
                                (
                                    collapsed_n1.min(collapsed_n2),
                                    collapsed_n2.max(collapsed_n1),
                                ),
                                (weight, out_node, target_node),
                            );
                            if previous.is_none() {
                                if is_mirror_biedge {
                                    matching_mirror_expanded_biedge_count += 1;
                                }
                            }
                            debug_assert!(
                                previous.is_none()
                                    || previous.unwrap_or((0, 0.into(), 0.into())).0 == weight
                            );
                        }
                    }
                }
            }
            let end = Instant::now();
            info!(
            "Took {:.6}s for computing paths and getting edges, of this {:.6}s are from dijkstra",
            (end - start).as_secs_f64(),
            dijkstra_time as f64 / 1e9
        );

            (
                edges,
                node_id_map,
                matching_mirror_biedge_count,
                matching_mirror_expanded_biedge_count,
            )
        } else {
            info!(
                "Using {} threads to run ~{} dijkstras",
                threads,
                out_nodes.len()
            );
            let start = Instant::now();
            let in_node_map = RelaxedAtomicBoolVec::new(graph.node_count());
            for node in &in_nodes {
                in_node_map.set(node.as_usize(), true);
            }

            // A collapsed variant of the edges where node ids are consecutive and binodes have been merged
            let mut edges = HashMap::new();
            let mut node_id_map = GraphMatchingNodeMap::new(graph);
            let mut matching_mirror_biedge_count = 0;
            let mut matching_mirror_expanded_biedge_count = 0;
            let executed_dijkstras = RelaxedCounter::default();

            #[allow(clippy::too_many_arguments)]
            fn compute_dijkstras<
                EdgeData: DijkstraWeightedEdgeData<usize>,
                Graph: StaticGraph<EdgeData = EdgeData>,
                DijkstraNodeWeightArray: NodeWeightArray<usize>,
                DijkstraHeapType: DijkstraHeap<usize, Graph::NodeIndex>,
            >(
                graph: &Graph,
                dijkstra: &mut Dijkstra<Graph, usize, DijkstraNodeWeightArray, DijkstraHeapType>,
                distances: &mut Vec<(Graph::NodeIndex, usize)>,
                shortest_paths: &mut Vec<(Graph::NodeIndex, Graph::NodeIndex, usize)>,
                out_nodes: &[Graph::NodeIndex],
                in_node_map: &RelaxedAtomicBoolVec,
                in_node_len: usize,
                k: usize,
                executed_dijkstras: &RelaxedCounter,
                output: bool,
                output_step: usize,
                total_dijkstras: usize,
            ) {
                let mut last_output = 0;

                for (i, &out_node) in out_nodes.iter().enumerate() {
                    dijkstra.shortest_path_lens(
                        graph,
                        out_node,
                        in_node_map,
                        in_node_len,
                        k - 1,
                        true,
                        distances,
                    );
                    for &(in_node, distance) in distances.iter() {
                        shortest_paths.push((out_node, in_node, distance));
                    }
                    executed_dijkstras.inc();
                    if output && executed_dijkstras.get() - last_output > output_step {
                        last_output += output_step;
                        println!(
                            "{}%, ~{} total shortest paths",
                            last_output * 100 / total_dijkstras,
                            (shortest_paths.len() as f64 * total_dijkstras as f64 / i as f64)
                                as u64,
                        );
                    }
                }
            }

            let dijkstra_start = Instant::now();

            let results = Mutex::new(Vec::new());
            let offset = Mutex::new(0);
            let shared_graph = &*graph;
            crossbeam::scope(|scope| {
                let mut thread_handles = Vec::new();

                info!("Creating dijkstra threads");

                for _ in 0..threads {
                    thread_handles.push(scope.spawn(|_| {
                        let graph = shared_graph;
                        let mut dijkstra = DefaultDijkstra::new(graph);
                        let mut distances = Vec::new();
                        let mut shortest_paths = Vec::new();
                        let mut chunk_size = 1024;

                        loop {
                            let (current_offset, current_limit) = {
                                let mut offset = offset.lock().unwrap();
                                let current_offset = *offset;
                                let remaining_nodes = out_nodes.len() - *offset;

                                if remaining_nodes == 0 {
                                    break;
                                }

                                chunk_size = chunk_size
                                    .min(remaining_nodes / threads)
                                    .max(10)
                                    .max(chunk_size / 10)
                                    .min(remaining_nodes);
                                let current_limit = current_offset + chunk_size;
                                *offset = current_limit;
                                (current_offset, current_limit)
                            };

                            let dijkstra_start = Instant::now();
                            compute_dijkstras(
                                graph,
                                &mut dijkstra,
                                &mut distances,
                                &mut shortest_paths,
                                &out_nodes[current_offset..current_limit],
                                &in_node_map,
                                in_nodes.len(),
                                k,
                                &executed_dijkstras,
                                false,
                                out_nodes.len() / 100,
                                out_nodes.len(),
                            );
                            let dijkstra_end = Instant::now();

                            let duration = (dijkstra_end - dijkstra_start).as_secs_f32();
                            chunk_size = ((chunk_size as f32)
                                * (TARGET_DIJKSTRA_BLOCK_TIME / duration))
                                as usize;
                            chunk_size = chunk_size.max(10);
                        }

                        results.lock().unwrap().append(&mut shortest_paths);
                    }));
                }

                info!("Waiting for dijkstra threads to finish");
            })
            .unwrap();
            let results = results.into_inner().unwrap();

            info!("Found {} shortest paths", results.len());
            let dijkstra_time = (Instant::now() - dijkstra_start).as_nanos();

            for (out_node, target_node, weight) in results {
                // Self-complemental nodes technically have a path of length 0 starting and ending at them,
                // but these paths do not change their imbalance.
                // So they are useless, and therefore we ignore them in the dijkstras.
                debug_assert_ne!(
                    out_node, target_node,
                    "Found shortest path with same start and end"
                );

                if weight == 0 {
                    error!(
                        "Found zero weight path from {} to {}",
                        out_node.as_usize(),
                        target_node.as_usize()
                    );
                    error!("original_weight = {}; k = {}", weight, k);
                    debug_assert_ne!(weight, 0);
                }

                let is_mirror_biedge = (out_node == graph.mirror_node(target_node).unwrap())
                    && out_node != target_node;
                if is_mirror_biedge {
                    matching_mirror_biedge_count += 1;
                }

                node_id_map.get_or_create_node_indexes(graph, out_node);
                node_id_map.get_or_create_node_indexes(graph, target_node);
                let collapsed_n1s = node_id_map.get_node_indexes(out_node);
                let collapsed_n2s = node_id_map.get_node_indexes(target_node);

                for &collapsed_n1 in collapsed_n1s {
                    for &collapsed_n2 in collapsed_n2s {
                        if collapsed_n1 == collapsed_n2 {
                            debug_assert!(
                                is_mirror_biedge,
                                "Found self-loop not caused by a mirror biedge"
                            );
                            continue;
                        }

                        // If it ever happens that two self-complemental nodes are connected by a shortest path,
                        // only one direction of the shortest path will be added.
                        // This is fine, as to balance the self-complemental nodes only one of the directions is needed,
                        // and it does not matter which.
                        let previous = edges.insert(
                            (
                                collapsed_n1.min(collapsed_n2),
                                collapsed_n2.max(collapsed_n1),
                            ),
                            (weight, out_node, target_node),
                        );
                        if previous.is_none() {
                            if is_mirror_biedge {
                                matching_mirror_expanded_biedge_count += 1;
                            }
                        }
                        debug_assert!(
                            previous.is_none()
                                || previous.unwrap_or((0, 0.into(), 0.into())).0 == weight
                        );
                    }
                }
            }
            let end = Instant::now();
            info!(
            "Took {:.6}s for computing paths and getting edges, of this {:.6}s are from dijkstra",
            (end - start).as_secs_f64(),
            dijkstra_time as f64 / 1e9
        );

            (
                edges,
                node_id_map,
                matching_mirror_biedge_count,
                matching_mirror_expanded_biedge_count,
            )
        };

    let transformed_node_count = node_id_map.node_count();
    info!(
        "Found {} nodes and {} edges",
        transformed_node_count,
        edges.len()
    );
    info!(
        "Matching problem contains {} edges that originate from {} mirror biedges",
        matching_mirror_expanded_biedge_count, matching_mirror_biedge_count
    );

    // Output matching graph transformed to a perfect minimal matching problem
    let matching_input_path = matching_file_prefix.to_owned() + ".minimalperfectmatching";
    info!("Outputting matching problem to {:?}", matching_input_path);
    let mut output_writer = BufWriter::new(File::create(&matching_input_path).unwrap());
    writeln!(
        output_writer,
        "{} {}",
        transformed_node_count * 2,
        edges.len() * 2 + transformed_node_count,
    )
    .unwrap();

    // Write the first copy of the matching graph with edges to the second copy
    let mut last_n1 = None;
    for (n1, n2, weight) in edges
        .iter()
        .map(|(&(n1, n2), &(weight, ..))| (n1, n2, weight))
        .sorted()
    {
        debug_assert_ne!(weight, 0);
        if let Some(last_n1) = last_n1.as_mut() {
            // Write edges between the two copies of the matching graph
            while *last_n1 < n1 {
                writeln!(
                    output_writer,
                    "{} {} {}",
                    *last_n1,
                    *last_n1 + transformed_node_count,
                    k - 1,
                )
                .unwrap();
                *last_n1 += 1;
            }
        }
        debug_assert_ne!(n1, n2, "Self loops are not allowed in the matching graph");
        writeln!(output_writer, "{} {} {}", n1, n2, weight).unwrap();
        last_n1 = Some(n1);
    }

    // Write the remaining edges between the two copies of the matching graph
    let mut last_n1 = last_n1.unwrap_or(0);
    while last_n1 < transformed_node_count {
        writeln!(
            output_writer,
            "{} {} {}",
            last_n1,
            last_n1 + transformed_node_count,
            k - 1,
        )
        .unwrap();
        last_n1 += 1;
    }

    // Write second copy of matching graph
    for (&(n1, n2), &(weight, ..)) in &edges {
        if n1 == n2 {
            continue;
        }

        writeln!(
            output_writer,
            "{} {} {}",
            n1 + transformed_node_count,
            n2 + transformed_node_count,
            weight
        )
        .unwrap();
    }
    drop(output_writer);

    // Matching if necessary
    let matching_output_path = matching_input_path.clone() + ".solution";

    if transformed_node_count != 0 {
        // Run matcher
        info!("Running matcher at {}", matcher_path);
        let matcher_output = Command::new(matcher_path)
            .arg("-e")
            .arg(&matching_input_path)
            .arg("-w")
            .arg(&matching_output_path)
            .stdout(Stdio::inherit())
            .output()
            .unwrap();
        assert!(
            matcher_output.status.success(),
            "Matcher was unsuccessful\nstderr: {}",
            String::from_utf8_lossy(&matcher_output.stderr)
        );
    } else {
        // Nothing to match
        info!("Nothing to match, generating empty output file");
        let mut writer = BufWriter::new(File::create(&matching_output_path).unwrap());
        writeln!(writer, "0 0").unwrap();
    }

    // Read matcher result
    info!("Applying matcher result to graph");
    debug_assert!(graph.verify_node_pairing());
    debug_assert!(graph.verify_edge_mirror_property());
    debug_assert_graph_has_no_consecutive_dummy_edges(graph, k);

    let input_reader = BufReader::new(File::open(&matching_output_path).unwrap());
    let mut lines = input_reader.lines();
    debug_assert_eq!(
        lines.next().unwrap().unwrap(),
        format!("{} {}", 2 * transformed_node_count, transformed_node_count)
    );

    let mut inserted_edges = 0;
    let mut bidirected_inserted_edges = 0;
    for line in lines {
        let line = line.unwrap();
        let mut columns = line.split(' ');
        let n1 = columns.next().unwrap().parse().unwrap();
        let n2 = columns.next().unwrap().parse().unwrap();

        if n1 >= transformed_node_count && n2 >= transformed_node_count {
            // If both nodes are from the second copy of the graph, do not add the duplicate edge.
            continue;
        }

        let n1 = if n1 >= transformed_node_count {
            n1 - transformed_node_count
        } else {
            n1
        };
        let n2 = if n2 >= transformed_node_count {
            n2 - transformed_node_count
        } else {
            n2
        };

        let &(weight, original_n1, original_n2) = if let Some(value) = edges.get(&(n1, n2)) {
            value
        } else if n1 == n2 {
            continue;
        } else {
            panic!("Edge does not exist: ({}, {})", n1, n2);
        };

        //debug!("Adding edges ({}, {}) and ({}, {})", original_n1.as_usize(), original_n2.as_usize(), graph.mirror_node(original_n2).unwrap().as_usize(), graph.mirror_node(original_n1).unwrap().as_usize());

        let sequence = dummy_sequence.clone();
        dummy_edge_id += 1;
        let edge_data = EdgeData::new(sequence, true, weight, dummy_edge_id);
        graph.add_edge(original_n1, original_n2, edge_data.clone());
        graph.add_edge(
            graph.mirror_node(original_n2).unwrap(),
            graph.mirror_node(original_n1).unwrap(),
            edge_data.mirror(),
        );

        inserted_edges += 2;
        if original_n1 == graph.mirror_node(original_n2).unwrap() {
            bidirected_inserted_edges += 2;
        }
    }

    info!("Inserted {} matched edges", inserted_edges);
    if bidirected_inserted_edges > 0 {
        warn!("Inserted {} bidirected loops", bidirected_inserted_edges);
    }

    /*for edge_index in graph.edge_indices() {
        println!("{} {:?}", edge_index.as_usize(), graph.edge_data(edge_index));
    }*/

    debug_assert!(graph.verify_node_pairing());
    debug_assert!(graph.verify_edge_mirror_property());
    debug_assert_graph_has_no_consecutive_dummy_edges(graph, k);

    info!("Making graph Eulerian by completing unmatched nodes");
    make_graph_eulerian_with_breaking_edges(graph, dummy_sequence, &mut dummy_edge_id, k);

    // Check if the graph now really is Eulerian, and if not, output some debug information
    if !decomposes_into_eulerian_bicycles(graph) {
        let non_eulerian_nodes_and_differences = find_non_eulerian_binodes_with_differences(graph);
        error!(
            "Failed to make the graph Eulerian. Non-Eulerian nodes and differences:\n{:?}",
            non_eulerian_nodes_and_differences
        );
        for node_and_difference in non_eulerian_nodes_and_differences {
            debug!(
                "Node index {} has matching node indices {:?}",
                node_and_difference.0.as_usize(),
                node_id_map.get_node_indexes_unchecked(node_and_difference.0)
            );
        }
        panic!("Failed to make the graph Eulerian.");
    }
    debug_assert!(graph.verify_node_pairing());
    debug_assert!(graph.verify_edge_mirror_property());
    debug_assert_graph_has_no_consecutive_dummy_edges(graph, k);

    info!("Finding Eulerian bicycle");
    //debug!("{:?}", graph);
    let mut eulerian_cycles = compute_minimum_bidirected_eulerian_cycle_decomposition(graph);
    info!("Found {} Eulerian bicycles", eulerian_cycles.len());

    info!("Breaking Eulerian bicycles at expensive temporary edges");
    let mut matchtigs = Vec::new();

    let mut removed_edges = 0;
    for eulerian_cycle in &mut eulerian_cycles {
        info!(
            "Processing Eulerian bicycle with {} biedges",
            eulerian_cycle.len()
        );
        debug_assert!(eulerian_cycle.is_circular_walk(graph));

        // Rotate cycle such that longest dummy is first edge
        let mut longest_dummy_weight = 0;
        let mut longest_dummy_index = 0;
        for (index, &edge) in eulerian_cycle.iter().enumerate() {
            let edge_data = graph.edge_data(edge);
            if edge_data.is_dummy() && edge_data.weight() > longest_dummy_weight {
                longest_dummy_weight = edge_data.weight();
                longest_dummy_index = index;
            }
        }
        if longest_dummy_weight > 0 {
            eulerian_cycle.rotate_left(longest_dummy_index);
        }

        let mut offset = 0;
        let mut last_edge_is_dummy = false;
        for (current_cycle_index, &current_edge_index) in eulerian_cycle.iter().enumerate() {
            let edge_data = graph.edge_data(current_edge_index);

            if edge_data.is_original() {
                last_edge_is_dummy = false;
            } else {
                if last_edge_is_dummy {
                    warn!(
                        "Found consecutive dummy edges at {}",
                        current_edge_index.as_usize()
                    );
                }
                last_edge_is_dummy = true;
            }

            if (edge_data.weight() >= k && edge_data.is_dummy())
                || (edge_data.is_dummy() && current_cycle_index == 0)
            {
                if offset < current_cycle_index {
                    matchtigs.push(eulerian_cycle[offset..current_cycle_index].to_owned());
                } else if current_cycle_index > 0 {
                    warn!("Found consecutive breaking edges");
                }
                offset = current_cycle_index + 1;
                removed_edges += 1;
            }
        }
        if offset < eulerian_cycle.len() {
            if graph
                .edge_data(*eulerian_cycle.last().unwrap())
                .is_original()
            {
                matchtigs.push(eulerian_cycle[offset..eulerian_cycle.len()].to_owned());
            } else if offset < eulerian_cycle.len() - 1 {
                matchtigs.push(eulerian_cycle[offset..eulerian_cycle.len() - 1].to_owned());
            }
        }
    }

    info!("Found {} expensive temporary edges", removed_edges);
    info!("Found {} matchtigs", matchtigs.len());

    for matchtig in &matchtigs {
        debug_assert!(!matchtig.is_empty());
        debug_assert!(graph.edge_data(*matchtig.first().unwrap()).is_original());
        debug_assert!(graph.edge_data(*matchtig.last().unwrap()).is_original());
    }

    matchtigs
}
