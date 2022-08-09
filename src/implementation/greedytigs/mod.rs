use crate::implementation::{
    debug_assert_graph_has_no_consecutive_dummy_edges, make_graph_eulerian_with_breaking_edges,
    MatchtigEdgeData, PerformanceDataType, RelaxedAtomicBoolSlice, RelaxedAtomicBoolVec,
    TARGET_DIJKSTRA_BLOCK_TIME,
};
use crate::{HeapType, NodeWeightArrayType, TigAlgorithm};
use atomic_counter::{AtomicCounter, RelaxedCounter};
use genome_graph::bigraph::algo::eulerian::{
    compute_eulerian_superfluous_out_biedges,
    compute_minimum_bidirected_eulerian_cycle_decomposition, decomposes_into_eulerian_bicycles,
    find_non_eulerian_binodes_with_differences,
};
use genome_graph::bigraph::interface::dynamic_bigraph::DynamicEdgeCentricBigraph;
use genome_graph::bigraph::interface::static_bigraph::StaticBigraph;
use genome_graph::bigraph::interface::BidirectedData;
use genome_graph::bigraph::traitgraph::index::{GraphIndex, OptionalGraphIndex};
use genome_graph::bigraph::traitgraph::interface::GraphBase;
use genome_graph::bigraph::traitgraph::walks::{EdgeWalk, VecEdgeWalk};
use std::collections::BinaryHeap;
use std::marker::PhantomData;
use std::ops::AddAssign;
use std::sync::Mutex;
use std::time::Instant;
use traitgraph_algo::dijkstra::epoch_array_dijkstra_node_weight_array::EpochNodeWeightArray;
use traitgraph_algo::dijkstra::performance_counters::{
    DijkstraPerformanceCounter, DijkstraPerformanceData, NoopDijkstraPerformanceCounter,
};
use traitgraph_algo::dijkstra::{
    Dijkstra, DijkstraExhaustiveness, DijkstraHeap, DijkstraWeightedEdgeData, NodeWeightArray,
};

/// The greedy matchtigs algorithm.
#[derive(Default)]
pub struct GreedytigAlgorithm<SequenceHandle> {
    _phantom_data: PhantomData<SequenceHandle>,
}

/// The options for the greedy matchtigs algorithm.
pub struct GreedytigAlgorithmConfiguration {
    /// The number of threads to use.
    pub threads: usize,
    /// The k used to build the de Bruijn graph.
    pub k: usize,
    /// If given, enables staged parallelism mode.
    /// In this mode, shortest path queries are limited in the amount of memory they can use, and larger queries are postponed to be executed with less queries in parallel.
    pub staged_parallelism_divisor: Option<f64>,
    /// If staged paralallelism mode is on, then the resource limits are calculated based on the existing nodes in the graph times this factor, divided by the number of threads.
    pub resource_limit_factor: usize,
    /// The type of the node weight array used by Dijkstra's algorithm.
    pub node_weight_array_type: NodeWeightArrayType,
    /// The type of the heap used by Dijkstra's algorithm.
    pub heap_type: HeapType,
    /// The type of the performance data collector used by Dikstra's algorithm.
    pub performance_data_type: PerformanceDataType,
}

impl<Graph: GraphBase, SequenceHandle: Default + Clone> TigAlgorithm<Graph>
    for GreedytigAlgorithm<SequenceHandle>
where
    Graph: DynamicEdgeCentricBigraph + Send + Sync,
    Graph::NodeIndex: Send + Sync,
    Graph::EdgeData: BidirectedData + Eq + Clone + MatchtigEdgeData<SequenceHandle>,
{
    type Configuration = GreedytigAlgorithmConfiguration;

    fn compute_tigs(
        graph: &mut Graph,
        configuration: &Self::Configuration,
    ) -> Vec<VecEdgeWalk<Graph>> {
        compute_greedytigs_choose_heap_type(graph, configuration)
    }
}

fn compute_greedytigs_choose_heap_type<
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
>(
    graph: &mut Graph,
    configuration: &GreedytigAlgorithmConfiguration,
) -> Vec<VecEdgeWalk<Graph>> {
    match configuration.heap_type {
        HeapType::StdBinaryHeap => {
            compute_greedytigs_choose_node_weight_array_type::<_, _, _, _, _, BinaryHeap<_>>(
                graph,
                configuration,
            )
        }
    }
}

fn compute_greedytigs_choose_node_weight_array_type<
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
>(
    graph: &mut Graph,
    configuration: &GreedytigAlgorithmConfiguration,
) -> Vec<VecEdgeWalk<Graph>> {
    match configuration.node_weight_array_type {
        NodeWeightArrayType::EpochNodeWeightArray => {
            compute_greedytigs_choose_dijkstra_performance_type::<
                _,
                _,
                _,
                _,
                _,
                DijkstraHeapType,
                EpochNodeWeightArray<_>,
            >(graph, configuration)
        }
        NodeWeightArrayType::HashbrownHashMap => {
            compute_greedytigs_choose_dijkstra_performance_type::<
                _,
                _,
                _,
                _,
                _,
                DijkstraHeapType,
                hashbrown::HashMap<_, _>,
            >(graph, configuration)
        }
    }
}

fn compute_greedytigs_choose_dijkstra_performance_type<
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
    configuration: &GreedytigAlgorithmConfiguration,
) -> Vec<VecEdgeWalk<Graph>> {
    match configuration.performance_data_type {
        PerformanceDataType::None => compute_greedytigs::<
            _,
            _,
            _,
            _,
            _,
            DijkstraHeapType,
            DijkstraNodeWeightArray,
            NoopDijkstraPerformanceCounter,
        >(graph, configuration),
        PerformanceDataType::Complete => compute_greedytigs::<
            _,
            _,
            _,
            _,
            _,
            DijkstraHeapType,
            DijkstraNodeWeightArray,
            DijkstraPerformanceCounter,
        >(graph, configuration),
    }
}

/// Computes greedy matchtigs in the given graph.
fn compute_greedytigs<
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
    DijkstraPerformance: DijkstraPerformanceData + Default + AddAssign + Send,
>(
    graph: &mut Graph,
    configuration: &GreedytigAlgorithmConfiguration,
) -> Vec<VecEdgeWalk<Graph>> {
    let threads = configuration.threads;
    let k = configuration.k;

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
            'missed_nodes,
            EdgeData: DijkstraWeightedEdgeData<usize>,
            Graph: StaticBigraph<EdgeData = EdgeData>,
            DijkstraNodeWeightArray: NodeWeightArray<usize>,
            DijkstraHeapType: DijkstraHeap<usize, Graph::NodeIndex>,
            DijkstraPerformance: DijkstraPerformanceData + Default,
        >(
            graph: &Graph,
            dijkstra: &mut Dijkstra<Graph, usize, DijkstraNodeWeightArray, DijkstraHeapType>,
            distances: &mut Vec<(Graph::NodeIndex, usize)>,
            shortest_paths: &mut Vec<(Graph::NodeIndex, Graph::NodeIndex, usize)>,
            out_nodes: &[Graph::NodeIndex],
            missed_nodes: RelaxedAtomicBoolSlice<'missed_nodes>,
            dijkstra_performance_limit: usize,
            in_node_map: &RelaxedAtomicBoolVec,
            locked_node_multiplicities: &[Mutex<isize>],
            k: usize,
            executed_dijkstras: &RelaxedCounter,
            output: bool,
            output_step: usize,
            total_dijkstras: usize,
        ) -> DijkstraPerformance {
            let mut last_output = 0;
            let mut dijkstra_performance = DijkstraPerformance::default();

            for (i, &out_node) in out_nodes.iter().enumerate() {
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
                    let dijkstra_status = dijkstra.shortest_path_lens(
                        graph,
                        out_node,
                        in_node_map,
                        target_amount,
                        k - 1,
                        true,
                        distances,
                        dijkstra_performance_limit,
                        dijkstra_performance_limit,
                        dijkstra_performance,
                    );
                    dijkstra_performance = dijkstra_status.performance_data;

                    if distances.is_empty() {
                        missed_nodes.set(
                            i,
                            dijkstra_status.exhaustiveness == DijkstraExhaustiveness::Complete,
                        );

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
                        missed_nodes.set(
                            i,
                            dijkstra_status.exhaustiveness == DijkstraExhaustiveness::Complete,
                        );

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

            dijkstra_performance
        }

        let dijkstra_start = Instant::now();

        let results = Mutex::new(Vec::new());
        // The linter suggestion to use an atomic value instead of the mutex lock here does not work,
        // as we increment the value based on its current value.
        let offset = Mutex::new(0);
        let shared_graph = &*graph;
        let mut dijkstra_performance = DijkstraPerformance::default();

        let mut unbalanced_out_nodes = out_nodes;
        let mut missed_nodes = RelaxedAtomicBoolVec::new(unbalanced_out_nodes.len());

        for iteration in 0.. {
            let threads = if iteration == 0 {
                threads
            } else {
                (((threads as f64)
                    / configuration
                        .staged_parallelism_divisor
                        .expect("Iteration > 0 will never be reached if this is None")
                        .powi(iteration as i32)) as usize)
                    .max(1)
            };
            let resource_limit =
                if threads == 1 || configuration.staged_parallelism_divisor.is_none() {
                    usize::MAX
                } else {
                    graph
                        .node_count()
                        .saturating_mul(configuration.resource_limit_factor)
                        / threads
                };

            crossbeam::scope(|scope| {
                info!("Starting {threads} dijkstra threads with a resource limit of {resource_limit} each");
                let mut thread_handles = Vec::new();

                for _ in 0..threads {
                    thread_handles.push(scope.spawn(|_| {
                        let graph = shared_graph;
                        let mut dijkstra =
                            Dijkstra::<_, _, DijkstraNodeWeightArray, DijkstraHeapType>::new(graph);
                        let mut distances = Vec::new();
                        let mut shortest_paths = Vec::new();
                        let mut chunk_size = 1024;
                        let mut dijkstra_performance = DijkstraPerformance::default();

                        loop {
                            let (current_offset, current_limit) = {
                                let mut offset = offset.lock().unwrap();
                                let current_offset = *offset;
                                let remaining_nodes = unbalanced_out_nodes.len() - *offset;

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
                            dijkstra_performance += compute_dijkstras(
                                graph,
                                &mut dijkstra,
                                &mut distances,
                                &mut shortest_paths,
                                &unbalanced_out_nodes[current_offset..current_limit],
                                missed_nodes.slice(current_offset..current_limit),
                                resource_limit,
                                &in_node_map,
                                &locked_node_multiplicities,
                                k,
                                &executed_dijkstras,
                                false,
                                unbalanced_out_nodes.len() / 100,
                                unbalanced_out_nodes.len(),
                            );
                            let dijkstra_end = Instant::now();

                            let duration = (dijkstra_end - dijkstra_start).as_secs_f32();
                            chunk_size = ((chunk_size as f32) * (TARGET_DIJKSTRA_BLOCK_TIME / duration))
                                as usize;
                            chunk_size = chunk_size.max(10);
                        }

                        results.lock().unwrap().append(&mut shortest_paths);
                        dijkstra_performance
                    }));
                }

                info!("Waiting for {} dijkstra threads to finish", thread_handles.len());
                for thread_handle in thread_handles {
                    dijkstra_performance += thread_handle.join().unwrap();
                }
            }).unwrap();

            unbalanced_out_nodes = unbalanced_out_nodes
                .iter()
                .zip(missed_nodes.iter())
                .filter_map(|(&out_node, missed)| if missed { Some(out_node) } else { None })
                .collect();
            missed_nodes.reinitialise(unbalanced_out_nodes.len());
            info!(
                "{} remaining unbalanced out nodes",
                unbalanced_out_nodes.len()
            );

            if unbalanced_out_nodes.is_empty() {
                break;
            }
            assert!(configuration.staged_parallelism_divisor.is_some(), "Have nodes where Dijkstra missed part of the search space, but not in staged parallelism mode");
        }

        let results = results.into_inner().unwrap();
        if let (Some(unnecessary_heap_elements), Some(iterations)) = (
            dijkstra_performance.unnecessary_heap_elements(),
            dijkstra_performance.iterations(),
        ) {
            info!(
                "Dijkstras had a factor of {:.3} unnecessary heap elements",
                unnecessary_heap_elements as f64 / iterations as f64
            );
        }

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
