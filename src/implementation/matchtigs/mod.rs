use crate::implementation::{
    append_to_filename, debug_assert_graph_has_no_consecutive_dummy_edges,
    make_graph_eulerian_with_breaking_edges, GraphMatchingNodeMap, MatchtigEdgeData,
    RelaxedAtomicBoolVec, TARGET_DIJKSTRA_BLOCK_TIME,
};
use crate::{HeapType, NodeWeightArrayType, TigAlgorithm};
use atomic_counter::{AtomicCounter, RelaxedCounter};
use genome_graph::bigraph::algo::eulerian::{
    compute_eulerian_superfluous_out_biedges,
    compute_minimum_bidirected_eulerian_cycle_decomposition, decomposes_into_eulerian_bicycles,
    find_non_eulerian_binodes_with_differences,
};
use genome_graph::bigraph::interface::dynamic_bigraph::DynamicEdgeCentricBigraph;
use genome_graph::bigraph::interface::BidirectedData;
use genome_graph::bigraph::traitgraph::index::{GraphIndex, OptionalGraphIndex};
use genome_graph::bigraph::traitgraph::interface::{GraphBase, StaticGraph};
use genome_graph::bigraph::traitgraph::walks::{EdgeWalk, VecEdgeWalk};
use itertools::Itertools;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::marker::PhantomData;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::Instant;
use traitgraph_algo::dijkstra::epoch_array_dijkstra_node_weight_array::EpochNodeWeightArray;
use traitgraph_algo::dijkstra::{
    DefaultDijkstra, Dijkstra, DijkstraHeap, DijkstraWeightedEdgeData, NodeWeightArray,
};

/// The matchtigs algorithm.
#[derive(Default)]
pub struct MatchtigAlgorithm<'a, 'b, SequenceHandle> {
    _phantom_data: PhantomData<(&'a (), &'b (), SequenceHandle)>,
}

/// The options for the matchtigs algorithm.
pub struct MatchtigAlgorithmConfiguration<'a, 'b> {
    /// The number of threads to use.
    pub threads: usize,
    /// The k used to build the de Bruijn graph.
    pub k: usize,
    /// The type of the node weight array used by Dijkstra's algorithm.
    pub node_weight_array_type: NodeWeightArrayType,
    /// The type of the heap used by Dijkstra's algorithm.
    pub heap_type: HeapType,
    /// The prefix of the path used to store the matching instance.
    pub matching_file_prefix: &'a Path,
    /// The path to the blossom5 binary.
    pub matcher_path: &'b Path,
}

impl<'a, 'b, Graph: GraphBase, SequenceHandle: Default + Clone> TigAlgorithm<Graph>
    for MatchtigAlgorithm<'a, 'b, SequenceHandle>
where
    Graph: DynamicEdgeCentricBigraph + Send + Sync,
    Graph::NodeIndex: Send + Sync,
    Graph::EdgeData: BidirectedData + Eq + Clone + MatchtigEdgeData<SequenceHandle>,
{
    type Configuration = MatchtigAlgorithmConfiguration<'a, 'b>;

    fn compute_tigs(
        graph: &mut Graph,
        configuration: &Self::Configuration,
    ) -> Vec<VecEdgeWalk<Graph>> {
        compute_matchtigs_choose_heap_type(graph, configuration)
    }
}

fn compute_matchtigs_choose_heap_type<
    'a,
    'b,
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
    configuration: &MatchtigAlgorithmConfiguration<'a, 'b>,
) -> Vec<VecEdgeWalk<Graph>> {
    match configuration.heap_type {
        HeapType::StdBinaryHeap => {
            compute_matchtigs_choose_node_weight_array_type::<_, _, _, _, _, BinaryHeap<_>>(
                graph,
                configuration,
            )
        }
    }
}

fn compute_matchtigs_choose_node_weight_array_type<
    'a,
    'b,
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
    configuration: &MatchtigAlgorithmConfiguration<'a, 'b>,
) -> Vec<VecEdgeWalk<Graph>> {
    match configuration.node_weight_array_type {
        NodeWeightArrayType::EpochNodeWeightArray => {
            compute_matchtigs::<_, _, _, _, _, DijkstraHeapType, EpochNodeWeightArray<_>>(
                graph,
                configuration,
            )
        }
        NodeWeightArrayType::HashbrownHashMap => {
            compute_matchtigs::<_, _, _, _, _, DijkstraHeapType, hashbrown::HashMap<_, _>>(
                graph,
                configuration,
            )
        }
    }
}

/// Computes matchtigs in the given graph.
/// The `matcher_path` should point to a [blossom5](https://pub.ist.ac.at/~vnk/software.html) binary.
/// The `matching_file_prefix` is the name-prefix of the file used to store the matching instance and its result.
fn compute_matchtigs<
    'a,
    'b,
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
    configuration: &MatchtigAlgorithmConfiguration<'a, 'b>,
) -> Vec<VecEdgeWalk<Graph>> {
    let threads = configuration.threads;
    let k = configuration.k;
    let matching_file_prefix = configuration.matching_file_prefix;
    let matcher_path = configuration.matcher_path;

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
    let matching_input_path =
        append_to_filename(matching_file_prefix.to_owned(), ".minimalperfectmatching");
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
    let matching_output_path = append_to_filename(matching_input_path.clone(), ".solution");

    if transformed_node_count != 0 {
        // Run matcher
        info!("Running matcher at {:?}", matcher_path);
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
