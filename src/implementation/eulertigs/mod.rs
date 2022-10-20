use crate::implementation::{
    debug_assert_graph_has_no_consecutive_dummy_edges, make_graph_eulerian_with_breaking_edges,
    MatchtigEdgeData, RelaxedAtomicBoolVec,
};
use crate::TigAlgorithm;
use genome_graph::bigraph::algo::eulerian::{
    compute_eulerian_superfluous_out_biedges,
    compute_minimum_bidirected_eulerian_cycle_decomposition, decomposes_into_eulerian_bicycles,
    find_non_eulerian_binodes_with_differences,
};
use genome_graph::bigraph::interface::dynamic_bigraph::DynamicEdgeCentricBigraph;
use genome_graph::bigraph::interface::BidirectedData;
use genome_graph::bigraph::traitgraph::index::{GraphIndex, OptionalGraphIndex};
use genome_graph::bigraph::traitgraph::interface::GraphBase;
use genome_graph::bigraph::traitgraph::walks::{EdgeWalk, VecEdgeWalk};
use log::{error, info, warn};
use std::marker::PhantomData;

/// The eulertigs algorithm.
#[derive(Default)]
pub struct EulertigAlgorithm<SequenceHandle> {
    _phantom_data: PhantomData<SequenceHandle>,
}

impl<Graph: GraphBase, SequenceHandle: Default + Clone> TigAlgorithm<Graph>
    for EulertigAlgorithm<SequenceHandle>
where
    Graph: DynamicEdgeCentricBigraph,
    Graph::EdgeData: BidirectedData + Eq + Clone + MatchtigEdgeData<SequenceHandle>,
{
    type Configuration = EulertigAlgorithmConfiguration;

    fn compute_tigs(
        graph: &mut Graph,
        configuration: &Self::Configuration,
    ) -> Vec<VecEdgeWalk<Graph>> {
        compute_eulertigs(graph, configuration)
    }
}

/// The options for the eulertigs algorithm.
pub struct EulertigAlgorithmConfiguration {
    /// The k used to build the de Bruijn graph.
    pub k: usize,
}

/// Computes eulertigs in the given graph.
fn compute_eulertigs<
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
    configuration: &EulertigAlgorithmConfiguration,
) -> Vec<VecEdgeWalk<Graph>> {
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

    info!("Making graph Eulerian by adding breaking dummy edges");
    let dummy_sequence = SequenceHandle::default();
    let mut dummy_edge_id = 0;
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
    let mut eulertigs = Vec::new();

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
                    eulertigs.push(eulerian_cycle[offset..current_cycle_index].to_owned());
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
                eulertigs.push(eulerian_cycle[offset..eulerian_cycle.len()].to_owned());
            } else if offset < eulerian_cycle.len() - 1 {
                eulertigs.push(eulerian_cycle[offset..eulerian_cycle.len() - 1].to_owned());
            }
        }
    }

    info!("Found {} expensive temporary edges", removed_edges);
    info!("Found {} eulertigs", eulertigs.len());

    for eulertig in &eulertigs {
        debug_assert!(!eulertig.is_empty());
        debug_assert!(graph.edge_data(*eulertig.first().unwrap()).is_original());
        debug_assert!(graph.edge_data(*eulertig.last().unwrap()).is_original());
    }

    eulertigs
}
