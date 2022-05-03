use crate::TigAlgorithm;
use genome_graph::bigraph::algo::walk_cover::arbitrary_biwalk_cover;
use genome_graph::bigraph::interface::static_bigraph::StaticEdgeCentricBigraph;
use genome_graph::bigraph::interface::BidirectedData;
use genome_graph::bigraph::traitgraph::interface::GraphBase;
use genome_graph::bigraph::traitgraph::walks::VecEdgeWalk;

/// The pathtig algorithm computes a heuristically small set of edge-disjoint paths, similar to simplitigs and UST-tigs.
/// This algorithm does not alter the graph.
#[derive(Default)]
pub struct PathtigAlgorithm;

impl<Graph: GraphBase> TigAlgorithm<Graph> for PathtigAlgorithm
where
    Graph: StaticEdgeCentricBigraph,
    Graph::EdgeData: BidirectedData + Eq,
{
    type Configuration = ();

    fn compute_tigs(
        graph: &mut Graph,
        _configuration: &Self::Configuration,
    ) -> Vec<VecEdgeWalk<Graph>> {
        compute_pathtigs(graph)
    }
}

/// Compute pathtigs for the given graph.
/// This is a heuristically small set of edge-disjoint paths, similar to simplitigs and UST-tigs.
fn compute_pathtigs<
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
