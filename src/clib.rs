//! Bindings of our algorithms for the C language.
//! These allow to pass the graph topology as a list of edges from a node-centric de Bruijn graph.
//! There is only one global graph that can be controlled via the bindings, no two graphs can be created at the same time.
//!
//! WARNING: These functions have not been tested properly and might produce unexpected results.

use crate::implementation::{
    compute_greedytigs, compute_matchtigs, compute_pathtigs, initialise_logging, MatchtigEdgeData,
};
use disjoint_sets::UnionFind;
use genome_graph::bigraph::implementation::node_bigraph_wrapper::PetBigraph;
use genome_graph::bigraph::interface::dynamic_bigraph::DynamicBigraph;
use genome_graph::bigraph::interface::static_bigraph::{StaticBigraph, StaticEdgeCentricBigraph};
use genome_graph::bigraph::interface::BidirectedData;
use genome_graph::bigraph::traitgraph::index::GraphIndex;
use genome_graph::bigraph::traitgraph::interface::{
    GraphBase, ImmutableGraphContainer, MutableGraphContainer,
};
use genome_graph::bigraph::traitgraph::traitsequence::interface::Sequence;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::time::Instant;
use traitgraph_algo::dijkstra::DijkstraWeightedEdgeData;

type MatchtigGraph = PetBigraph<(), ExternalEdgeData>;

static mut GRAPH: Option<MatchtigGraph> = None;
static mut UNION_FIND: Option<UnionFind> = None;

fn get_graph<'a>() -> &'a mut MatchtigGraph {
    unsafe {
        if GRAPH.is_none() {
            GRAPH = Some(Default::default());
        }

        GRAPH.as_mut().unwrap()
    }
}

#[inline]
fn get_union_find<'a>() -> &'a mut UnionFind {
    unsafe {
        if UNION_FIND.is_none() {
            UNION_FIND = Some(Default::default());
        }

        UNION_FIND.as_mut().unwrap()
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
struct ExternalEdgeData {
    weight: usize,
    /// 0 for original edges, >0 for dummies.
    dummy_edge_id: usize,
    unitig_id: usize,
    forwards: bool,
}

impl MatchtigEdgeData<usize> for ExternalEdgeData {
    fn is_dummy(&self) -> bool {
        self.dummy_edge_id != 0
    }

    fn is_forwards(&self) -> bool {
        self.forwards
    }

    fn new(sequence_handle: usize, forwards: bool, weight: usize, dummy_id: usize) -> Self {
        Self {
            weight,
            dummy_edge_id: dummy_id,
            unitig_id: sequence_handle,
            forwards,
        }
    }
}

impl BidirectedData for ExternalEdgeData {
    fn mirror(&self) -> Self {
        let mut result = self.clone();
        result.forwards = !result.forwards;
        result
    }
}

impl DijkstraWeightedEdgeData<usize> for ExternalEdgeData {
    fn weight(&self) -> usize {
        self.weight
    }
}

/// Initialise the data structures and the logging mechanism of this library.
/// Call this exactly once before interacting with this library.
#[no_mangle]
pub extern "C" fn matchtigs_initialise() {
    initialise_logging();
    get_graph();
}

/// Initialise the data structures used to build the graph.
/// Call this whenever you want to build a new graph.
/// This replaces the old graph.
#[no_mangle]
pub extern "C" fn matchtigs_initialise_graph(unitig_amount: usize) {
    *get_union_find() = UnionFind::new(unitig_amount * 4);
}

#[inline]
const fn unitig_forward_in_node(unitig: usize) -> usize {
    unitig * 4
}

#[inline]
const fn unitig_forward_out_node(unitig: usize) -> usize {
    unitig * 4 + 2
}

#[inline]
const fn unitig_backward_in_node(unitig: usize) -> usize {
    unitig * 4 + 3
}

#[inline]
const fn unitig_backward_out_node(unitig: usize) -> usize {
    unitig * 4 + 1
}

/// Pass an edge to the graph builder.
/// The edge is from `unitig_a` to `unitig_b`, identified by their id in the closed interval `[0, unitig_amount - 1]`.
/// The strands indicate that the forward variant of the unitig is incident to the edge if `True`,
/// and that the reverse complement variant of the unitig is incident to the edge if `False`.
/// This requires that `matchtigs_initialise_graph` was called before.
#[no_mangle]
pub extern "C" fn matchtigs_merge_nodes(
    unitig_a: usize,
    strand_a: bool,
    unitig_b: usize,
    strand_b: bool,
) {
    //debug!("Merging {}{} with {}{}", unitig_a, if strand_a {"+"} else {"-"}, unitig_b, if strand_b {"+"} else {"-"});

    let out_a = if strand_a {
        unitig_forward_out_node(unitig_a)
    } else {
        unitig_backward_out_node(unitig_a)
    };
    let in_b = if strand_b {
        unitig_forward_in_node(unitig_b)
    } else {
        unitig_backward_in_node(unitig_b)
    };

    let mirror_in_a = if strand_a {
        unitig_backward_in_node(unitig_a)
    } else {
        unitig_forward_in_node(unitig_a)
    };
    let mirror_out_b = if strand_b {
        unitig_backward_out_node(unitig_b)
    } else {
        unitig_forward_out_node(unitig_b)
    };

    //debug!("Unioning ({}, {}) and ({}, {})", out_a, in_b, mirror_in_a, mirror_out_b);
    get_union_find().union(out_a, in_b);
    get_union_find().union(mirror_in_a, mirror_out_b);
}

/// Call this after passing all edges with `matchtigs_merge_nodes`.
/// `unitig_weights` must be an array of length `unitig_amount` containing the number of kmers in each unitig.
/// The entry at position `i` must belong to the unitig with index `i`.
///
/// # Safety
/// Unsafe because it dereferences the given raw pointer, with an offset of up to `unitig_amount - 1`.
#[no_mangle]
pub unsafe extern "C" fn matchtigs_build_graph(unitig_weights: *const usize) {
    let start = Instant::now();

    let unitig_weights = {
        assert!(!unitig_weights.is_null());

        std::slice::from_raw_parts(unitig_weights, get_union_find().len() / 4)
    };

    get_union_find().force();
    let mut representatives = get_union_find().to_vec();
    representatives.sort_unstable();
    representatives.dedup();

    for _ in 0..representatives.len() {
        get_graph().add_node(());
    }

    for (unitig_id, &unitig_weight) in unitig_weights.iter().enumerate() {
        let n1: <MatchtigGraph as GraphBase>::NodeIndex = representatives
            .binary_search(&get_union_find().find(unitig_forward_in_node(unitig_id)))
            .unwrap()
            .into();
        let n2: <MatchtigGraph as GraphBase>::NodeIndex = representatives
            .binary_search(&get_union_find().find(unitig_forward_out_node(unitig_id)))
            .unwrap()
            .into();
        let mirror_n2: <MatchtigGraph as GraphBase>::NodeIndex = representatives
            .binary_search(&get_union_find().find(unitig_backward_in_node(unitig_id)))
            .unwrap()
            .into();
        let mirror_n1: <MatchtigGraph as GraphBase>::NodeIndex = representatives
            .binary_search(&get_union_find().find(unitig_backward_out_node(unitig_id)))
            .unwrap()
            .into();

        get_graph().set_mirror_nodes(n1, mirror_n1);
        get_graph().set_mirror_nodes(n2, mirror_n2);

        get_graph().add_edge(
            n1,
            n2,
            ExternalEdgeData::new(unitig_id, true, unitig_weight, 0),
        );
        get_graph().add_edge(
            mirror_n2,
            mirror_n1,
            ExternalEdgeData::new(unitig_id, false, unitig_weight, 0),
        );
    }

    assert!(get_graph().verify_node_pairing());
    assert!(get_graph().verify_edge_mirror_property());

    let end = Instant::now();
    info!(
        "Took {:.6}s to build the tig graph",
        (end - start).as_secs_f64()
    );
}

/// Compute tigs in the created graph.
/// Requires that `matchtigs_build_graph` was called before.
/// The `tig_algorithm` should be `1` for unitigs, `2` for pathtigs (similar to UST-tigs and simplitigs), `3` for greedy matchtigs and `4` for matchtigs.
/// `matchtig_file_prefix` must be a path to a file used to communicate with the matcher (blossom5).
/// `matcher_path` must be a path pointing to a binary of blossom5.
///
/// The output is passed through the last three parameters `tigs_edge_out`, `tigs_insert_out` and `tigs_out_limits`,
/// with the number of output tigs being the return value of this function.
/// Tigs are stored as consecutive subarrays of `tigs_edge_out` and `tigs_insert_out`, with the indices indicated by `tigs_out_limits`.
/// For example, if this function returns 2, then the first three values of `tigs_out_limits` are valid.
/// If those are e.g. `[0, 3, 5]`, then the edges of the first tig are in `tigs_edge_out` at indices `0` to `2` (inclusive), and the indices of the second tig are at indices `3` to `4` (inclusive).
/// If an edge index is negative, it means that the reverse complement of the corresponding unitig is used.
/// In `tigs_insert_out`, there is a positive value if the edge is a dummy edge, indicating how many kmers it contains.
/// If the value is zero, the edge is not a dummy edge but corresponds to a unitig, which can then be identified with `tigs_edge_out`.
///
/// # Safety
/// Unsafe because it dereferences the given raw pointers with different offsets.
#[no_mangle]
pub unsafe extern "C" fn matchtigs_compute_tigs(
    tig_algorithm: usize,
    threads: usize,
    k: usize,
    matching_file_prefix: *const c_char,
    matcher_path: *const c_char,
    tigs_edge_out: *mut isize,
    tigs_insert_out: *mut usize,
    tigs_out_limits: *mut usize,
) -> usize {
    info!("Computing tigs for k = {} and {} threads", k, threads);
    info!(
        "Graph has {} nodes and {} edges",
        get_graph().node_count(),
        get_graph().edge_count()
    );

    let matching_file_prefix = {
        assert!(!matching_file_prefix.is_null());

        CStr::from_ptr(matching_file_prefix)
    }
    .to_str()
    .unwrap();

    let matcher_path = {
        assert!(!matcher_path.is_null());

        CStr::from_ptr(matcher_path)
    }
    .to_str()
    .unwrap();

    let tigs_edge_out = {
        assert!(!tigs_edge_out.is_null());

        std::slice::from_raw_parts_mut(tigs_edge_out, get_graph().edge_count() * 2)
    };

    let tigs_insert_out = {
        assert!(!tigs_insert_out.is_null());

        std::slice::from_raw_parts_mut(tigs_insert_out, get_graph().edge_count() * 2)
    };

    let tigs_out_limits = {
        assert!(!tigs_out_limits.is_null());

        std::slice::from_raw_parts_mut(tigs_out_limits, get_graph().edge_count())
    };

    let tigs = match tig_algorithm {
        1 => get_graph()
            .edge_indices()
            .filter_map(|e| {
                if e.as_usize() % 2 == 0 {
                    Some(vec![e])
                } else {
                    None
                }
            })
            .collect(),
        2 => compute_pathtigs(get_graph()),
        3 => compute_matchtigs(get_graph(), threads, k, matching_file_prefix, matcher_path),
        4 => compute_greedytigs(get_graph(), threads, k),
        tig_algorithm => panic!("Unknown tigs algorithm identifier {}", tig_algorithm),
    };

    let mut limit = 0;
    for (i, tig) in tigs.iter().enumerate() {
        for (edge_index, &edge) in tig.iter().enumerate() {
            let edge_data = get_graph().edge_data(edge);
            tigs_edge_out[limit + edge_index] =
                edge_data.unitig_id as isize * if edge_data.is_forwards() { 1 } else { -1 };
            tigs_insert_out[limit + edge_index] = if edge_data.is_original() {
                0
            } else {
                edge_data.weight()
            };
        }
        limit += tig.len();
        tigs_out_limits[i] = limit;
    }

    tigs.len()
}
