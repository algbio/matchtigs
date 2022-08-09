//! The command line interface of the matchtigs crate.
//! This interface allows convenient access to the implemented algorithms without writing any code of your own.
//! Supported file formats are fasta and GFA.

#![warn(missing_docs)]

use crate::implementation::eulertigs::{EulertigAlgorithm, EulertigAlgorithmConfiguration};
use crate::implementation::greedytigs::{GreedytigAlgorithm, GreedytigAlgorithmConfiguration};
use crate::implementation::matchtigs::{MatchtigAlgorithm, MatchtigAlgorithmConfiguration};
use crate::implementation::pathtigs::PathtigAlgorithm;
use crate::implementation::{
    initialise_logging, write_duplication_bitvector_to_file, HeapType, MatchtigEdgeData,
    NodeWeightArrayType, PerformanceDataType, TigAlgorithm,
};
use clap::Parser;
use genome_graph::bigraph::interface::BidirectedData;
use genome_graph::bigraph::traitgraph::index::GraphIndex;
use genome_graph::bigraph::traitgraph::interface::ImmutableGraphContainer;
use genome_graph::bigraph::traitgraph::interface::StaticGraph;
use genome_graph::bigraph::traitgraph::traitsequence::interface::Sequence;
use genome_graph::bigraph::traitgraph::walks::EdgeWalk;
use genome_graph::compact_genome::implementation::{DefaultGenome, DefaultSequenceStore};
use genome_graph::compact_genome::interface::alphabet::dna_alphabet::DnaAlphabet;
use genome_graph::compact_genome::interface::alphabet::Alphabet;
use genome_graph::compact_genome::interface::sequence::{GenomeSequence, OwnedGenomeSequence};
use genome_graph::compact_genome::interface::sequence_store::{HandleWithLength, SequenceStore};
use genome_graph::io::bcalm2::{
    read_bigraph_from_bcalm2_as_edge_centric_from_file, PlainBCalm2NodeData,
};
use genome_graph::io::fasta::{read_bigraph_from_fasta_as_edge_centric_from_file, FastaNodeData};
use genome_graph::io::gfa::{read_gfa_as_edge_centric_bigraph_from_file, BidirectedGfaNodeData};
use genome_graph::io::SequenceData;
use log::LevelFilter;
use std::fmt::Debug;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::{Path, PathBuf};
use traitgraph_algo::dijkstra::DijkstraWeightedEdgeData;

#[macro_use]
extern crate log;
//#[macro_use]
//extern crate clap;

mod implementation;

/// The command line parser.
#[derive(Parser, Debug)]
#[clap(
    version = env!("CARGO_PKG_VERSION"),
    author = "Sebastian Schmidt <sebastian.schmidt@helsinki.fi>",
    about = "Matchtigs: minimum plain text representation of kmer sets.",
)]
pub struct Cli {
    /// GFA file containing the input unitigs.
    /// Either a GFA input file a fasta input file, or a bcalm input file must be given.
    #[clap(long, conflicts_with_all = &["fa-in", "bcalm-in"], required_unless_present_any = &["fa-in", "bcalm-in"])]
    gfa_in: Option<PathBuf>,

    /// Fasta file containing the input unitigs.
    /// If possible, pass GFA or bcalm2 fasta files, as those contain the topology of the graph,
    /// speeding up the parsing process.
    /// Either a GFA input file a fasta input file, or a bcalm input file must be given.
    #[clap(
        long,
        conflicts_with_all = &["gfa-in", "bcalm-in"],
        required_unless_present_any = &["gfa-in", "bcalm-in"],
        requires = "k"
    )]
    fa_in: Option<PathBuf>,

    /// Bcalm2 Fasta file containing the input unitigs.
    /// Bcalm2 encodes the topology of the graph inside the fasta file, which makes using this
    /// option faster than `--fa-in` for bcalm2 fasta files.
    /// Either a GFA input file a fasta input file, or a bcalm input file must be given.
    #[clap(
    long,
    conflicts_with_all = &["fa-in", "gfa-in"],
    required_unless_present_any = &["fa-in", "gfa-in"],
    requires = "k"
    )]
    bcalm_in: Option<PathBuf>,

    /// Compute pathtigs and write them to the given file in GFA format.
    #[clap(long)]
    pathtigs_gfa_out: Option<PathBuf>,

    /// Compute pathtigs and write them to the given file in fasta format.
    #[clap(long)]
    pathtigs_fa_out: Option<PathBuf>,

    /// Compute eulertigs and write them to the given file in GFA format.
    #[clap(long)]
    eulertigs_gfa_out: Option<PathBuf>,

    /// Compute eulertigs and write them to the given file in fasta format.
    #[clap(long)]
    eulertigs_fa_out: Option<PathBuf>,

    /// Compute greedy matchtigs and write them to the given file in GFA format.
    #[clap(long)]
    greedytigs_gfa_out: Option<PathBuf>,

    /// Compute greedy matchtigs and write them to the given file in fasta format.
    #[clap(long)]
    greedytigs_fa_out: Option<PathBuf>,

    /// Compute matchtigs and write them to the given file in GFA format.
    #[clap(long)]
    matchtigs_gfa_out: Option<PathBuf>,

    /// Compute matchtigs and write them to the given file in fasta format.
    #[clap(long)]
    matchtigs_fa_out: Option<PathBuf>,

    /// Output a file with bitvectors in ASCII format, with a 0 for each duplicated instance of a kmer in the greedytigs.
    /// The bitvectors are separated by newline characters.
    /// Taking all kmers with a 1 results in a set of all original kmers with no duplicates.
    #[clap(long)]
    greedytigs_duplication_bitvector_out: Option<PathBuf>,

    /// Output a file with bitvectors in ASCII format, with a 0 for each duplicated instance of a kmer in the matchtigs.
    /// The bitvectors are separated by newline characters.
    /// Taking all kmers with a 1 results in a set of all original kmers with no duplicates.
    #[clap(long)]
    matchtigs_duplication_bitvector_out: Option<PathBuf>,

    /// The kmer size used to compute the input unitigs.
    /// This is required when using a fasta file as input.
    /// GFA files contain the required information.
    #[clap(short, long)]
    k: Option<usize>,

    /// The number of threads used to compute greedy matchtigs and matchtigs.
    #[clap(short, long, default_value = "1")]
    threads: usize,

    /// The command used to run blossom5.
    #[clap(long, default_value = "blossom5")]
    blossom5_command: PathBuf,

    /// The data structure to store the weight of visited nodes in Dijkstra's algorithm.
    #[clap(long, default_value = "EpochNodeWeightArray")]
    dijkstra_node_weight_array_type: NodeWeightArrayType,

    /// The heap data structure used by Dijkstra's algorithm.
    #[clap(long, default_value = "StdBinaryHeap")]
    dijkstra_heap_type: HeapType,

    /// The performance data collector used by Dijkstra's algorithm.
    #[clap(long, default_value = "None")]
    dijkstra_performance_data_type: PerformanceDataType,

    /// If given enables staged parallelism mode.
    /// In this mode, the Dijkstras are executed first with full parallelism
    /// (according to the given number of threads) but with limited memory resources
    /// (see `--dijkstra-resource-limit-factor`).
    /// Then the number of threads is divided by this number, and the Dijkstras that
    /// failed before due to resource limitations are retried with more resources.
    /// The resources are increased relative to the number of threads, i.e. if the number
    /// of threads is divided by four, then the resources each thread has is multiplied by four.
    #[clap(long)]
    dijkstra_staged_parallelism_divisor: Option<f64>,

    /// Limits the memory used by each Dijkstra execution if in staged parallelism mode
    /// (see `--dijkstra-staged-parallelism-divisor`).
    /// Each thread is allowed to use queue space as well as distance array space to store up to
    /// `NODES * FACTOR / THREADS` nodes.
    /// `NODES` is the number of nodes, `FACTOR` is the factor given by this parameter,
    /// and `THREADS` is the number of threads.
    /// The number of threads decreases in each stage of execution as described in
    /// `--dijkstra-staged-parallelism-divisor`.
    #[clap(long, default_value = "1")]
    dijkstra_resource_limit_factor: usize,

    /// Print the de Bruijn graph constructed from the input unitigs.
    #[clap(long)]
    debug_print_graph: bool,

    /// Print the tigs as sequences of edge ids.
    #[clap(long)]
    debug_print_walks: bool,

    #[clap(long, default_value = "Info")]
    log_level: LevelFilter,
}

/// Edge data of a graph.
/// It contains all the information required to run the compression algorithms, and output the tigs afterwards.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
pub struct CliEdgeData<SequenceHandle> {
    /// A handle to the sequence represented by this edge.
    sequence_handle: SequenceHandle,
    /// True if the edge represents the sequence pointed to by the handle, false if it represents the reverse complement.
    forward: bool,
    /// The weight of the edge.
    /// This is the number of kmers represented by this edge.
    weight: usize,
    /// 0 for original edges, >0 for dummies.
    dummy_edge_id: usize,
}

impl<SequenceHandle> DijkstraWeightedEdgeData<usize> for CliEdgeData<SequenceHandle> {
    fn weight(&self) -> usize {
        self.weight
    }
}

impl<SequenceHandle: Clone> BidirectedData for CliEdgeData<SequenceHandle> {
    fn mirror(&self) -> Self {
        let mut result = self.clone();
        result.forward = !result.forward;
        result
    }
}

impl<AlphabetType: Alphabet, GenomeSequenceStore: SequenceStore<AlphabetType>>
    SequenceData<AlphabetType, GenomeSequenceStore> for CliEdgeData<GenomeSequenceStore::Handle>
{
    fn sequence_handle(&self) -> &<GenomeSequenceStore as SequenceStore<AlphabetType>>::Handle {
        &self.sequence_handle
    }

    fn sequence_ref<'a>(
        &self,
        source_sequence_store: &'a GenomeSequenceStore,
    ) -> Option<&'a <GenomeSequenceStore as SequenceStore<AlphabetType>>::SequenceRef> {
        if self.forward {
            let handle =
                <Self as SequenceData<AlphabetType, GenomeSequenceStore>>::sequence_handle(self);
            Some(source_sequence_store.get(handle))
        } else {
            None
        }
    }

    fn sequence_owned<
        ResultSequence: for<'a> OwnedGenomeSequence<'a, AlphabetType, ResultSubsequence>,
        ResultSubsequence: for<'a> GenomeSequence<'a, AlphabetType, ResultSubsequence> + ?Sized,
    >(
        &self,
        source_sequence_store: &GenomeSequenceStore,
    ) -> ResultSequence {
        let handle =
            <Self as SequenceData<AlphabetType, GenomeSequenceStore>>::sequence_handle(self);
        if self.forward {
            source_sequence_store.get(handle).convert()
        } else {
            source_sequence_store
                .get(handle)
                .convert_with_reverse_complement()
        }
    }
}

impl<SequenceHandle: Clone> MatchtigEdgeData<SequenceHandle> for CliEdgeData<SequenceHandle> {
    fn is_dummy(&self) -> bool {
        self.dummy_edge_id != 0
    }

    fn is_forwards(&self) -> bool {
        self.forward
    }

    fn new(
        sequence_handle: SequenceHandle,
        forwards: bool,
        weight: usize,
        dummy_id: usize,
    ) -> Self {
        Self {
            sequence_handle,
            forward: forwards,
            weight,
            dummy_edge_id: dummy_id,
        }
    }
}

impl<SequenceHandle, Data> From<BidirectedGfaNodeData<SequenceHandle, Data>>
    for CliEdgeData<SequenceHandle>
{
    fn from(node_data: BidirectedGfaNodeData<SequenceHandle, Data>) -> Self {
        Self {
            sequence_handle: node_data.sequence_handle,
            forward: node_data.forward,
            weight: 0,
            dummy_edge_id: 0,
        }
    }
}

impl<SequenceHandle> From<PlainBCalm2NodeData<SequenceHandle>> for CliEdgeData<SequenceHandle> {
    fn from(node_data: PlainBCalm2NodeData<SequenceHandle>) -> Self {
        Self {
            sequence_handle: node_data.sequence_handle,
            forward: node_data.forwards,
            weight: 0,
            dummy_edge_id: 0,
        }
    }
}

impl<SequenceHandle> From<FastaNodeData<SequenceHandle>> for CliEdgeData<SequenceHandle> {
    fn from(node_data: FastaNodeData<SequenceHandle>) -> Self {
        Self {
            sequence_handle: node_data.sequence_handle,
            forward: node_data.forwards,
            weight: 0,
            dummy_edge_id: 0,
        }
    }
}

/// The graph type used by the CLI.
/// It uses a petgraph with the CliEdgeData.
pub type CliGraph<GenomeSequenceStoreHandle> =
    genome_graph::bigraph::implementation::node_bigraph_wrapper::NodeBigraphWrapper<
        genome_graph::bigraph::traitgraph::implementation::petgraph_impl::PetGraph<
            (),
            CliEdgeData<GenomeSequenceStoreHandle>,
        >,
    >;

/// Compute the edge weights of the graph from the sequences represented by the edges.
/// The weight is the number of kmers represented by an edge, so it is `characters - (k - 1)`.
pub fn compute_edge_weights<
    NodeData,
    SequenceHandle: HandleWithLength,
    Graph: StaticGraph<NodeData = NodeData, EdgeData = CliEdgeData<SequenceHandle>>,
>(
    graph: &mut Graph,
    k: usize,
) {
    for edge_index in graph.edge_indices() {
        let edge_data = graph.edge_data_mut(edge_index);
        let weight = edge_data.sequence_handle.len() + 1 - k;
        debug_assert_ne!(
            weight,
            0,
            "Found sequence of length {} on edge {}",
            edge_data.sequence_handle.len(),
            edge_index.as_usize()
        );
        edge_data.weight = weight;
    }
}

/// Assert that the edge graph lables are correct.
/// This function is empty if compiled without debug assertions.
pub fn debug_assert_graph_edge_labels<
    NodeData,
    AlphabetType: Alphabet + Debug + Eq + 'static,
    GenomeSequenceStore: SequenceStore<AlphabetType>,
    EdgeData: MatchtigEdgeData<GenomeSequenceStore::Handle> + SequenceData<AlphabetType, GenomeSequenceStore>,
    Graph: StaticGraph<NodeData = NodeData, EdgeData = EdgeData>,
>(
    graph: &Graph,
    source_sequence_store: &GenomeSequenceStore,
    k: usize,
) {
    if !cfg!(debug_assertions) {
        return;
    }

    for node_index in graph.node_indices() {
        for in_neighbor in graph.in_neighbors(node_index) {
            for out_neighbor in graph.out_neighbors(node_index) {
                let first_data = graph.edge_data(in_neighbor.edge_id);
                let second_data = graph.edge_data(out_neighbor.edge_id);

                let first_data_sequence: DefaultGenome<AlphabetType> =
                    first_data.sequence_owned(source_sequence_store);
                let second_data_sequence: DefaultGenome<AlphabetType> =
                    second_data.sequence_owned(source_sequence_store);

                let first_kmer = &first_data_sequence[first_data_sequence.len() - k + 1..];
                let second_kmer = &second_data_sequence[..k - 1];

                debug_assert_eq!(first_kmer, second_kmer);
            }
        }
    }
}

/// Write a sequence of walks in fasta format.
pub fn write_walks_fasta<
    'ws,
    NodeData,
    AlphabetType: Alphabet + 'static,
    GenomeSequenceStore: SequenceStore<AlphabetType>,
    GraphEdgeData: MatchtigEdgeData<GenomeSequenceStore::Handle> + SequenceData<AlphabetType, GenomeSequenceStore>,
    Graph: StaticGraph<NodeData = NodeData, EdgeData = GraphEdgeData>,
    Walk: 'ws + for<'w> EdgeWalk<'w, Graph, SubWalk>,
    SubWalk: for<'w> EdgeWalk<'w, Graph, SubWalk> + ?Sized,
    WalkSource: IntoIterator<Item = &'ws Walk>,
    P: AsRef<Path>,
>(
    graph: &Graph,
    walks: WalkSource,
    source_sequence_store: &GenomeSequenceStore,
    k: usize,
    file_path: P,
    debug_file_path: Option<&str>,
) {
    let mut writer = BufWriter::new(File::create(file_path).unwrap());
    let mut debug_writer = debug_file_path
        .map(|debug_file_path| BufWriter::new(File::create(debug_file_path).unwrap()));

    for (i, walk) in walks.into_iter().enumerate() {
        let first_edge = *walk.first().unwrap();
        let first_data = graph.edge_data(first_edge);
        debug_assert!(first_data.is_original());
        debug_assert!(graph.edge_data(*walk.last().unwrap()).is_original());

        writeln!(writer, ">{}", i + 1).unwrap();
        if let Some(debug_writer) = &mut debug_writer {
            writeln!(debug_writer, "tig {}", i + 1).unwrap();
        }

        let first_data_sequence: DefaultGenome<AlphabetType> =
            first_data.sequence_owned(source_sequence_store);
        let first_data_sequence = first_data_sequence.as_string();

        write!(writer, "{}", first_data_sequence).unwrap();
        if let Some(debug_writer) = &mut debug_writer {
            write!(
                debug_writer,
                "| {}{} {} ",
                first_edge.as_usize(),
                if first_data.is_forwards() { "f" } else { "r" },
                first_data_sequence,
            )
            .unwrap();
        }

        let mut previous = *walk.first().unwrap();
        for &current in walk.iter().skip(1) {
            let previous_data = graph.edge_data(previous);
            let current_data = graph.edge_data(current);
            debug_assert!(previous_data.is_original() || current_data.is_original());

            if current_data.is_dummy() {
                if let Some(debug_writer) = &mut debug_writer {
                    write!(
                        debug_writer,
                        "| skip dummy {} weight {} ",
                        current.as_usize(),
                        current_data.weight(),
                    )
                    .unwrap();
                }
                previous = current;
                continue;
            }

            let offset = if previous_data.is_original() {
                k - 1
            } else {
                k - 1 - previous_data.weight()
            };

            let current_data_sequence: DefaultGenome<AlphabetType> =
                current_data.sequence_owned(source_sequence_store);
            let current_data_sequence = &current_data_sequence[offset..].as_string();

            if let Some(debug_writer) = &mut debug_writer {
                write!(
                    debug_writer,
                    "| {}{}:off {} {} ",
                    current.as_usize(),
                    if current_data.is_forwards() { "f" } else { "r" },
                    offset,
                    current_data_sequence
                )
                .unwrap();
            }

            write!(writer, "{}", current_data_sequence).unwrap();

            previous = current;
        }
        writeln!(writer).unwrap();
        if let Some(debug_writer) = &mut debug_writer {
            writeln!(debug_writer).unwrap();
        }
    }
}

/// Write a set of walks in GFA format.
pub fn write_walks_gfa<
    'ws,
    NodeData,
    AlphabetType: Alphabet + 'static,
    GenomeSequenceStore: SequenceStore<AlphabetType>,
    GraphEdgeData: MatchtigEdgeData<GenomeSequenceStore::Handle> + SequenceData<AlphabetType, GenomeSequenceStore>,
    Graph: StaticGraph<NodeData = NodeData, EdgeData = GraphEdgeData>,
    Walk: 'ws + for<'w> EdgeWalk<'w, Graph, SubWalk>,
    SubWalk: for<'w> EdgeWalk<'w, Graph, SubWalk> + ?Sized,
    WalkSource: IntoIterator<Item = &'ws Walk>,
    P: AsRef<Path>,
>(
    graph: &Graph,
    walks: WalkSource,
    source_sequence_store: &GenomeSequenceStore,
    k: usize,
    header: &Option<String>,
    file_path: P,
    debug_file_path: Option<&str>,
)
//where <GenomeSequenceStore as SequenceStore<AlphabetType>>::SequenceRef: Index<RangeFrom<usize>,Output =<GenomeSequenceStore as SequenceStore<AlphabetType>>::SequenceRef >,
{
    let mut writer = BufWriter::new(File::create(file_path).unwrap());
    let mut debug_writer = debug_file_path
        .map(|debug_file_path| BufWriter::new(File::create(debug_file_path).unwrap()));

    let header = if let Some(header) = header {
        header.clone()
    } else {
        format!("H\tKL:Z:{}", k)
    };
    writeln!(writer, "{}", header).unwrap();
    if let Some(debug_writer) = &mut debug_writer {
        writeln!(debug_writer, "{}", header).unwrap();
    }

    for (i, walk) in walks.into_iter().enumerate() {
        let first_edge = *walk.first().unwrap();
        let first_data = graph.edge_data(first_edge);
        debug_assert!(first_data.is_original());
        debug_assert!(graph.edge_data(*walk.last().unwrap()).is_original());

        write!(writer, "S\t{}\t", i + 1).unwrap();
        if let Some(debug_writer) = &mut debug_writer {
            writeln!(debug_writer, "tig {}", i + 1).unwrap();
        }

        let first_data_sequence: DefaultGenome<AlphabetType> =
            first_data.sequence_owned(source_sequence_store);
        let first_data_sequence = first_data_sequence.as_string();

        write!(writer, "{}", first_data_sequence).unwrap();
        if let Some(debug_writer) = &mut debug_writer {
            write!(
                debug_writer,
                "| {}{} {} ",
                first_edge.as_usize(),
                if first_data.is_forwards() { "f" } else { "r" },
                first_data_sequence,
            )
            .unwrap();
        }

        let mut previous = *walk.first().unwrap();
        for &current in walk.iter().skip(1) {
            let previous_data = graph.edge_data(previous);
            let current_data = graph.edge_data(current);
            debug_assert!(previous_data.is_original() || current_data.is_original());

            if current_data.is_dummy() {
                if let Some(debug_writer) = &mut debug_writer {
                    write!(
                        debug_writer,
                        "| skip dummy {} weight {} ",
                        current.as_usize(),
                        current_data.weight(),
                    )
                    .unwrap();
                }
                previous = current;
                continue;
            }

            let offset = if previous_data.is_original() {
                k - 1
            } else {
                k - 1 - previous_data.weight()
            };

            if let Some(current_data_sequence) = current_data.sequence_ref(source_sequence_store) {
                let current_data_sequence =
                    &current_data_sequence[offset..current_data_sequence.len()];

                if let Some(debug_writer) = &mut debug_writer {
                    write!(
                        debug_writer,
                        "| {}{}:off {} {} ",
                        current.as_usize(),
                        if current_data.is_forwards() { "f" } else { "r" },
                        offset,
                        String::from_utf8(
                            current_data_sequence
                                .iter()
                                .cloned()
                                .map(AlphabetType::character_to_ascii)
                                .collect::<Vec<_>>()
                        )
                        .unwrap(),
                    )
                    .unwrap();
                }

                for character in current_data_sequence.iter() {
                    writer
                        .write_all(&[AlphabetType::character_to_ascii(character.clone())])
                        .unwrap();
                }
            } else {
                assert!(current_data.is_backwards());

                let handle = current_data.sequence_handle();
                let sequence_ref = source_sequence_store.get(handle);

                if let Some(debug_writer) = &mut debug_writer {
                    write!(
                        debug_writer,
                        "| {}{}:off {} {} ",
                        current.as_usize(),
                        if current_data.is_forwards() { "f" } else { "r" },
                        offset,
                        String::from_utf8(
                            sequence_ref
                                .reverse_complement_iter()
                                .map(AlphabetType::character_to_ascii)
                                .collect::<Vec<_>>()
                        )
                        .unwrap(),
                    )
                    .unwrap();
                }

                for character in sequence_ref.reverse_complement_iter() {
                    writer
                        .write_all(&[AlphabetType::character_to_ascii(character)])
                        .unwrap();
                }
            }

            previous = current;
        }
        writeln!(writer).unwrap();
        if let Some(debug_writer) = &mut debug_writer {
            writeln!(debug_writer).unwrap();
        }
    }
}

fn debug_print_walks<
    'ws,
    Graph: ImmutableGraphContainer,
    Walk: 'ws + for<'w> EdgeWalk<'w, Graph, Subwalk>,
    Subwalk: for<'w> EdgeWalk<'w, Graph, Subwalk> + ?Sized,
    WalkSource: 'ws + IntoIterator<Item = &'ws Walk>,
>(
    _graph: &Graph,
    walks: WalkSource,
) {
    info!("Printing walks to stdout, because --debug-print-walks was set");
    for walk in walks.into_iter() {
        for (i, edge_index) in walk.iter().enumerate() {
            if i != 0 {
                print!(", ");
            }
            print!("{}", edge_index.as_usize());
        }
        println!();
    }
}

fn main() {
    let opts: Cli = Cli::parse();
    initialise_logging();

    debug!("Command line options:\n{opts:#?}");

    // Load graph
    let mut sequence_store = DefaultSequenceStore::<DnaAlphabet>::default();

    let (mut graph, k, gfa_header): (CliGraph<_>, _, _) = if let Some(gfa_in) = &opts.gfa_in {
        info!("Reading gfa as edge centric bigraph from {gfa_in:?}");
        let (graph, gfa_read_file_properties) =
            read_gfa_as_edge_centric_bigraph_from_file(&gfa_in, &mut sequence_store, false)
                .unwrap();
        let k = gfa_read_file_properties.k;
        if let Some(required_k) = opts.k {
            debug_assert_eq!(k, required_k);
        }
        let gfa_header = gfa_read_file_properties.header.unwrap();

        (graph, k, Some(gfa_header))
    } else if let Some(fa_in) = &opts.fa_in {
        let k = opts.k.unwrap();
        info!("Reading fa as edge centric bigraph with k = {k} from {fa_in:?}");
        let graph =
            read_bigraph_from_fasta_as_edge_centric_from_file(&fa_in, &mut sequence_store, k)
                .unwrap();
        (graph, k, None)
    } else if let Some(bcalm_in) = &opts.bcalm_in {
        let k = opts.k.unwrap();
        info!("Reading bcalm2 fa as edge centric bigraph with k = {k} from {bcalm_in:?}");
        let graph =
            read_bigraph_from_bcalm2_as_edge_centric_from_file(&bcalm_in, &mut sequence_store, k)
                .unwrap();
        (graph, k, None)
    } else {
        unreachable!("Excluded by cli conditions");
    };

    let unitigs_size_in_memory = sequence_store.size_in_memory();
    let unitigs_size_in_memory_mib = unitigs_size_in_memory / (1024 * 1024);
    info!("The sequences take a total {unitigs_size_in_memory_mib}MiB of memory");
    info!("k = {k}");
    info!(
        "Graph has {} nodes and {} edges",
        graph.node_count(),
        graph.edge_count()
    );
    debug_assert_graph_edge_labels(&graph, &sequence_store, k);

    if opts.debug_print_graph {
        info!("Printing graph to stdout, because --debug-print-graph was set");
        for edge_index in graph.edge_indices() {
            let endpoints = graph.edge_endpoints(edge_index);
            let edge_data = graph.edge_data(edge_index);
            let sequence = sequence_store.get(&edge_data.sequence_handle);
            println!(
                "{} ({} -> {}) {}",
                edge_index.as_usize(),
                endpoints.from_node.as_usize(),
                endpoints.to_node.as_usize(),
                sequence.as_string()
            );
        }
    }

    let do_compute_pathtigs = opts.pathtigs_fa_out.is_some() || opts.pathtigs_gfa_out.is_some();
    let do_compute_eulertigs = opts.eulertigs_fa_out.is_some() || opts.eulertigs_gfa_out.is_some();
    let do_compute_greedytigs = opts.greedytigs_fa_out.is_some()
        || opts.greedytigs_gfa_out.is_some()
        || opts.greedytigs_duplication_bitvector_out.is_some();
    let do_compute_matchtigs = opts.matchtigs_fa_out.is_some()
        || opts.matchtigs_gfa_out.is_some()
        || opts.matchtigs_duplication_bitvector_out.is_some();

    if do_compute_pathtigs {
        info!("Computing pathtigs");
        let pathtigs = PathtigAlgorithm::compute_tigs(&mut graph, &());

        if let Some(fa_out) = &opts.pathtigs_fa_out {
            info!("Writing pathtigs as fasta to {fa_out:?}");
            write_walks_fasta(&graph, &pathtigs, &sequence_store, k, fa_out, None);
        }

        if let Some(gfa_out) = &opts.pathtigs_gfa_out {
            info!("Writing pathtigs as gfa to {gfa_out:?}");
            write_walks_gfa(
                &graph,
                &pathtigs,
                &sequence_store,
                k,
                &gfa_header,
                gfa_out,
                None,
            );
        }

        if opts.debug_print_walks {
            debug_print_walks(&graph, &pathtigs);
        }
    }

    if do_compute_eulertigs {
        info!("Computing eulertigs");
        let eulertigs =
            EulertigAlgorithm::compute_tigs(&mut graph, &EulertigAlgorithmConfiguration { k });

        if let Some(fa_out) = &opts.eulertigs_fa_out {
            info!("Writing eulertigs as fasta to {fa_out:?}");
            write_walks_fasta(&graph, &eulertigs, &sequence_store, k, fa_out, None);
        }

        if let Some(gfa_out) = &opts.eulertigs_gfa_out {
            info!("Writing eulertigs as gfa to {gfa_out:?}");
            write_walks_gfa(
                &graph,
                &eulertigs,
                &sequence_store,
                k,
                &gfa_header,
                gfa_out,
                None,
            );
        }

        if opts.debug_print_walks {
            debug_print_walks(&graph, &eulertigs);
        }
    }

    if do_compute_greedytigs || do_compute_matchtigs {
        // Find shortest paths between nodes with missing edges
        info!("Computing edge weights for shortest path queries");
        compute_edge_weights(&mut graph, k);
    }

    if do_compute_greedytigs {
        info!("Computing greedytigs");
        let mut graph = graph.clone();
        let greedytigs = GreedytigAlgorithm::compute_tigs(
            &mut graph,
            &GreedytigAlgorithmConfiguration {
                threads: opts.threads,
                k,
                staged_parallelism_divisor: opts.dijkstra_staged_parallelism_divisor,
                resource_limit_factor: opts.dijkstra_resource_limit_factor,
                heap_type: opts.dijkstra_heap_type,
                node_weight_array_type: opts.dijkstra_node_weight_array_type,
                performance_data_type: opts.dijkstra_performance_data_type,
            },
        );

        if let Some(fa_out) = &opts.greedytigs_fa_out {
            info!("Writing greedytigs as fasta to {fa_out:?}");
            write_walks_fasta(&graph, &greedytigs, &sequence_store, k, fa_out, None);
        }

        if let Some(gfa_out) = &opts.greedytigs_gfa_out {
            info!("Writing greedytigs as gfa to {gfa_out:?}");
            write_walks_gfa(
                &graph,
                &greedytigs,
                &sequence_store,
                k,
                &gfa_header,
                gfa_out,
                None,
            );
        }

        if let Some(duplication_bitvector_out) = &opts.greedytigs_duplication_bitvector_out {
            info!("Writing greedytig duplication bitvector to {duplication_bitvector_out:?}");
            write_duplication_bitvector_to_file(&graph, &greedytigs, duplication_bitvector_out)
                .unwrap();
        }

        if opts.debug_print_walks {
            debug_print_walks(&graph, &greedytigs);
        }
    }

    if do_compute_matchtigs {
        info!("Computing matchtigs");
        let mut graph = graph.clone();
        let matchtigs = MatchtigAlgorithm::compute_tigs(
            &mut graph,
            &MatchtigAlgorithmConfiguration {
                threads: opts.threads,
                k,
                node_weight_array_type: opts.dijkstra_node_weight_array_type,
                heap_type: opts.dijkstra_heap_type,
                matching_file_prefix: opts
                    .matchtigs_fa_out
                    .as_ref()
                    .unwrap_or_else(|| opts.matchtigs_gfa_out.as_ref().unwrap()),
                matcher_path: &opts.blossom5_command,
            },
        );

        if let Some(fa_out) = &opts.matchtigs_fa_out {
            info!("Writing matchtigs as fasta to {fa_out:?}");
            write_walks_fasta(&graph, &matchtigs, &sequence_store, k, fa_out, None);
        }

        if let Some(gfa_out) = &opts.matchtigs_gfa_out {
            info!("Writing matchtigs as gfa to {gfa_out:?}");
            write_walks_gfa(
                &graph,
                &matchtigs,
                &sequence_store,
                k,
                &gfa_header,
                gfa_out,
                None,
            );
        }

        if let Some(duplication_bitvector_out) = &opts.matchtigs_duplication_bitvector_out {
            info!("Writing matchtig duplication bitvector to {duplication_bitvector_out:?}");
            write_duplication_bitvector_to_file(&graph, &matchtigs, duplication_bitvector_out)
                .unwrap();
        }

        if opts.debug_print_walks {
            debug_print_walks(&graph, &matchtigs);
        }
    }

    info!("Done");
}
