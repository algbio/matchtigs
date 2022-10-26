//! A library containing the matchtigs algorithms.
//! Use this to compute pathtigs (similar to simplitigs and UST-tigs), greedy matchtigs and matchtigs for arbitrary graph types.

#![warn(missing_docs)]

#[macro_use]
extern crate log;

pub mod clib;
mod implementation;

pub use implementation::{
    eulertigs::EulertigAlgorithm, eulertigs::EulertigAlgorithmConfiguration,
    greedytigs::GreedytigAlgorithm, greedytigs::GreedytigAlgorithmConfiguration,
    matchtigs::MatchtigAlgorithm, matchtigs::MatchtigAlgorithmConfiguration,
    pathtigs::PathtigAlgorithm, write_duplication_bitvector, write_duplication_bitvector_to_file,
    HeapType, MatchtigEdgeData, NodeWeightArrayType, TigAlgorithm,
};
