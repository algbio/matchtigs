//! A library containing the matchtigs algorithms.
//! Use this to compute pathtigs (similar to simplitigs and UST-tigs), greedy matchtigs and matchtigs for arbitrary graph types.

#![warn(missing_docs)]

#[macro_use]
extern crate log;

pub mod clib;
mod implementation;

pub use implementation::{
    GreedytigAlgorithm, GreedytigAlgorithmConfiguration, HeapType, MatchtigAlgorithm,
    MatchtigAlgorithmConfiguration, NodeWeightArrayType, PathtigAlgorithm, TigAlgorithm,
};
