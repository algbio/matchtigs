# Matchtigs: minimum plain text representation of kmer sets

This is an implementation of different algorithms for computing small and minimum plain text representations of kmer sets.
The algorithms expect unitigs as an input, which can e.g. be computed with [BCALM2](https://github.com/GATB/bcalm).

## Features

 * Compute [matchtigs and greedy matchtigs](https://doi.org/10.1101/2021.12.15.472871) with multiple threads
 * Compute Eulertigs
 * Compute pathtigs (a heuristic similar to [simplitigs](https://doi.org/10.1186/s13059-021-02297-z))
 * Both fasta and GFA format supported
 * Output (ASCII-) bitvectors of duplicate kmers for applications that require unique kmers

## Installation via [conda/mamba](https://docs.conda.io/en/latest/)

Install `matchtigs` with
```bash
mamba install -c conda-forge -c bioconda matchtigs
```

## Installation via [cargo](https://crates.io/)

### Requirements

Rust `>= 1.58.1`, best installed via [rustup](https://rustup.rs/).

### Installation

Install `matchtigs` with
```bash
cargo install matchtigs
```

## Usage

Computing matchtigs and greedy matchtigs from a fasta file and saving them as GFA (without topology):
```bash
matchtigs --fa-in unitigs.fa --matchtigs-gfa-out matchtigs.gfa --greedytigs-gfa-out greedy-matchtigs.gfa
```

Computing Eulertigs from a GFA file and saving them as both GFA (without topology) and fasta:
```bash
matchtigs --fa-in unitigs.fa --eulertigs-gfa-out eulertigs.gfa --eulertigs-fa-out eulertigs.fa
```

Use the `--help` option to get an overview of available options.
```bash
matchtigs --help
```

## Citation

**preprint**

Schmidt, S., Khan, S., Alanko, J., & Tomescu, A. I. (2021). Matchtigs: minimum plain text representation of kmer sets. _bioRxiv_. [https://doi.org/10.1101/2021.12.15.472871](https://doi.org/10.1101/2021.12.15.472871).
