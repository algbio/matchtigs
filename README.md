# Matchtigs & Eulertigs: minimum plain text representation of kmer sets - with and without repetitions

This is an implementation of different algorithms for computing small and minimum plain text representations of kmer sets.
The algorithms expect unitigs as an input, which can e.g. be computed with [BCALM2](https://github.com/GATB/bcalm).

## Features

 * Compute [matchtigs and greedy matchtigs](https://doi.org/10.1101/2021.12.15.472871) with multiple threads
 * Compute Eulertigs
 * Compute pathtigs (heuristical Eulertigs similar to [ProphAsm](https://doi.org/10.1186/s13059-021-02297-z))
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

**Note:** when computing unitigs with bcalm2, it is much faster to use `--bcalm-in`:
```bash
matchtigs --bcalm-in unitigs.fa --eulertigs-gfa-out eulertigs.gfa --eulertigs-fa-out eulertigs.fa
```

Use the `--help` option to get an overview of available options.
```bash
matchtigs --help
```

## Citation

**matchtigs preprint**

Schmidt, S., Khan, S., Alanko, J., & Tomescu, A. I. (2021). Matchtigs: minimum plain text representation of kmer sets. _bioRxiv_. [https://doi.org/10.1101/2021.12.15.472871](https://doi.org/10.1101/2021.12.15.472871).

**Eulertigs preprint (accepted at WABI 2022)**

Schmidt, S. and Alanko, J. (2022). Eulertigs: minimum plain text representation of k-mer sets without repetitions in linear time. _bioRxiv_. [https://doi.org/10.1101/2022.05.17.492399](https://doi.org/10.1101/2022.05.17.492399).
