# Matchtigs & Eulertigs: minimum plain text representation of kmer sets - with and without repetitions

This is an implementation of different algorithms for computing small and minimum plain text representations of kmer sets.
The algorithms expect unitigs as an input, which can e.g. be computed with [BCALM2](https://github.com/GATB/bcalm).

## Features

 * Compute [matchtigs and greedy matchtigs](https://doi.org/10.1101/2021.12.15.472871) with multiple threads
 * Compute Eulertigs
 * Compute pathtigs (heuristical Eulertigs similar to [ProphAsm](https://doi.org/10.1186/s13059-021-02297-z))
 * Both fasta and GFA format supported, as well as gzip compression if the files end in `.gz`
 * The annotations in a fasta file output by bcalm2 can be used to speed up loading (use `--bcalm-in` instead of `--fasta-in`)
 * Output (ASCII-) bitvectors of duplicate kmers for applications that require unique kmers

## Installation

The matchtigs tool can be installed with the following methods.
Out of the box, it implements all algorithms but the optimal matchtig algorithm.
This is because the optimal matchtig algorithm uses the commercial software blossom V, which is free to use for researchers, but cannot be publicly redistributed.
For all practical purposes, we recommend using greedy matchtigs, as they are a lot more efficient to compute, and have very similar cumulative length and string count as matchtigs (see the matchtigs publication linked below).

### Installation via [conda/mamba](https://docs.conda.io/en/latest/)

Install `matchtigs` with
```bash
mamba install -c conda-forge -c bioconda matchtigs
```

### Installation via [cargo](https://crates.io/)

#### Requirements

Rust `>= 1.60.0`, best installed via [rustup](https://rustup.rs/).

#### Installation

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

**Eulertigs (WABI 2022 best paper award)**

Schmidt, S. and Alanko, J. (2022). Eulertigs: minimum plain text representation of k-mer sets without repetitions in linear time. WABI 2022. [10.4230/LIPIcs.WABI.2022.2](https://doi.org/10.4230/LIPIcs.WABI.2022.2).
