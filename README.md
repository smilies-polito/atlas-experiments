# ATLAS - Advanced Trajectory Learning from multi-omics At Single-cell resolution
This repository contains the code associated to the original paper from ATLAS- Advanced Trajectory Learning from multi-omics At Single-cell resolution.

ATLAS is Python package for multi-omic Trajectory Inference (TI) from paired single-cell RNA-seq and ATAC-seq data. Since chromatin accessibility reflects regulatory potential and often precedes transcriptional changes, integrating it with RNA expression provides a more complete view of cellular dynamics than transcriptomics alone. ATLAS extends established TI frameworks to jointly leverage multi-omics data, allowing chromatin accessibility to directly inform pseudotime ordering and fate probabilities.

ATLAS included in the [scverse](https://scverse.org/) ecosystem. ATLAS source code can be found [here](https://github.com/smilies-polito/atlas-smilies).

![ATLAS workflow](imgs/fig1.svg)


## Release Notes 
v1.0.0:
- Release after revision round 1. 

v0.1.0:
- First release

## How To Cite:
### ATLAS Primary Publication 
- Leclercq A, Martini L, Bardini R, Savino A, Di Carlo S. ATLAS: a scverse-compatible package for multi-omic single-cell trajectory inference integration. bioRxiv (2026). doi:10.64898/2026.05.23.727175

### Related Works
- Lange, M., Bergen, V., Klein, M. et al. CellRank for directed single-cell fate mapping. Nat Methods 19, 159–170 (2022). https://doi.org/10.1038/s41592-021-01346-6
- Weiler, P., Lange, M., Klein, M. et al. CellRank 2: unified fate mapping in multiview single-cell data. Nat Methods 21, 1196–1205 (2024). https://doi.org/10.1038/s41592-024-02303-9
- Setty, M., Kiseliovas, V., Levine, J. et al. Characterization of cell fate probabilities in single-cell data with Palantir. Nat Biotechnol 37, 451–460 (2019). https://doi.org/10.1038/s41587-019-0068- Li, H., Zhang, Z., Squires, M. et al. scMultiSim: simulation of single-cell multi-omics and spatial data guided by gene regulatory networks and cell–cell interactions. Nat Methods 22, 982–993 (2025). https://doi.org/10.1038/s41592-025-02651-0

### Datasets
- Fresh Embryonic E18 Mouse Brain (5k), Single Cell Multiome ATAC + Gene Expression Dataset by Cell Ranger ARC 2.0.0, 10x Genomics, (2021, May 3).
- Sai Ma et al. Chromatin Potential Identified by Shared Single-Cell Profiling of RNA and Chromatin. Cell, 183(4):1103–1116.e20, November 2020.
- Alexandro E. Trevino et al. Chromatin and gene-regulatory dynamics of the developing human cerebral cortex at single-cell resolution. Cell, 184(19):5053–5069.e23, September 2021.


### Experimental Setup

Follow these steps to setup for reproducing the experiments. 

1. Install `Apptainer` version from [this link](https://apptainer.org/). Results in the manuscrpt were produced using Apptainer v1.4.0. 
2. Install `htslib` version 1.21 from [this link](https://www.htslib.org/download/)
3. Clone the repository in your home folder via
```bash
git clone -b public https://github.com/smilies-polito/atlas-experiments.git
cd atlas-experiments
```
4. Move to the repository folder and build the `singularity` container with:
```bash
sudo apptainer build container/container.sif container/container.def 
```
or 
```bash
apptainer build --fakeroot container/container.sif container/container.def 
```

## Reproduce Analyses
### Required Data and Input Folders
Input data must be downloaded and places in the `atlas-experiments/data` folder under the correct dataset-related subfolder. Here we provide the urls for the data, the related subfolder and the minimum files to download to endure correct reproducibility, along with bash scripts:

| Dataset | URLs | Subfolder | Mandatory Files and Compatible Naming |
|---|---|---|---|
| Fresh Embryonic E18 Mouse Brain | [10x Genomics](https://www.10xgenomics.com/datasets/fresh-embryonic-e-18-mouse-brain-5-k-1-standard-2-0-0) | `atlas-experiments/data/embryonic_mouse_brain` | `filtered_feature_bc_matrix/`, `e18_mouse_brain_fresh_5k_atac_fragments.tsv.gz`, `e18_mouse_brain_fresh_5k_atac_fragments.tsv.gz.tbi` |
| SHARE-seq Mouse Hair Follicle | [scATAC-seq](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4156597), [scRNA-seq](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4156608) | `atlas-experiments/data/mouse_hair` | `GSM4156608_skin.late.anagen.rna.counts.txt`, `GSM4156597_skin.late.anagen.counts.txt`, `GSM4156597_skin.late.anagen.barcodes.txt`, `GSM4156597_skin.late.anagen.peaks.bed`, `GSM4156597_skin_celltype.txt`, `GSM4156597_skin.late.anagen.atac.fragments.sorted.bed.gz`, `GSM4156597_skin.late.anagen.atac.fragments.sorted.bed.gz.tbi` |
| Human Fetal Brain | [GEO: GSE162170](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE162170) | `atlas-experiments/data/human_brain` | `GSE162170_multiome_cluster_names.txt`, `GSE162170_multiome_cell_metadata.txt`, `GSE162170_multiome_rna_counts.tsv.gz`, `GSE162170_multiome_atac_gene_activities.tsv.gz` |
| Synthetic data generated via scMultiSim | [Link](https://zenodo.org/records/20611526) | `atlas-experiments/data/simulated_data` | |

Fresh Embryonic E18 Mouse Brain:
```bash
cd atlas-experiments/data/embryonic_mouse_brain
curl -O https://cf.10xgenomics.com/samples/cell-arc/2.0.0/e18_mouse_brain_fresh_5k/e18_mouse_brain_fresh_5k_filtered_feature_bc_matrix.tar.gz
tar -xvf e18_mouse_brain_fresh_5k_filtered_feature_bc_matrix.tar.gz
curl -O https://cf.10xgenomics.com/samples/cell-arc/2.0.0/e18_mouse_brain_fresh_5k/e18_mouse_brain_fresh_5k_atac_fragments.tsv.gz
curl -O https://cf.10xgenomics.com/samples/cell-arc/2.0.0/e18_mouse_brain_fresh_5k/e18_mouse_brain_fresh_5k_atac_fragments.tsv.gz.tbi
```

SHARE-seq Mouse Hair Follicle (requires `htslib` for `bgzip` and `tabix`):
```bash
cd atlas-experiments/data/mouse_hair
curl -O https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4156nnn/GSM4156597/suppl/GSM4156597_skin_celltype.txt.gz
curl -O https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4156nnn/GSM4156597/suppl/GSM4156597_skin.late.anagen.counts.txt.gz
curl -O https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4156nnn/GSM4156597/suppl/GSM4156597_skin.late.anagen.peaks.bed.gz
curl -O https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4156nnn/GSM4156608/suppl/GSM4156608_skin.late.anagen.rna.counts.txt.gz
curl -O https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4156nnn/GSM4156597/suppl/GSM4156597_skin.late.anagen.barcodes.txt.gz
curl -O https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4156nnn/GSM4156597/suppl/GSM4156597_skin.late.anagen.atac.fragments.bed.gz

gunzip GSM4156597_skin_celltype.txt.gz
gunzip GSM4156597_skin.late.anagen.barcodes.txt.gz
gunzip GSM4156597_skin.late.anagen.counts.txt.gz
gunzip GSM4156608_skin.late.anagen.rna.counts.txt.gz
gunzip GSM4156597_skin.late.anagen.peaks.bed.gz
gunzip GSM4156597_skin.late.anagen.atac.fragments.bed.gz
sort -k1,1 -k2,2n GSM4156597_skin.late.anagen.atac.fragments.bed > GSM4156597_skin.late.anagen.atac.fragments.sorted.bed
bgzip GSM4156597_skin.late.anagen.atac.fragments.sorted.bed
tabix -p bed GSM4156597_skin.late.anagen.atac.fragments.sorted.bed.gz
```

The gene coordinates used to compute gene activities for the SHARE-seq Mouse Hair Follicle data are provided in `atlas-experiments/data/mouse_hair/all_features.tsv` and do not need to be downloaded. They were extracted from the `gene` entries of the Ensembl release 102 annotation for mouse (GRCm38, [Mus_musculus.GRCm38.102.gtf.gz](http://ftp.ensembl.org/pub/release-102/gtf/mus_musculus/Mus_musculus.GRCm38.102.gtf.gz)), keeping gene name, chromosome, start, end and strand.

Human Fetal Brain:
```bash
cd atlas-experiments/data/human_brain
curl -O https://ftp.ncbi.nlm.nih.gov/geo/series/GSE162nnn/GSE162170/suppl/GSE162170_multiome_atac_gene_activities.tsv.gz
curl -O https://ftp.ncbi.nlm.nih.gov/geo/series/GSE162nnn/GSE162170/suppl/GSE162170_multiome_cell_metadata.txt.gz
curl -O https://ftp.ncbi.nlm.nih.gov/geo/series/GSE162nnn/GSE162170/suppl/GSE162170_multiome_rna_counts.tsv.gz
curl -O https://ftp.ncbi.nlm.nih.gov/geo/series/GSE162nnn/GSE162170/suppl/GSE162170_multiome_cluster_names.txt.gz

gunzip GSE162170_multiome_cluster_names.txt.gz
gunzip GSE162170_multiome_cell_metadata.txt.gz
```

Synthetic Data:
```bash
cd atlas-experiments/data
curl -O https://zenodo.org/records/20611526/files/simulated_data.tar.gz
tar -xvf simulated_data.tar.gz
```

### Output Folders
All results generated by this repository are stored in the `atlas-experiments/output` directory.

| Dataset | Output Subfolder | Folder Content |
|---|---|---|
| Fresh Embryonic E18 Mouse Brain | `atlas-experiments/output/embryonic_mouse_brain` | Preprocessed and analysed `.h5mu` files, pseudotime `.tsv` files, metrics in `results.tsv` and the gene coordinates used for gene activities (`features.tsv`) |
| SHARE-seq Mouse Hair Follicle | `atlas-experiments/output/mouse_hair` | Preprocessed and analysed `.h5mu` files, pseudotime `.tsv` files, metrics in `results.tsv` and the gene coordinates used for gene activities (`features.tsv`) |
| Human Fetal Brain | `atlas-experiments/output/human_brain` | Preprocessed and analysed `.h5mu` files, pseudotime `.tsv` files and metrics in `results.tsv` |
| Synthetic Data | `atlas-experiments/output/simulations` | Preprocessed synthetic datasets, metrics of each experiment, summary tables and figures (see below) |
| Benchmark memory and time | `atlas-experiments/output/benchmark` | Input datasets and results of the time and memory benchmark, in `results_pex/` (Palantir-based strategy) and `results_crx/` (CellRank-based strategy) |

Results of the synthetic simulations are organized into experiment-specific subfolders:

| Subfolder | Folder Content |
|---|---|
| `atlas-experiments/output/simulations/mudata` | Preprocessed synthetic datasets (`.h5mu`), one per parameter combination |
| `atlas-experiments/output/simulations/palantir` | Metrics of ATLAS (Palantir-based TI) in `results.csv`, box plots and radar plot |
| `atlas-experiments/output/simulations/pseudotime_kernel` | Metrics of ATLAS (CellRank-based TI) in `results.csv`, box plots and radar plot |
| `atlas-experiments/output/simulations/palantir_rna` | Metrics of Palantir (scRNA-seq only) in `results.csv` |
| `atlas-experiments/output/simulations/pseudotime_kernel_rna` | Metrics of CellRank (scRNA-seq only) in `results.csv` |

The performance and robustness tables comparing ATLAS with the scRNA-seq only baselines (`performance_<algorithm>.tsv`, `robustness_<algorithm>.tsv`) are saved directly in `atlas-experiments/output/simulations`.

All other figures are saved in `atlas-experiments/figures`, following the convention of [scanpy](https://scanpy.readthedocs.io/) and [muon](https://muon.scverse.org/). 

### Run Experiments

The full analysis is implemented as a Snakemake workflow that runs inside an `apptainer` container. The pipeline processes the three real datasets and the grid of synthetic simulations. 
Snakemake is available in the container. 
The pipeline has been tested on SLURM; other schedulers require adapting the profile. Furthermore the actual Snakefile is cofigured with a specific threading configuration, which might be adapted according to the specific settings in which simulations are performed. 

Before running populate the `data/` subfolders with the input files as described in the previous sections and perform a dry run to confirm the DAG resolves correctly. The dry run should list a total of 10,047 jobs. 
```bash
apptainer exec container/container.sif snakemake -n
```

To locally run all jobs on as single node using all available cores, run:
```bash
mkdir -p logs
apptainer exec container/container.sif snakemake --cores all
```

To run experiments on a SLURM cluster (recommended), wrap the call into an `sbatch` script. This work on any slurm cluster, as long as the `#SBATCH` directives are adapted to the site's partition.
```bash
# submit.sh
#!/bin/bash
#SBATCH --job-name=ATLAS
#SBATCH --partition=
#SBATCH --cpu-per-task=
#SBATCH --mem
#others

mkdir -p logs
apptainer exec container/container.sif snakemake --cores 32
```

Then: 
```bash
sbatch submit.sh
```

It is not mandatory to run the entire workflow. Snakemake can target individual output files and it will automatically execute all the upstream rules required to produce them. Since all rules depend on wildcards, they cannot be called by name and must be run by targeting one of their output files. Here we provide an overview of the available rules:
| Rule | Dataset | Outcome |
| --- | --- | --- |
| e18brain\_preprocessing | Fresh Embryonic E18 Mouse Brain | Preprocesses the E18 Mouse Brain data and outputs `.h5mu` file |
| e18brain\_atlas\_palantir | Fresh Embryonic E18 Mouse Brain | Applies ATLAS (Palantir-based stategy) on the E18 Mouse Brain and produces results from Supp. Fig. 6 |
| e18brain\_atlas\_cellrank | Fresh Embryonic E18 Mouse Brain | Applies ATLAS (CellRank-based stategy) on the E18 Mouse Brain and produces results from Supp. Fig. 6 |
| e18brain\_rna\_palantir | Fresh Embryonic E18 Mouse Brain | Applies originsl Palantir (scRNA-seq only based TI) on the E18 Mouse Brain and produces results from Supp. Fig. 7 |
| e18brain\_rna\_cellrank | Fresh Embryonic E18 Mouse Brain | Applies originsl CellRank (scRNA-seq only based TI) on the E18 Mouse Brain and produces results from Supp. Fig. 7 |
| e18brain\_expression | Fresh Embryonic E18 Mouse Brain | Gene Expression for target genes on the E18 Mouse Brain and produces results from Supp. Fig. 8 |
| preprocess\_synthetic | Synthetic Data | ATLAS preprocessing on all combination of Synthetic Data |
| synthetic\_palantir | Synthetic Data | ATLAS (Palantir-based TI) on all combination of Synthetic Data |
| synthetic\_pseudotime\_kernel | Synthetic Data | ATLAS (CellRank-based TI) on all combination of Synthetic Data |
| synthetic\_palantir\_rna | Synthetic Data | scRNA-seq only experiments using Palantir |
| synthetic\_pseudotime\_kernel\_rna | Synthetic Data | scRNA-seq only experiments using CellRank |
| mouseHair\_preprocessing | SHARE-seq Mouse Hair | Preprocesses the SHARE-seq Mouse Hair data and outputs `.h5mu` file |
| mouseHair\_rna\_cellrank | SHARE-seq Mouse Hair | Applies scRNA-seq only TI (CellRank) on the SHARE-seq Mouse Hair and and produces results from Supp. Fig. 5 |
| mouseHair\_rna\_palantir | SHARE-seq Mouse Hair | Applies scRNA-seq only TI (Palantir) on the SHARE-seq Mouse Hair and and produces results from Supp. Fig. 5 |
| mouseHair\_atlas\_cellrank | SHARE-seq Mouse Hair | Applies ATLAS TI (CellRank-based strategy) on the SHARE-seq Mouse Hair and and produces results from Fig.1 and Supp. Fig. 5 |
| mouseHair\_atlas\_palantir | SHARE-seq Mouse Hair | Applies ATLAS TI (Palantir-based strategy) on the SHARE-seq Mouse Hair and and produces results from Fig.1 and Supp. Fig. 5 |
| humanBrain\_preprocessing |  Human Fetal Brain | Preprocesses the Human Fetal Brain data and outputs `.h5mu` file |
| humanBrain\_expression |  Human Fetal Brain | Gene Expression for target genes on the Human Fetal Brain and produces results from Supp. Fig. 9 | 
| humanBrain\_rna\_cellrank | Human Fetal Brain | Applies scRNA-seq only TI (CellRank) on the Human Fetal Brain and and produces results from Supp. Fig. 11 and 13 |
| humanBrain\_rna\_palantir | Human Fetal Brain | Applies scRNA-seq only TI (Palantir) on the Human Fetal Brain and and produces results from Supp. Fig. 11 and 13 |
| humanBrain\_atlas\_cellrank | Human Fetal Brain | Applies ATLAS TI (CellRank-based strategy) on the Human Fetal Brain and and produces results from Supp. Fig. 10 and 12 |
| humanBrain\_atlas\_palantir | Human Fetal Brain | Applies ATLAS TI (Palantir-based strategy) on the Human Fetal Brain and and produces results from Supp. Fig. 10 and 12 |

Example:
```bash
# Run preprocessing on all three real datasets
mkdir -p logs
apptainer exec container/container.sif snakemake --cores 4 \
    output/embryonic_mouse_brain/20:10_15:15:None.h5mu \
    output/mouse_hair/20:10_15:15:None.h5mu \
    output/human_brain/non_removed_20:10_15:15:None.h5mu \
    output/human_brain/removed_20:10_15:15:None.h5mu
```

```bash
# Run Figure 1: ATLAS (Palantir-based and CellRank-based strategies) on the SHARE-seq Mouse Hair Follicle
mkdir -p logs
apptainer exec container/container.sif snakemake --cores 4 \
    output/mouse_hair/palantir_20:10_15:15:None.h5mu \
    output/mouse_hair/cellrank_20:10_15:15:None_8.h5mu
# Gene regulation trends along the lineages (Figure 1) are not part of the workflow and are produced by
apptainer exec container/container.sif python3 -m real_data.mouse_hair.figure1_hf_lineages \
    --h5mu output/mouse_hair/cellrank_20:10_15:15:None_8.h5mu
```

Most runs are based on wildcards (similar to hyperparameters) combinations. To run a rule with a specific set of wildcards it is sufficient to target the specific output file. For example:
```bash
# Run ATLAS (CellRank-based TI) on the SHARE-seq Mouse Hair Follicle with 30 RNA neighbors, 30 activity neighbors, 30 WNN neighbors and 8 macrostates
mkdir -p logs
apptainer exec container/container.sif snakemake --cores 2 \
    output/mouse_hair/cellrank_20:10_30:30:30_8.h5mu
```

```bash
# Run ATLAS (Palantir-based TI) on a specific synthetic dataset configuration
mkdir -p logs
apptainer exec container/container.sif snakemake --cores 2 \
    output/simulations/palantir/three_branches_0.5:0.5_20:10_30:30:30.done
```

![rule dependency](imgs/graphviz.svg)

### Supplementary Analyses

The following analyses are not part of the Snakemake workflow. They are run from the repository root inside the container, once the corresponding results are available.

**Synthetic data: box plots, radar plots and tables.** These scripts summarize the results of the synthetic simulations stored in `output/simulations/<algorithm>/results.csv` and `output/simulations/<algorithm>_rna/results.csv`, where `<algorithm>` is either `palantir` (ATLAS Palantir-based strategy) or `pseudotime_kernel` (ATLAS CellRank-based strategy). The results files of the paper are provided in the repository, so the scripts can be run without re-running the simulations.
```bash
for algorithm in palantir pseudotime_kernel; do
    # box plots and radar plot, saved in output/simulations/<algorithm>/
    apptainer exec container/container.sif python3 -m supplementary.supplementary34_box_radar_plots --algorithm $algorithm
    # performance and robustness tables comparing ATLAS with the scRNA-seq only baseline, saved in output/simulations/
    apptainer exec container/container.sif python3 -m supplementary.supplementary34_tables --algorithm $algorithm
done
```

> [!NOTE]
> The radar plot is exported with Plotly through `kaleido`, which requires a Chrome or Chromium browser. The container does not include one, so inside the container the radar plot may fail while the box plots and the tables are produced normally. In that case, run `supplementary34_box_radar_plots.py` in a Python environment with the packages of `container/requirements.txt` and a Chrome browser installed.

**Number of macrostates.** This script plots the metrics of ATLAS and of CellRank (scRNA-seq only) against the number of macrostates, reading `output/<dataset>/results.tsv`. It requires the CellRank runs of the SHARE-seq Mouse Hair Follicle produced by the workflow for all numbers of macrostates, and saves the figure in `figures/metrics_vs_macrostates.png` and `figures/metrics_vs_macrostates.pdf`.
```bash
apptainer exec container/container.sif python3 -m supplementary.supplementary_n_macrostates --folder mouse_hair
```

> [!NOTE]
> `supplementary_n_macrostates.py` supports only the SHARE-seq Mouse Hair Follicle (`--folder mouse_hair`). 

**Time and memory benchmark.** The benchmark of the Palantir-based and CellRank-based strategies is run with its own launchers, and its results are provided in `output/benchmark/`. Instructions to reproduce it are in [`supplementary/benchmark/README.md`](supplementary/benchmark/README.md).

## Repository Structure

The complete repository structure can be found here: 
```
atlas-experiments/
├── container/
│   ├── container.def                  # Apptainer/Singularity container definition file
│   ├── pyvitae.def                    # Container definition file for the VITAE comparison
│   └── requirements.txt               # Python requirements installed in container.def
├── data/
│   ├── embryonic_mouse_brain/         # Fresh Embryonic E18 Mouse Brain data
│   │   ├── archvelo_labels.csv        # ArchVelo cell type labels
│   │   ├── cell_annotations.tsv       # MultiVelo cell type labels
│   │   └── multivelo_umap.csv         # MultiVelo UMAP embedding
│   ├── human_brain/                   # Human Fetal Brain data
│   │   ├── gene_annotations.tsv       # Ensembl gene IDs to gene names
│   │   └── to_remove.tsv              # Outlier cells removed in the "removed" setting
│   ├── mouse_hair/                    # SHARE-seq Mouse Hair Follicle data
│   │   └── all_features.tsv           # Gene coordinates (Ensembl release 102, GRCm38)
│   └── simulated_data/
│       ├── five_branches/             # Tsv files related to synthetic data 5-branches developmental tree
│       └── three_branches/            # Tsv files related to synthetic data 3-branches developmental tree
├── figures/
├── imgs/
│   ├── graphviz.svg                   # Snakemake rule graph
│   └── workflow.svg                   # ATLAS workflow
├── output/
│   ├── benchmark/                     # Results of the time and memory benchmark
│   │   ├── results_crx/               # CellRank-based strategy
│   │   │   ├── failures.tsv
│   │   │   ├── resources_crx.tsv
│   │   │   ├── scaling_resources_crx.tsv
│   │   │   └── summary_resources_crx.tsv
│   │   └── results_pex/               # Palantir-based strategy
│   │       ├── resources_pex.tsv
│   │       ├── scaling_resources_pex.tsv
│   │       └── summary_resources_pex.tsv
│   ├── embryonic_mouse_brain/         # Results and support data for Fresh Embryonic E18 Mouse Brain
│   ├── human_brain/                   # Results and support data for Human Fetal Brain
│   ├── mouse_hair/                    # Results and support data for SHARE-seq Mouse Hair
│   └── simulations/                   # Results and support data for Synthetic Data
│       ├── mudata/                    # Preprocessed synthetic datasets
│       ├── palantir/                  # Results and support data for ATLAS (Palantir-based TI) on synthetic data
│       │   └── results.csv
│       ├── palantir_rna/              # Results and support data for Palantir on synthetic data (scRNA-seq only)
│       │   └── results.csv
│       ├── pseudotime_kernel/         # Results and support data for ATLAS (CellRank-based TI) on synthetic data
│       │   └── results.csv
│       └── pseudotime_kernel_rna/     # Results and support data for CellRank on synthetic data (scRNA-seq only)
│           └── results.csv
├── real_data/                         # Folder with code for real dataset experiments
│   ├── __init__.py
│   ├── embryonic_mouse_brain/         # Code for the Fresh Embryonic E18 Mouse Brain
│   │   ├── preprocessing.py
│   │   ├── supplementary6_e18_atlas_cellrank.py
│   │   ├── supplementary6_e18_atlas_palantir.py
│   │   ├── supplementary7_e18_rna_cellrank.py
│   │   ├── supplementary7_e18_rna_palantir.py
│   │   └── supplementary8_e18_gex.py
│   ├── human_brain/                   # Code for the Human Fetal Brain
│   │   ├── get_features.py
│   │   ├── preprocessing.py
│   │   ├── supplementary1012_hb_atlas_cellrank.py
│   │   ├── supplementary1012_hb_atlas_palantir.py
│   │   ├── supplementary1113_hb_rna_cellrank.py
│   │   ├── supplementary1113_hb_rna_palantir.py
│   │   └── supplementary9_hb_gex.py
│   ├── mouse_hair/                    # Code for the SHARE-seq Mouse Hair Follicle
│   │   ├── __init__.py
│   │   ├── figure1_hf_atlas_cellrank.py
│   │   ├── figure1_hf_atlas_palantir.py
│   │   ├── figure1_hf_lineages.py
│   │   ├── preprocessing.py
│   │   ├── supplementary5_mh_rna_cellrank.py
│   │   └── supplementary5_mh_rna_palantir.py
│   └── utils.py
├── simulated_data/                    # Folder with code for synthetic data experiments
│   ├── __init__.py
│   ├── palantir_rna_run.py            # scRNA-seq only experiment on synthetic data using Palantir
│   ├── palantir_run.py                # ATLAS on synthetic data (Palantir-based TI)
│   ├── preprocessing.py               # Preprocessing using ATLAS and scanpy
│   ├── pseudotime_kernel_rna_run.py   # scRNA-seq only experiment on synthetic data using CellRank
│   ├── pseudotime_kernel_run.py       # ATLAS on synthetic data (CellRank-based TI)
│   ├── simulate_data.R                # Generation of the synthetic data with scMultiSim
│   ├── supervised_metrics.py          # Code for supervised evaluation
│   └── utils.py
├── supplementary/                     # Folder with code for additional analyses and visualisations
│   ├── benchmark/                     # Time and memory benchmark
│   │   ├── README.md
│   │   ├── analyze_benchmark.py
│   │   ├── benchmark_crx.py
│   │   ├── benchmark_pex.py
│   │   ├── memory_monitor.py
│   │   ├── prepare_crx_pseudotime.py
│   │   ├── run_benchmark_crx.sh
│   │   ├── run_benchmark_pex.sh
│   │   ├── subset_dataset.py
│   │   └── utils.py
│   ├── vitae/                         # Comparison with VITAE
│   │   ├── run_vitae.py
│   │   ├── save_matrices.py
│   │   └── vitae_figures.py
│   ├── supplementary34_box_radar_plots.py
│   ├── supplementary34_tables.py
│   └── supplementary_n_macrostates.py
├── .gitignore
├── README.md
└── Snakefile
```

## Information about R Environment
Synthetic data were generated using scMultiSim v1.0.0 and R v.4.4.1. Code for synthetic data generation can be found at `atlas-experiments/simulated_data/simulate_data.R`. 
Set the working environment by inserting the path to `atlas-experiments/data/simulated_data/<differentiation_tree>` in `setwd(...)` at line 74.
Set the `tree` parameter either to `Phyla3()` or `Phyla5()` in the `sim_true_counts` function at line 93. 
