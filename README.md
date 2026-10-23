# ATLAS - Advanced Trajectory Learning from multi-omics At Single-cell resolution
This repository contains the code associated to the original paper from ATLAS- Advanced Trajectory Learning from multi-omics At Single-cell resolution.

ATLAS is Python package for multi-omic Trajectory Inference (TI) from paired single-cell RNA-seq and ATAC-seq data. Since chromatin accessibility reflects regulatory potential and often precedes transcriptional changes, integrating it with RNA expression provides a more complete view of cellular dynamics than transcriptomics alone. ATLAS extends established TI frameworks to jointly leverage multi-omics data, allowing chromatin accessibility to directly inform pseudotime ordering and fate probabilities.

ATLAS is scverse-compatible and currently under review for inclusion in the [scverse](https://scverse.org/) ecosystem. ATLAS source code can be found [here](https://github.com/smilies-polito/atlas-smilies).

![ATLAS workflow](imgs/workflow.svg)


## Release Notes 
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

1. Install `Apptainer` version from [this link](https://apptainer.org/). results in the manuscrpt were produced using Apptainer v1.4.0. 
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
| Fresh Embryonic E18 Mouse Brain other metadata | [MultiVelo labels](https://github.com/welch-lab/MultiVelo/blob/main/Examples/cell_annotations.tsv), [ArchVelo labels and UMAP](https://figshare.com/articles/dataset/summit_counts_matrix_csv_gz/27931914) | `atlas-experiments/data/embryonic_mouse_brain` | `cell_annotations.tsv`, `archvelo_labels.csv`, `multivelo_umap.csv` |
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
| Fresh Embryonic E18 Mouse Brain | `atlas-experiments/output/embryonic_mouse_brain` | Files, models, and intermediate results generated from analyses on the Fresh Embryonic E18 Mouse Brain dataset |
| SHARE-seq Mouse Hair Follicle | `atlas-experiments/output/mouse_hair` | Files, models, and intermediate results generated from analyses on the SHARE-seq Mouse Hair Follicle dataset |
| Human Fetal Brain | `atlas-experiments/output/human_brain` | Files, models, and intermediate results generated from analyses on the Human Fetal Brain dataset |
| Synthetic Data | `atlas-experiments/output/simulations` | Files, models, and intermediate results generated from synthetic simulations |
| Benchmark memory and time | `atlas-experiments/output/benchmark` | Files, models, and intermediate results generated from time and memory benchmark |

Results and intermediate files generated during synthetic simulations are further organized into experiment-specific subfolders such as `atlas-experiments/output/simulations/palantir`, `atlas-experiments/output/simulations/palantir_rna`, etc.

Figures are saved in `atlas-experiments/figures` as performed in [scanpy](https://scanpy.readthedocs.io/) and [muon](https://muon.scverse.org/). 

### Run Experiments

The full analysis is implemented as a Snakemake workflow that runs inside an `apptainer` container. The pipeline processes the three real datasets and the grid of synthetic simulations. 
Snakemake is available in the container. 
The pipeline has been tested on SLURM; other schedulers require adapting the profile. Furthermore the actual Snakefile is cofigured with a specific threading configuration, which might be adapted according to the specific settings in which simulations are performed. 

Before running populate the `data/` subfolders with the input files as described in the previous sections and perform a dry run to confirm the DAG resolves correctly. The dry run should enlist a total of X rules. 
```bash
apptainer exec container/container.sif snakemake -n
```

To list all target files that would be produces by `rule all`, run:
```bash
apptainer exec container/container.sif snakemake --list-target-rules
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

It is not mandatory to run the entire workflow. Snakemake can target individual output files or rules and it will automatically execute all the upstream rules rerquired to produce them. Here we provide an overview of the available rules:
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
    humanBrain_preprocessing e18brain_preprocessing mouseHair_preprocessing
```

```bash
# Run Figure 1
mkdir -p logs
apptainer exec container/container.sif snakemake --cores 4 \
    fig1_mouseHair_atlas lineages
```

Most runs are based on wildcards (similar to hyperparameters) combinations. To run a rule with a specific set of wildcards it is sufficient to target the specific output file. For example:
```bash
# Run ATLAS experiment on the SHARE-seq Mouse Hair dataser on a specific configuration
mkdir -p logs
apptainer exec container/container.sif snakemake --cores 2 \
    output/mouse_hair/hyper_done/rna30_act30_wnn30_states8.done
```

```bash
# Run ATLAS (Palantir-based TI) on a specific synthetic dataset configuration
mkdir -p logs
apptainer exec container/container.sif snakemake --cores 2 \
    output/simulations/palantir/three_branches_0.5:0.5_20:10_30:30:30.done
```

![rule dependency](imgs/graphviz.svg)

## Repository Structure

The complete repository structure can be found here: 
```
atlas-experiments/
├── container/ 
│   ├── requirements.txt               # repository requirements 
│   └── container.def                  # singularity container definition file
├── data/
│   ├── embryonic_mouse_brain/         # Fresh Embryonic E18 Mouse Brain data
│   │   └── cell_annotations.tsv 
│   ├── human_brain/                   # Human Fetal Brain Data
│   │   └── to_remove.tsv
│   ├── mouse_hair/                    # SHARE-seq Mouse Hair Follicle data
│   ├── simulated_data/   
│   │   ├── five_branches/             # Tsv files related to sinthetic data 5-branches developmental tree
│   │   └── three_branches/            # Tsv files related to sinthetic data 3-branches developmental tree
├── figures/
├── imgs/ 
│   └── workflow.svg      
├── output/               
│   ├── embryonic_mouse_brain/         # Results and support data for Fresh Embryonic E18 Mouse Brain
│   ├── human_brain/                   # Results and support data for Human Fetal Brain
│   ├── mouse_hair/                    # Results and support data for SHARE-seq Mouse Hair
│   ├── simulations/                   # Results and support data for Synthetic Data
│   │   ├── palantir/                  # Results and support data for ATLAS (Palantir-based TI) on synthetic data
│   │   ├── palantir_rna/              # Results and support data for Palantir on synthetic data (scRNA-seq only)
│   │   ├── pseudotime_kernel/         # Results and support data for ATLAS (CellRank-based TI) on synthetic data
│   │   └── pseudotime_kernel_rna/     # Results and support data for CellRank on synthetic data (scRNA-seq only)
├── real_data/                         # Folder with code for real dataset experiments
│   │   ├── mouse_hair/    	       # Results for the SHARE-seq Mouse Hair Follicle 
│   │  	│   ├── preprocessing.py
│   │   │   ├── figure1_hf_atlas_cellrank.py 
│   │   │   ├── figure1_hf_atlas_palantir.py 
│   │   │   ├── supplementary5_mh_rna_cellrank.py 
│   │   │   ├── supplementary5_mh_rna_palantir.py 
│   │   │   └── figure1_hf_lineages.py
│   │   ├── human_brain/    	       # Results for the Human Fetal Brain 
│   │  	│   ├── get_features.py
│   │  	│   ├── preprocessing.py
│   │   │   ├── supplementary1012_hb_atlas_palantir.py
│   │   │   ├── supplementary1012_hb_atlas_cellrank.py
│   │   │   ├── supplementary1113_hb_rna_palantir.py
│   │   │   ├── supplementary1113_hb_rna_cellrank.py
│   │   │   └── supplementary9_hb_gex.py
│   │   ├── embryonic_mouse_brain/    	       # Results for the Fresh Embryonic E18 Mouse Brain
│   │  	│   ├── preprocessing.py
│   │   │   ├── supplementary6_e18_atlas_palantir.py
│   │   │   ├── supplementary6_e18_atlas_cellrank.py
│   │   │   ├── supplementary7_e18_rna_palantir.py
│   │   │   ├── supplementary7_e18_rna_cellrank.py
│   │   │   └── supplementary8_e18_gex.py
│   └── utils.py      
├── simulated_data/                    # Folder with code for synthetic data experiments
│   ├── preprocessing.py               # Preprocessing using ATLAS and scanpy
│   ├── palantir_rna_run.py            # scRNA-seq only experiment on synthetic data using Palantir
│   ├── palantir_run.py                # ATLAS on synthetic casa (Palantir-based TI)
│   ├── pseudotime_rna_run.py          # scRNA-seq only experiment on synthetic data using CellRank
│   ├── pseudotime_run.py              # ATLAS on synthetic data (CellRank-based TI)
│   ├── supervised_metrics.py          # Code for supervised evaluation
│   └── utils.py
├── supplementary/                     # Folder with code for additional visualisations 
│   ├── supplementary34_box_radar_plots.py 
│   └── supplementary34_tables.py
├── .gitignore  
├── README.md  
└── SnakeFile
```

## Information about R Environment
Synthetic data were generated using scMultiSim v1.0.0 and R v.4.4.1. Code for synthetic data generation can be found at `atlas-experiments/simulated_data/simulate.R`. 
Set the working environment by inserting the path to `atlas-experiments/data/simulated_data/<differentiation_tree>` in `setwd(...)` at line 74.
Set the `tree` parameter either to `Phyla3()` or `Phyla5()` in the `sim_true_counts` function at line 93. 
