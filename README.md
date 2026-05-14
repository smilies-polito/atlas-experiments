# ATLAS - Advanced Trajectory Learning from multi-omics At Single-cell resolution
This repository contains the code associated to the original paper from ATLAS- Advanced Trajectory Learning from multi-omics At Single-cell resolution.

## Release Notes 
v1.0:
- First release

## How To Cite:
### ATLAS Primary Publication 
- TO DO

### Related Works
- Lange, M., Bergen, V., Klein, M. et al. CellRank for directed single-cell fate mapping. Nat Methods 19, 159–170 (2022). https://doi.org/10.1038/s41592-021-01346-6
- Weiler, P., Lange, M., Klein, M. et al. CellRank 2: unified fate mapping in multiview single-cell data. Nat Methods 21, 1196–1205 (2024). https://doi.org/10.1038/s41592-024-02303-9
- Setty, M., Kiseliovas, V., Levine, J. et al. Characterization of cell fate probabilities in single-cell data with Palantir. Nat Biotechnol 37, 451–460 (2019). https://doi.org/10.1038/s41587-019-0068- Li, H., Zhang, Z., Squires, M. et al. scMultiSim: simulation of single-cell multi-omics and spatial data guided by gene regulatory networks and cell–cell interactions. Nat Methods 22, 982–993 (2025). https://doi.org/10.1038/s41592-025-02651-0

### DataSets
- Fresh Embryonic E18 Mouse Brain (5k), Single Cell Multiome ATAC + Gene Expression Dataset by Cell Ranger ARC 2.0.0, 10x Genomics, (2021, May 3).
-  Sai Ma et al. Chromatin Potential Identified by Shared Single-Cell Profiling of RNA and Chromatin. Cell, 183(4):1103–1116.e20, November 2020.
- Alexandro E. Trevino et al. Chromatin and gene-regulatory dynamics of the developing human cerebral cortex at single-cell resolution. Cell, 184(19):5053–5069.e23, September 2021.

## Experimental Setup 
Follow these steps to setup for reproducing the experiments. 
1. Install \texttt{Apptainer} version 1.4.5
2. Clone the repository in your home folder: COMMAND LINE
3. Move to the repository folder and build the \texttt{singularity} container with:
```
cd scvemo/container
sudo singularity build container.sif container.def 
```
or 
```
cd scvemo/container
singularity build --fakeroot container.sif container.def 
```

## Reproduce Analyses
### Required Data
Data must be downloaded and placed in the correct folder 
- Fresh Embryonic E18 Mouse Brain data are available [here](https://www.10xgenomics.com/datasets/fresh-embryonic-e-18-mouse-brain-5-k-1-standard-2-0-0). Data must be stored in the \texttt{scvemo/data/embryonic\_mouse\_brain} folder. Mandatory data are \texttt{filtered\_feature\_bc\_matrix} folder and the \texttt{e18\_mouse\_brain\_fresh\_5k\_atac\_fragments.tsv.gz} files. Cell annotations are already provided in the repository. 
- SHARE-seq Mouse Hair Follicle data are available at: [scATAC-seq here](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4156597) and [scRNA-seq here](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4156608). Data must be stored in the \texttt{scvemo/data/mouse\_hair} folder. Mandatory data are the \texttt{GSM4156608\_skin.late.anagen.rna.counts.txt}, \texttt{GSM4156597\_skin.late.anagen.counts.txt}, \texttt{GSM4156597\_skin.late.anagen.barcodes.txt}, \texttt{GSM4156597\_skin.late.anagen.peaks.bed}, \texttt{GSM4156597\_skin\_celltype.txt} and \texttt{GSM4156597\_skin.late.anagen.atac.sorted.fragments.bed.gz} files. 
- Human Fetal Brain data are available [here](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE162170). Data must be stored in the \texttt{scvemo/data/human\_brain} folder. Mandatory data are the \texttt{GSE162170\_multiome\_cluster\_names.txt}, \texttt{GSE162170\_multiome\_cell\_metadata.txt}, \texttt{GSE162170\_multiome\_rna\_counts.tsv.gz}, \texttt{GSE162170\_multiome\_atac\_gene\_activities.tsv.gz} files. 
- Synthetic data generated via scMultiSim can be found [here]().

### Repro 






