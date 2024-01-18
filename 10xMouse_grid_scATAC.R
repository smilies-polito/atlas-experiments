library(Signac)
library(Seurat)
library(EnsDb.Mmusculus.v79)
library(data.table)
library(itertools)
library(ggplot2)
library(patchwork)
set.seed(52)

#if (!require("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")

#BiocManager::install("biovizBase")

setwd('C:/Users/aless/Desktop/scVEMO')
PATH = file.path(getwd(), 'filtered_feature_bc_matrix')
CELL.ANNOTATIONS = file.path(getwd(), 'cell_annotations.tsv')
BARCODES.METRICS = file.path(getwd(), 'e18_mouse_brain_fresh_5k_per_barcode_metrics.csv')
SAVING.FOLDER = file.path(file.path(getwd(), 'signac_data'), 'E18_embryonic_mouse_brain')

# Load data
counts <- Read10X(PATH)

chrom.assay <- CreateChromatinAssay(
  counts = counts$Peaks,
  sep = c(":", "-"),
  genome='mm10',
  fragments = "./e18_mouse_brain_fresh_5k_atac_fragments.tsv.gz",
  min.cells = 10
)

adata <- CreateSeuratObject(
  counts = chrom.assay,
  assay = "peaks"
)

# Load annotations from ENSEMBLE
annotations <- GetGRangesFromEnsDb(ensdb=EnsDb.Mmusculus.v79)
seqlevelsStyle(annotations) <- 'UCSC'
genome(annotations) <- 'mm10'
Annotation(adata) <- annotations

# Read Multivelo Labels and subset for common barcodes + add cell labels 
labels <- read.table(file = CELL.ANNOTATIONS, sep = '\t', header = TRUE, row.names = 1)
cells.use <- rownames(labels)
adata <- subset(adata, cells = cells.use)
metadata <- adata[[]]
metadata <- merge(metadata, labels, by='row.names')
adata$cell.type <- metadata$celltype

# QC metrics
adata <- TSSEnrichment(adata, fast=FALSE)
adata <- NucleosomeSignal(object = adata)

low.count <- quantile(adata$nCount_peaks, c(.02))
high.count <- quantile(adata$nCount_peaks, c(.98))
low.TSS <- quantile(adata$TSS.enrichment, c(.02))
high.TSS <- quantile(adata$TSS.enrichment, c(.98))
low.nucleo <- quantile(adata$nucleosome_signal, c(.02))
high.nucleo <- quantile(adata$nucleosome_signal, c(.98))

p1 <- VlnPlot(adata, features = "nCount_peaks") &  theme(legend.position = 'none') & geom_hline(yintercept = low.count, colour='blue')  & geom_hline(yintercept = high.count, colour='blue')
p2 <- VlnPlot(adata, features = "TSS.enrichment") &  theme(legend.position = 'none') & geom_hline(yintercept = low.TSS, colour='blue')  & geom_hline(yintercept = high.TSS, colour='blue')
p3 <- VlnPlot(adata, features = "nucleosome_signal") &  theme(legend.position = 'none') &  geom_hline(yintercept = low.nucleo, colour='blue')  & geom_hline(yintercept = high.nucleo, colour='blue')
wrap_plots(p1, p2, p3, ncol = 3)
ggsave(file.path(SAVING.FOLDER, "scATAC_QC.png"), plot=last_plot(), width = 10, height = 6)

adata <- subset(
  x = adata,
  subset = nCount_peaks > low.count &
    nCount_peaks < high.count &
    nucleosome_signal < high.nucleo &
    nucleosome_signal > low.nucleo &
    TSS.enrichment > low.TSS &
    TSS.enrichment < high.TSS
)

adata <- RunTFIDF(adata)

ATAC_preprocessing <- function(adata, k.param, n.dims, copy=T){
  object <- copy(adata) %||% copy %||% adata
  object <- FindTopFeatures(object, min.cutoff='q0') #all Features included
  object <- RunSVD(object) #since features=Null uses VariableFeatures
  depth.plot <- DepthCor(object) & geom_hline(yintercept = -0.7, color="red")  & geom_hline(yintercept = 0.7, color="red")
  depth.title <- paste("ATAC", "DepthCor", "K", k.param, "DIMS", n.dims, sep="_") 
  ggsave(file.path(SAVING.FOLDER, paste0(depth.title, '.png')), plot=depth.plot, width = 6, height = 6)

  #UMAP, Neighbors  
  dimensions <- 2:n.dims
  object <- RunUMAP(object, reduction='lsi', dims=dimensions, n.neighbors = k.param)
  umap.plot <- DimPlot(object, group.by = 'cell.type')
  umap.title <- paste("ATAC", "UMAP", "K", k.param, "DIMS", n.dims, sep="_")
  ggsave(file.path(SAVING.FOLDER, paste0(umap.title, '.png')), plot= umap.plot, width = 6, height = 6)
  
  object <- FindNeighbors(object, reduction='lsi', dims=dimensions, k.param= k.param)
  object <- FindClusters(object, verbose = FALSE, algorithm = 3)
  cluster.plot <- DimPlot(object, label = TRUE)
  cluster.title <- paste("ATAC", "CLUSTER", "K", k.param, "DIMS", n.dims, sep="_")
  ggsave(file.path(SAVING.FOLDER, paste0(cluster.title, '.png')), plot= cluster.plot, width = 6, height = 6)

}


# DA RIFARE CON expand.grid E QUALCOSA TIPO MAPPLY
n.dims <- seq(10, 50, by=10)
n.k <- seq(10, 90, by=10)
for(dim in n.dims){
  for(k in n.k){
    ATAC_preprocessing(adata, k, dim)
  }
}
