library(Signac)
library(Seurat)
library(SeuratDisk)
library(EnsDb.Mmusculus.v79)
library(data.table)
library(itertools)
library(ggplot2)
library(patchwork)
library(readr)

set.seed(52)

setwd('C:/Users/aless/Desktop/scVEMO')
PATH = file.path(getwd(), 'filtered_feature_bc_matrix')
CELL.ANNOTATIONS = file.path(getwd(), 'cell_annotations.tsv')
BARCODES.METRICS = file.path(getwd(), 'e18_mouse_brain_fresh_5k_per_barcode_metrics.csv')
SAVING.FOLDER = file.path(file.path(getwd(), 'signac_folder'), 'E18_embryonic_mouse_brain_ATAC')

if(!file.exists(SAVING.FOLDER)){dir.create(SAVING.FOLDER)}


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
  assay = "ATAC"
)

# Remove non developmental lineages (Microglia, Cajal-Retzius, Interneurons) using Multivelo cell annotations. 
cell.types <- data.frame(read_tsv(CELL.ANNOTATIONS))
cell.use <- subset(cell.types, celltype != 'Interneurons1' & celltype!= "Interneurons2" & 
                     celltype!="Interneurons3" & celltype != "Cajal-Retzius" & celltype != 'Microglia')
cell.use <- cell.use[, c(1)]
adata <- subset(adata, cells = cell.use)


# Load annotations from ENSEMBLE
annotations <- GetGRangesFromEnsDb(ensdb=EnsDb.Mmusculus.v79)
seqlevelsStyle(annotations) <- 'UCSC'
genome(annotations) <- 'mm10'
Annotation(adata) <- annotations


# QC metrics and filtering 
adata <- TSSEnrichment(adata, fast=FALSE)
adata <- NucleosomeSignal(object = adata)

low.count <- quantile(adata$nCount_ATAC, c(.02))
high.count <- quantile(adata$nCount_ATAC, c(.98))
low.TSS <- quantile(adata$TSS.enrichment, c(.02))
high.TSS <- quantile(adata$TSS.enrichment, c(.98))
low.nucleo <- quantile(adata$nucleosome_signal, c(.02))
high.nucleo <- quantile(adata$nucleosome_signal, c(.98))

p1 <- VlnPlot(adata, features = "nCount_ATAC") &  theme(legend.position = 'none') & geom_hline(yintercept = low.count, colour='blue')  & geom_hline(yintercept = high.count, colour='blue')
p2 <- VlnPlot(adata, features = "TSS.enrichment") &  theme(legend.position = 'none') & geom_hline(yintercept = low.TSS, colour='blue')  & geom_hline(yintercept = high.TSS, colour='blue')
p3 <- VlnPlot(adata, features = "nucleosome_signal") &  theme(legend.position = 'none') &  geom_hline(yintercept = low.nucleo, colour='blue')  & geom_hline(yintercept = high.nucleo, colour='blue')
wrap_plots(p1, p2, p3, ncol = 3)
ggsave(file.path(SAVING.FOLDER, "scATAC_QC.png"), plot=last_plot(), width = 10, height = 6)

adata <- subset(
  x = adata,
  subset = nCount_ATAC > low.count &
    nCount_ATAC < high.count &
    nucleosome_signal < high.nucleo &
    nucleosome_signal > low.nucleo &
    TSS.enrichment > low.TSS &
    TSS.enrichment < high.TSS
)

setwd(SAVING.FOLDER)
# LSI KNN UMAP + Clustering 
adata <- RunTFIDF(adata)
adata <- FindTopFeatures(adata, min.cutoff='q0') #all Features included
adata <- RunSVD(adata) #since features=Null uses VariableFeatures

depth.plot <- DepthCor(adata, n=30) & geom_hline(yintercept = -0.7, color="red")  & geom_hline(yintercept = 0.7, color="red")
depth.title <- paste("ATAC", "DepthCor", sep="_") 
ggsave(file.path(getwd(), paste0(depth.title, '.png')), plot=depth.plot, width = 6, height = 6)


ATAC_preprocessing<- function(row, adata, copy=T, plot=T){
  data <- copy(adata) %||% copy %||% adata

  dimensions <- 2:row['lsi'] # Skipping first dimension since highly correlated
  data <- RunUMAP(data, reduction='lsi', dims=dimensions, n.neighbors = row['k'])
  data <- FindNeighbors(data, reduction='lsi', dims=dimensions, k.param= row['k'])
  data <- FindClusters(data, verbose = FALSE, algorithm = 3)
  
  if(plot){
      cluster.title <- paste("ATAC", "CLUSTER", "K", row['k'], "DIMS", row['lsi'], sep="_")
      cluster.plot <- DimPlot(data, label = TRUE) + ggtitle(cluster.title)
      ggsave(file.path(getwd(), paste0(cluster.title, '.png')), plot= cluster.plot, width = 6, height = 6)
  }
  
  filename <- paste(paste("scATAC", paste0(row['k'], "K", row['lsi'], "LSI"), sep="_"), ".h5Seurat")
  SaveH5Seurat(data, filename = filename)
  Convert(filename, dest = "h5ad")

}

grid <- expand.grid(seq(10, 30, 5), c(10,20,30,50,60,80))
colnames(grid) <- c('lsi', 'k')
  
apply(grid, MARGIN=1, FUN=ATAC_preprocessing, adata=adata, copy=TRUE, plot=TRUE)



