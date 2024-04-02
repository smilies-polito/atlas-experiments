library(Signac)
library(Seurat)
library(loomR)
library(EnsDb.Mmusculus.v79)
library(Matrix)
library(SeuratDisk)
library(ggplot2)
library(patchwork)
library(data.table)
library(itertools)

set.seed(52)

setwd('C:/Users/aless/Desktop/scVEMO')
PATH = file.path(getwd(), 'filtered_feature_bc_matrix')
LOOM = file.path(getwd(), '10X_multiome_mouse_brain.loom')
SAVING.FOLDER = file.path(getwd(), 'signac_data', 'E18_embryonic_mouse_brain_multiomics')

if (!file.exists(SAVING.FOLDER)){dir.create(SAVING.FOLDER)}

source("save_scATAC.R")

SharedCounts <- function(spliced, unspliced, min.threshold){
  # Emulates scvelo search for genes with shared counts above the threshold:
  # params: 
  # spliced: matrix with spliced counts
  # unspliced: matrix with unspliced counts
  # min.threshold: min number of cells for shared count
  # output:
  # list of booleans whose shared count > min.threshold
  
  #1. pointwise product of spliced and unspliced 
  non.zeros <- (spliced>0)*(unspliced>0)
  X <- spliced*non.zeros + unspliced*non.zeros
  
  #2. Sum along columns to count cells 
  matrix.sum <- apply(X, MARGIN=c(2), sum)
  return(matrix.sum>=min.threshold)
}


annotations <- GetGRangesFromEnsDb(ensdb=EnsDb.Mmusculus.v79)
seqlevelsStyle(annotations) <- 'UCSC'
genome(annotations) <- 'mm10'

loom <- connect(filename=LOOM, mode="r+", skip.validate=TRUE)

matrix <- Matrix(loom[['matrix']][, ], sparse=TRUE)
rownames(matrix) <- paste(substr(loom[['col_attrs/CellID']][], 10, 25), '1', sep='-')
colnames(matrix) <- loom[['row_attrs/Gene']][]
matrix <- t(matrix)

unspliced <- Matrix(loom[['layers/unspliced']][, ], sparse=TRUE)
spliced <- Matrix(loom[['layers/spliced']][, ], sparse=TRUE)

loom$close_all()

adata <- CreateSeuratObject(CreateAssayObject(matrix))
# adata[["percent.mt"]] <- PercentageFeatureSet(adata, pattern = "^mt-") # mitopercent = 0 for loom 


# Filter for min shared counts > 10: 
indices <- SharedCounts(spliced, unspliced, 10)
genes.use <- rownames(adata)
to.be.removed <- genes.use[which(indices==FALSE)] 
genes.use <- genes.use[! genes.use %in% to.be.removed]
adata <- subset(adata, features = genes.use)


# Add ATAC values 
counts <- Read10X(PATH)
chromassay <- CreateChromatinAssay(
   counts = counts$Peaks,
   sep = c(":", "-"),
   genome='mm10',
   fragments = "./e18_mouse_brain_fresh_5k_atac_fragments.tsv.gz",
   min.cells = 10,
   annotation = annotations
)

adata[['ATAC']]<- chromassay

DefaultAssay(adata)<-'ATAC'
adata <- NucleosomeSignal(adata)
adata <- TSSEnrichment(adata)

# Check for nanssince three cells are in RNA but not in ATAC
# print(which(is.na(adata$nCount_peaks)))
# print(which(is.na(adata$TSS.enrichment)))
# print(which(is.na(adata$nucleosome_signal)))
# print(which(is.na(adata$nCount_RNA)))
# remove three cells in position 2478 4347 4766 they are present in RNA assay but not in peak assay

cells.use <- colnames(adata)
to.be.removed <- cells.use[c(2478, 4347, 4766)]
cells.use <- cells.use[! cells.use %in% to.be.removed]
adata <- subset(adata, cells = cells.use)

low.peaks <- quantile(adata$nCount_ATAC, c(.02))
high.peaks <- quantile(adata$nCount_ATAC, c(.98))
low.TSS <- quantile(adata$TSS.enrichment, c(.02))
high.TSS <- quantile(adata$TSS.enrichment, c(.98))
low.nucleo <- quantile(adata$nucleosome_signal, c(.02))
high.nucleo <- quantile(adata$nucleosome_signal, c(.98))
low.counts <- quantile(adata$nCount_RNA, c(0.02))
high.counts <- quantile(adata$nCount_RNA, c(.98))

p1 <- VlnPlot(adata, features = "nCount_ATAC") &  theme(legend.position = 'none') & geom_hline(yintercept = low.peaks, colour='blue')  & geom_hline(yintercept = high.peaks, colour='blue')
p2 <- VlnPlot(adata, features = "TSS.enrichment") &  theme(legend.position = 'none') & geom_hline(yintercept = low.TSS, colour='blue')  & geom_hline(yintercept = high.TSS, colour='blue')
p3 <- VlnPlot(adata, features = "nucleosome_signal") &  theme(legend.position = 'none') &  geom_hline(yintercept = low.nucleo, colour='blue')  & geom_hline(yintercept = high.nucleo, colour='blue')
p4 <- VlnPlot(adata, features= "nCount_RNA") & theme(legend.position='none') & geom_hline(yintercept=low.counts, colour='blue') & geom_hline(yintercept= high.counts, colour='blue')
wrap_plots(p1, p2, p3, p4, ncol = 4)
ggsave(file.path(SAVING.FOLDER, "multiomics_scATAC_QC.png"), plot=last_plot(), width = 14, height = 6)
 
adata <- subset(
   x = adata,
   subset = nCount_ATAC > low.peaks &
     nCount_ATAC < high.peaks &
     nucleosome_signal < high.nucleo &
     nucleosome_signal > low.nucleo &
     TSS.enrichment > low.TSS &
     TSS.enrichment < high.TSS &
     nCount_RNA > low.counts &
     nCount_RNA < high.counts
)
# Remain 4236 cells 

# Preprocessing
DefaultAssay(adata) <- "RNA"
adata <- NormalizeData(adata)
adata <- FindVariableFeatures(adata)
adata <- ScaleData(adata)
adata <- RunPCA(adata) # Automatically uses highly var features since featuers=NULL

# Elbow Plot + % explained variance
p0 <- ElbowPlot(adata, ndims=50, reduction='pca') 
variance_explained <- data.frame(cumsum(Stdev(adata, reduction='pca')**2/adata@reductions$pca@misc$total.variance*100), seq(1:50))
names(variance_explained) <- c('variance', 'element')
p1 <- ggplot(data=variance_explained, aes(element, variance)) + geom_point(aes(element, variance), size=2) + 
  xlab("") + ylab("Explained Variance") &
  geom_hline(yintercept=variance_explained$variance[10], color='blue', linetype='dotted') &
  geom_hline(yintercept=variance_explained$variance[15], color='blue', linetype='dotted') &
  geom_hline(yintercept=variance_explained$variance[20], color='blue', linetype='dotted') & 
  geom_hline(yintercept=variance_explained$variance[25], color='blue', linetype='dotted') &
  geom_hline(yintercept=variance_explained$variance[30], color='blue', linetype='dotted')
wrap_plots(p0, p1, ncol=2)
ggsave(file.path(SAVING.FOLDER, "multiomics_PCA.png"), plot=last_plot(), width = 14, height = 6)


DefaultAssay(adata)<- 'ATAC'
adata <- RunTFIDF(adata)
adata <- FindTopFeatures(adata, min.cutoff='q0') #all Features included
adata <- RunSVD(adata, n=50) 

depth.plot <- DepthCor(adata, reduction='lsi') & geom_hline(yintercept = -0.7, color="red")  & geom_hline(yintercept = 0.7, color="red")
ggsave(file.path(SAVING.FOLDER, 'multiomics_SVD_depthcor.png'), plot=depth.plot, width = 6, height = 6)

SaveSeuratAssay(adata, SAVING.FOLDER, assay="ATAC") # No subset for variable features since all features included 
SaveSeuratAssay(adata, SAVING.FOLDER, assay="RNA", var.features = TRUE)

grid <- expand.grid(seq(10, 30, 5), seq(10, 30, 5), c(0.7, 1.0, 1.4), c(10,20,30,50,60,80))
colnames(grid) <- c('lsi', 'pca', 'res', 'k')

preprocessing<- function(row, adata, path, copy=TRUE){
  folder.path <- file.path(path, paste(row['k'], row['lsi'], row['pca'], row['res'], sep='_'))
  dir.create(folder.path)
  data <- copy(adata) %||% copy %||% adata
  data <- FindMultiModalNeighbors(
    object = data, 
    k.nn <- row['k'], 
    reduction.list = list("pca", "lsi"), 
    dims.list = list(1:row['pca'], 2:row['lsi']),
    modality.weight.name = "RNA.weight", 
    verbose = TRUE
    )

 data <- RunUMAP(
    object = data,
    nn.name = "weighted.nn",
    reduction.name = "wnn_umap",
    reduction.key= "wnnUMAP_",
    verbose = TRUE
    )

 data <- FindClusters(
   data, 
   graph.name = "wsnn",
   algorithm = 3, 
   random.seed=52, 
   resolution= row['res'], 
   verbose = FALSE)
 
 path<- file.path(folder.path,paste("multiomics", paste0(row['k'], "K", row['lsi'], "LSI", row['pca'], "PCA", row['res'], "res", '.png'), sep='_'))
 title <- paste("WNN clusters K=", row['k'], "PC=", row['pca'], "LSI=", row['lsi'], "res=", row['res'], sep=' ')
 clusters.plot <- DimPlot(data, reduction='wnn_umap', group.by = "seurat_clusters") + ggtitle(title)
 ggsave(path, plot=clusters.plot, width = 6, height = 6)
 
 SaveSeuratMetadata(data, folder.path)
 SaveSeuratReduction(data, folder.path, 'pca')
 SaveSeuratReduction(data, folder.path, 'lsi')
 SaveSeuratReduction(data, folder.path, 'wnn_umap')
 SaveSeuratNeighbors(data, folder.path, 'wsnn')
}

apply(grid, MARGIN=1, FUN=preprocessing, adata=adata, copy=TRUE, path=SAVING.FOLDER)


