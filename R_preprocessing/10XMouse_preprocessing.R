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
library(readr)
library(dplyr)

set.seed(52)

setwd('C:/Users/aless/Desktop/scVEMO')
PATH = file.path(getwd(), 'filtered_feature_bc_matrix')
LOOM = file.path(getwd(), '10X_multiome_mouse_brain.loom')
CELL.TYPES = file.path(getwd(), 'cell_annotations.tsv')
SAVING.FOLDER = file.path(getwd(), 'data_folder')
PLOT.FOLDER = file.path(getwd(), "figures")

if (!dir.exists(SAVING.FOLDER)){dir.create(SAVING.FOLDER)}
if (!dir.exists(PLOT.FOLDER)) {dir.create(PLOT.FOLDER)}  

SharedCounts <- function(adata, min.threshold){
  # Emulates scvelo search for genes with shared counts above the threshold:
  # params: 
  # adata: SeuratObject with spliced and unspliced assays
  # min.threshold: min number of cells for shared count
  # output:
  # list of booleans whose shared count > min.threshold
  
  #1. pointwise product of spliced and unspliced 
  non.zeros <- (adata[['spliced']]@counts>0)*(adata[['unspliced']]@counts>0)
  X <- adata[['spliced']]@counts*non.zeros + adata[['unspliced']]@counts*non.zeros

  #2. Sum along columns to count cells 
  matrix.sum <- apply(X, MARGIN=c(1), sum)
  return(matrix.sum>=min.threshold)
}


annotations <- GetGRangesFromEnsDb(ensdb=EnsDb.Mmusculus.v79)

seqlevelsStyle(annotations) <- 'UCSC'
genome(annotations) <- 'mm10'

loom <- connect(filename=LOOM, mode="r+", skip.validate=TRUE)

matrix <- Matrix(loom[['matrix']][, ], sparse=TRUE)
rownames(matrix) <- paste(substr(loom[['col_attrs/CellID']][], 10, 25), '1', sep='-')
colnames(matrix) <- loom[['row_attrs/Gene']][]

unspliced <- Matrix(loom[['layers/unspliced']][, ], sparse=TRUE)
spliced <- Matrix(loom[['layers/spliced']][, ], sparse=TRUE)
dimnames(spliced)<-dimnames(matrix)
dimnames(unspliced)<-dimnames(matrix)

loom$close_all()

matrix <- t(matrix)
adata <- CreateSeuratObject(CreateAssayObject(matrix))
adata[['spliced']]<- CreateAssayObject(t(spliced))
adata[['unspliced']] <- CreateAssayObject(t(unspliced))


# Filter genes with min shared counts > 10 (as in scVELO)
indices <- SharedCounts(adata, 10)
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

# Remove non developmental lineages (Microglia, Cajal-Retzius, Interneurons) using Multivelo cell annotations. 
cell.types <- data.frame(read_tsv(CELL.TYPES))
cell.use <- subset(cell.types, celltype != 'Interneurons1' & celltype!= "Interneurons2" & 
                     celltype!="Interneurons3" & celltype != "Cajal-Retzius" & celltype != 'Microglia')
cell.use <- cell.use[, c(1)]
adata <- subset(adata, cells = cell.use)

DefaultAssay(adata)<-'ATAC'
adata <- NucleosomeSignal(adata)
adata <- TSSEnrichment(adata)

# Check for nans: incompatibilities between rna and atac (only cell in position 1850)
print(which(is.na(adata$nCount_ATAC)))
print(which(is.na(adata$TSS.enrichment)))
print(which(is.na(adata$nucleosome_signal)))
print(which(is.na(adata$nCount_RNA)))

cells.use <- colnames(adata)
to.be.removed <- cells.use[c(1850)]
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
ggsave(file.path(PLOT.FOLDER, "quality_metrics.png"), plot=last_plot(), width = 14, height = 6)
 
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

DefaultAssay(adata) <- "RNA"
adata <- NormalizeData(adata)
adata <- FindVariableFeatures(adata)
adata <- ScaleData(adata)
adata <- RunPCA(adata) 

#   Add var.features to spliced and unspliced assays: 
#   enables saving with Seurat Disk spliced and unspliced arrays along with RNA (single .h5ad file)
var.features <- adata[['RNA']]@var.features
adata[['spliced']]@var.features <- adata[['RNA']]@var.features
adata[['unspliced']]@var.features <- adata[['RNA']]@var.features


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
ggsave(file.path(PLOT.FOLDER, "PCA_varianceElbow.png"), plot=last_plot(), width = 14, height = 6)


DefaultAssay(adata)<- 'ATAC'
adata <- RunTFIDF(adata)
adata <- FindTopFeatures(adata, min.cutoff='q0') #all Features included
adata <- RunSVD(adata, n=50) 

depth.plot <- DepthCor(adata, reduction='lsi', n=30) & geom_hline(yintercept = -0.7, color="red")  & geom_hline(yintercept = 0.7, color="red")
ggsave(file.path(PLOT.FOLDER, 'SVD_depthcor.png'), plot=depth.plot, width = 6, height = 6)

##################################################################################################################################################
############################################################## PREPROCESSING scRNA ###############################################################
##################################################################################################################################################
RNA.FOLDER = file.path(SAVING.FOLDER, 'scRNA')
if (!dir.exists(RNA.FOLDER)){dir.create(RNA.FOLDER)}


DefaultAssay(adata)<- "RNA"
scRNA.grid <- expand.grid(seq(10, 30, 5), c(10, 20, 30, 50, 60, 80), c(1.0))
colnames(scRNA.grid) <- c('pca', 'k',  'res')

scRNA_preprocessing <- function(row, adata, folder, copy=TRUE){
  data <- copy(adata) %||% copy %||% adata
  # Neighbors and clusters using scanpy: already integrated for scVelo velocities computations.

  adata.path = file.path(folder, paste0("scRNA_", row['k'], 'K', row['pca'], 'PC', '.h5Seurat'))

  # Subset for variable features to be able to save and convert using SeuratDisk also spliced and unspliced layers.
  genes.in.use <- rownames(GetAssayData(data, assay = "RNA", slot = "scale.data"))

  spliced <- GetAssayData(data, assay = "spliced") %>% .[rownames(.) %in% genes.in.use,] %>% CreateAssayObject()
  unspliced <- GetAssayData(data, assay = "unspliced") %>% .[rownames(.) %in% genes.in.use,] %>% CreateAssayObject()
  data[["spliced"]] <- spliced
  data[["unspliced"]] <- unspliced

  SaveH5Seurat(data, filename=adata.path)
  Convert(adata.path, dest='h5ad')
}

apply(scRNA.grid, MARGIN=1, FUN=scRNA_preprocessing, adata=adata, folder=RNA.FOLDER, copy=TRUE)


##################################################################################################################################################
############################################################## PREPROCESSING scATAC ##############################################################
##################################################################################################################################################
ATAC.FOLDER = file.path(SAVING.FOLDER, 'scATAC')
ATAC.PLOT = file.path(PLOT.FOLDER, 'scATAC_clusters')
if (!dir.exists(ATAC.FOLDER)){dir.create(ATAC.FOLDER)}
if (!dir.exists(ATAC.PLOT)){dir.create(ATAC.PLOT)}

DefaultAssay(adata)<- "ATAC"
scATAC.grid <- expand.grid(seq(10, 30, 5), c(10, 20, 30, 50, 60, 80), c(1.0))
colnames(scATAC.grid) <- c('lsi', 'k',  'res')

scATAC_preprocessing <- function(row, adata, folder, plot.folder, copy=TRUE){
  data <- copy(adata) %||% copy %||% adata
  data <- FindNeighbors(data, k.param=row['k'], reduction="lsi", dims=2:row['lsi'])
  data <- FindClusters(data, verbose=F, algorithm =3, graph.name = "ATAC_snn", cluster.name="ATAC_clusters")
  data <- RunUMAP(data, reduction.key = 'lsi', dims=2:row['lsi'], n.neighbors=row['k'], reduction.name = "umap")
  cluster.title <- paste("scATAC", "K=", row['k'], "LSI=", row['lsi'], sep=" ")
  cluster.path <- paste0("scATAC", "_", "K", row['k'], "LSI", row['lsi'], ".png")
  cluster.plot <- DimPlot(data, label = TRUE, reduction = 'umap') + ggtitle(cluster.title)
  ggsave(file.path(plot.folder, cluster.path), plot= cluster.plot, width = 6, height = 6)

  adata.path = file.path(folder, paste0("scATAC_", row['k'], 'K', row['lsi'], 'LSI', '.h5Seurat'))
  SaveH5Seurat(data, filename=adata.path)
  Convert(adata.path, dest='h5ad')
}

apply(scATAC.grid, MARGIN=1, FUN=scATAC_preprocessing, folder=ATAC.FOLDER, plot.folder=ATAC.PLOT, adata=adata, copy=TRUE)


##################################################################################################################################################
############################################################## PREPROCESSING multiomics ##########################################################
##################################################################################################################################################
MULTIOMICS.FOLDER = file.path(SAVING.FOLDER, 'multiomics')
MULTIOMICS.PLOT = file.path(PLOT.FOLDER, 'multiomics_clusters')
if (!dir.exists(MULTIOMICS.FOLDER)){dir.create(MULTIOMICS.FOLDER)}
if (!dir.exists(MULTIOMICS.PLOT)){dir.create(MULTIOMICS.PLOT)}

DefaultAssay(adata)<- "RNA"
multiomics.grid <- expand.grid(seq(10,30,5), seq(10, 30, 5), c(10, 20, 30, 50, 60, 80), c(1.0))
colnames(multiomics.grid) <- c('lsi', 'pca', 'k', 'res')


multiomics_preprocessing<- function(row, adata, path, folder, plot.folder, copy=TRUE, save.neighbors=FALSE){
  data <- copy(adata) %||% copy %||% adata
  
  genes.in.use <- rownames(GetAssayData(data, assay = "RNA", slot = "scale.data"))
  
  spliced <- GetAssayData(data, assay = "spliced") %>% .[rownames(.) %in% genes.in.use,] %>% CreateAssayObject()
  unspliced <- GetAssayData(data, assay = "unspliced") %>% .[rownames(.) %in% genes.in.use,] %>% CreateAssayObject()
  data[["spliced"]] <- spliced
  data[["unspliced"]] <- unspliced

  data <- FindMultiModalNeighbors(
    object = data,
    k.nn = row['k'],
    reduction.list = list("pca", "lsi"),
    dims.list = list(1:row['pca'], 2:row['lsi']),
    modality.weight.name = list("RNA.weight", "ATAC.weight"), 
    verbose = TRUE
    )

 data <- RunUMAP(object = data, nn.name = "weighted.nn", reduction.name = "umap", verbose = TRUE)
 data <- FindClusters(data, graph.name = "wsnn", algorithm = 3, random.seed= 52, verbose = FALSE)

 cluster.title <- paste("multiomics", "K=", row['k'], "PC=", row['pca'], "LSI=", row['lsi'], sep=" ")
 cluster.path <- paste0("multiomics", "_", "K", row['k'], "PC", row['pca'], "LSI", row['lsi'], ".png")
 cluster.plot <- DimPlot(data, reduction='umap', group.by = "seurat_clusters") + ggtitle(cluster.title) 
 ggsave(file.path(plot.folder, cluster.path), plot= cluster.plot, width = 6, height = 6)

 # Non funziona per salvare ATAC spliced and unspliced devo salvare diversi assays
 adata.path = file.path(folder, paste0("multiomics_", row['k'], 'K', row['pca'], "PC", row['lsi'], 'LSI', '.h5Seurat'))
 SaveH5Seurat(data, filename=adata.path)
 rna.dest = file.path(folder, paste0("RNA_", row['k'], "K", row['pca'], "PC", row['lsi'], "LSI", '.h5ad'))
 Convert(adata.path, assay="RNA", dest=rna.dest)
 atac.dest = file.path(folder, paste0("ATAC_", row['k'], "K", row['pca'], "PC", row['lsi'], "LSI", '.h5ad'))
 Convert(adata.path, assay="ATAC", dest=atac.dest)
 

 if(save.neighbors){
   neighbors =  Neighbors(data, "weighted.nn")
   path = paste0(row['k'], "K", row['lsi'], "LSI", row['pca'], "PC", ".txt")
   write.table(neighbors@nn.idx, file.path(folder, paste0("idx_", path)), sep=',', row.names=F, col.names=F, quote=F)
   write.table(neighbors@nn.dist, file.path(folder, paste0("dist_", path)), sep=',', row.names=F, col.names=F, quote=F)
 }
 
}

apply(multiomics.grid, MARGIN=1, FUN=multiomics_preprocessing, adata=adata, folder = MULTIOMICS.FOLDER, plot.folder = MULTIOMICS.PLOT, copy=TRUE, save.neighbors=FALSE)


##################################################################################################################################################
############################################################## PREPROCESSING multivelo ###########################################################
##################################################################################################################################################
apply(multiomics.grid, MARGIN=1, FUN=multiomics_preprocessing, adata=adata, folder = MULTIOMICS.FOLDER, plot.folder = MULTIOMICS.PLOT, copy=TRUE, save.neighbors=TRUE)
