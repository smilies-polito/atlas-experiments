library(dplyr)
library(tibble)
library(Seurat)
library(patchwork)
library(Signac)
library(rtracklayer)
library(txdbmaker)
library(ggplot2)
library(SeuratDisk)
library(SeuratObject)
library(Matrix)

set.seed(52)
setwd("...") #folder where data is stored 
options(Seurat.object.assay.version = "v3")

change_cell_names <- function(assay, experiment.string){
  # Add experiment name to distinguish cells among experiments
  cell.names <- colnames(assay)
  cell.names <- gsub("-1$", "", cell.names) #remove -1
  cell.names <- paste0(experiment.string, cell.names)
  colnames(assay) <- cell.names
  return(assay)
}


plot_genes <- function(object, gene.name, plot.folder){
  feature.plot<- FeaturePlot(object, feature = gene.name)
  vln.plot<- VlnPlot(object, group.by="Cluster.Name", feature = gene.name)
  wrap_plots(feature.plot, vln.plot, ncol=2)
  ggsave(file.path(plot.folder, paste0(gene.name, ".png")), plot=last_plot(), width = 16, height = 10)
}

saving.folder <- file.path(getwd(), "Robjects")
if(!dir.exists(saving.folder)){dir.create(saving.folder)}

plot.folder <- file.path(getwd(), "Rplots")
if(!dir.exists(plot.folder)){dir.create(plot.folder)}

data.path.dc2r2_r1 <- file.path(getwd(), "dc2r2_r1")
# data.path.dc2r2_r2 <- file.path(getwd(), "dc2r2_r2")
data.path.dc1r3_r1 <- file.path(getwd(), "dc1r3_r1")

gtf.path <- file.path(getwd(), "gencode.v27.annotation.gtf")
gtf.file <- makeTxDbFromGFF(gtf.path, format="gtf")
annotations <-  exonsBy(gtf.file, by="gene")

counts.dc1r3_r1 <- Read10X(file.path(data.path.dc1r3_r1, "filtered_feature_bc_matrix"))
counts.dc2r2_r1 <- Read10X(file.path(data.path.dc2r2_r1, "filtered_feature_bc_matrix"))

rna.dc1r3_r1 <- counts.dc1r3_r1$`Gene Expression`
atac.dc1r3_r1 <- counts.dc1r3_r1$Peaks

rna.dc2r2_r1 <- counts.dc2r2_r1$`Gene Expression`
atac.dc2r2_r1 <- counts.dc2r2_r1$Peaks

# Geni dsono tutti uguali ho fatto check posso fare semplice cbind
rna.dc1r3_r1 <- change_cell_names(rna.dc1r3_r1,  "hft_ctx_w21_dc1r3_r1_")
rna.dc2r2_r1 <- change_cell_names(rna.dc2r2_r1,  "hft_ctx_w21_dc2r2_r1_")
merged.rna <- cbind(rna.dc1r3_r1, rna.dc2r2_r1)
obj<- CreateSeuratObject(merged.rna, assay="RNA")

# Peaks differ in the atac counts (only 177 peaks in common) cannot use cbind
chrom.assay.dc1r3_r1 <- CreateChromatinAssay(
  counts = atac.dc1r3_r1,
  sep = c(":", "-"),
  genome='mm10',
  fragments = file.path(data.path.dc1r3_r1,"atac_fragments.tsv.gz"),
  annotation = annotations$ENSG00000000003.14
)
chrom.assay.dc1r3_r1 <- change_cell_names(chrom.assay.dc1r3_r1,  "hft_ctx_w21_dc1r3_r1_")

chrom.assay.dc2r2_r1 <- CreateChromatinAssay(
  counts = atac.dc2r2_r1,
  sep = c(":", "-"),
  genome='mm10',
  fragments = file.path(data.path.dc2r2_r1,"atac_fragments.tsv.gz"),
  annotation = annotations$ENSG00000000003.14
)
chrom.assay.dc2r2_r1 <- change_cell_names(chrom.assay.dc2r2_r1,  "hft_ctx_w21_dc2r2_r1_")
merged.atac <-merge(chrom.assay.dc1r3_r1, chrom.assay.dc2r2_r1)

obj[["ATAC"]] <- merged.atac

# Load valid cells according to https://doi.org/10.1016/j.cell.2021.07.039 and subset for valid cells only (those passing QC metrics)
metadata <- read.delim(file.path(getwd(), "multiome_cell_metadata.txt"), sep="\t", header=TRUE)
metadata <- tibble::column_to_rownames(metadata, "Cell.ID")
metadata <- subset(metadata, select="seurat_clusters")

# Filter valid cells according to https://doi.org/10.1016/j.cell.2021.07.039 (those passing QC metrics)
obj<- subset(obj, cells=rownames(metadata))

# Add cell cluster according to https://doi.org/10.1016/j.cell.2021.07.039 
obj[[]]<- merge(obj[[]], metadata, by="row.names", all.x=TRUE)

# Add cluster names 
cluster.names <- read.delim(file.path(getwd(), "multiome_cluster_names.txt"), header=T, sep="\t")
cluster.names.rna <- cluster.names[cluster.names$Assay=="Multiome RNA", ]
obj[[]] <- dplyr::left_join(obj[[]], cluster.names.rna, by = c("seurat_clusters"="Cluster.ID"))

# Subset for cortical development: remove interneurons (IN), microglia (MG),
# endothelial cells EC/pericytes Peric which refer to bloos vessels development
clusters.to.keep <- c("Cyc. Prog.", "GluN2", "GluN3", "GluN4", "GluN5", "mGPC/OPC", "nIPC/GluN1", "RG", "SP")
cells.to.keep <- rownames(subset(obj[[]], Cluster.Name!="IN1" & Cluster.Name!="IN2" & Cluster.Name!="IN3" &
                                   Cluster.Name!="MG" & Cluster.Name!="EC/Peric."))
obj <- subset(obj, cells=cells.to.keep)   

experiment <- sapply(obj[[]]$Row.names, function(c) substr(c, 13, 20))
obj[["experiment"]] <- as.factor(experiment)

# Preprocessing
DefaultAssay(obj) <- "RNA"
obj<- NormalizeData(obj)
obj <- FindVariableFeatures(obj)
obj <- ScaleData(obj)
obj <- RunPCA(obj) 

DefaultAssay(obj) <- "ATAC"
obj <- RunTFIDF(obj)
obj <- FindTopFeatures(obj, min.cutoff='q0') #all Features included
obj <- RunSVD(obj, n=50) 


p0 <- ElbowPlot(obj, ndims=50, reduction='pca') 
variance_explained <- data.frame(cumsum(Stdev(obj, reduction='pca')**2/obj@reductions$pca@misc$total.variance*100), seq(1:50))
names(variance_explained) <- c('variance', 'element')
p1 <- ggplot(data=variance_explained, aes(element, variance)) + geom_point(aes(element, variance), size=2) + 
  xlab("") + ylab("Explained Variance") &
  geom_hline(yintercept=variance_explained$variance[10], color='blue', linetype='dotted') &
  geom_hline(yintercept=variance_explained$variance[15], color='blue', linetype='dotted') &
  geom_hline(yintercept=variance_explained$variance[20], color='blue', linetype='dotted') & 
  geom_hline(yintercept=variance_explained$variance[25], color='blue', linetype='dotted') &
  geom_hline(yintercept=variance_explained$variance[30], color='blue', linetype='dotted')
wrap_plots(p0, p1, ncol=2)

depth.plot <- DepthCor(obj, reduction='lsi', n=30) & geom_hline(yintercept = -0.7, color="red")  & geom_hline(yintercept = 0.7, color="red")

# Neighbors, PCA, UMAP
DefaultAssay(obj)<- "RNA"
n.lsi <- 10
n.pca <- 30 
k.nn <- 30

obj <- FindNeighbors(
  object=obj,
  k.param = k.nn,
  assay="RNA", 
  dims = 1:n.pca, 
  return.neighbor = TRUE
)

obj <- RunUMAP(object=obj, nn.name = "RNA.nn", reduction.name="umap", verbose=FALSE)
cluster.plot <- DimPlot(obj, reduction='umap', group.by = "Cluster.Name")
batch.plot <- DimPlot(obj, reduction='umap', group.by="experiment")
wrap_plots(cluster.plot, batch.plot, ncol=2)
ggsave(file.path(plot.folder, "dimplot.png"), plot=last_plot(), width=20, height=10)


obj <- FindMultiModalNeighbors(
  object=obj, 
  k.nn = k.nn, 
  reduction.list = list("pca", "lsi"), 
  dims.list = list(1:n.pca, 2:n.lsi), 
  modality.weight.name = list("RNA.weight", "ATAC.weight"),
  verbose=FALSE
)

obj <- RunUMAP(object=obj, nn.name = "weighted.nn", reduction.name="umap", verbose=FALSE)
cluster.plot <- DimPlot(obj, reduction='umap', group.by = "Cluster.Name")
batch.plot <- DimPlot(obj, reduction='umap', group.by="experiment")
wrap_plots(cluster.plot, batch.plot, ncol=2)
ggsave(file.path(plot.folder, "dimplot_multimodal.png"), plot=last_plot(), width=20, height=10)

# Feature plot and Vln plots for gene that have been associated to increasing and decreasing pseudotime in GluN development 
genes <- c("MEF2C", "NEUROG1", "NEUROD2", "JUN", "FOS", "RUNX1", "HSPA1A", "HSPA1B") 
genes <- c("GRIN2A", "GRIN2B", "GRIN3A")
lapply(genes, function(gene) plot_genes(object=obj, gene.name = gene, plot.folder = plot.folder))


# Saving
adata.path = file.path(saving.folder, paste0("multiomics_", k.nn, 'K', n.pca, "PC", n.lsi, 'LSI', '.h5Seurat'))
SaveH5Seurat(obj, filename=adata.path)
rna.dest = file.path(saving.folder, paste0("RNA_", k.nn, "K", n.pca, "PC", n.lsi, "LSI", '.h5ad'))
Convert(adata.path, assay="RNA", dest=rna.dest)
atac.dest = file.path(saving.folder, paste0("ATAC_", k.nn, "K", n.pca, "PC", n.lsi, "LSI", '.h5ad'))
Convert(adata.path, assay="ATAC", dest=atac.dest)

