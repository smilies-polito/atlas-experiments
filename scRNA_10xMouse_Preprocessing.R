library(Seurat)


WD_PATH<- file.path("SET PATH WITH DATA ")
setwd(WD_PATH)
DATA_PATH <-file.path(WD_PATH, "filtered_feature_bc_matrix")

data <-Read10X(paste0(DATA_PATH, "/"))
adata<- CreateSeuratObject(counts = data$`Gene Expression`, min.cells = 10)

# PREPROCESSING ----------------------------------------------------------------------
#Compute proportions of transcripts mapping to mitochondrial genes
adata[['mt']] <- PercentageFeatureSet(object=adata, pattern="^mt-")

VlnPlot(adata, features = c('nFeature_RNA', 'nCount_RNA', 'mt'), ncol=3)
plot1 <- FeatureScatter(adata, feature1 = "nCount_RNA", feature2 = "mt")
plot2 <- FeatureScatter(adata, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")
plot1+plot2

# Remove cells with mitopercent greater than 5% and within 2 and 98 percentiles
lower <- as.integer(quantile(adata[['nFeature_RNA']]$nFeature_RNA, probs=c(0.02)))
upper <- as.integer(quantile(adata[['nFeature_RNA']]$nFeature_RNA, probs=c(0.98)))
adata <- subset(adata, subset = nFeature_RNA > lower & nFeature_RNA < upper & mt < 5)

# Log-normalize data
adata <- NormalizeData(adata)

# Highly variable features
adata <- FindVariableFeatures(adata, selection.method = "vst", nfeatures = 2000)
plot1 <- VariableFeaturePlot(adata)

# Scaling data
all.genes <- rownames(adata)
adata <- ScaleData(adata, features = all.genes)

# Save preprocessed data
saveRDS(adata, file = file.path(WD_PATH, '10xEmbryonicMouseBrain_Preprocessed.rds'))
    

# STUDY PCs AND NEIGHBORS ------------------------------------------------------------------------
adata <- RunPCA(adata, features = VariableFeatures(adata), npcs = 100)
ElbowPlot(adata, ndims = 100, reduction='pca')
pcs <- 30 #da elbow plot
k <- as.integer(sqrt(ncol(adata))) # rule of thumb

adata <- FindNeighbors(adata, dims=1:pcs, k.param=k)
adata <- FindClusters(adata, random.seed=52)
adata <- RunUMAP(adata, dims= 1:pcs)
DimPlot(adata, reduction = "umap")

