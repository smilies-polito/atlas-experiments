# SAVE DATA TO BE LOADED INTO PYTHON AND PREPROCESSED USING MUON
library(Seurat)
library(Signac)
library(Matrix)

setwd("...") # set to repository 
rds.file.name <- ... # name file from figshare
donor <- ... #either donor1 o donor2 
data.path <- file.path(getwd(), "data", "lineage_tracing", donor)
data <- readRDS(file.path(data.path, rds.file.name))

gene.names <- rownames(data[["RNA"]])
cells.barcodes <- colnames(data[["RNA"]])
peak.names <- rownames(data[["ATAC"]])
save.path <- file.path(data.path, "all.genes.csv")
write.table(gene.names, save.path, row.names = F, col.names = F)
save.path <- file.path(data.path, "all.barcodes.csv")
write.csv(cells.barcodes, save.path, row.names = F)
save.path <- file.path(data.path, "all.peaks.csv")
write.table(peak.names, save.path, row.names = F, col.names = F)

rna.counts <- t(data[["RNA"]]$counts)
save.path <- file.path(data.path, "all.rna.mtx")
writeMM(rna.counts, save.path)

atac.counts <- t(data[["ATAC"]]$counts)
save.path <- file.path(data.path, "all.atac.mtx")
writeMM(atac.counts, save.path)

metadata <- data[[]]
head(metadata)
save.path <- file.path(data.path, "all.metadata.csv")
write.csv(metadata, save.path, row.names = T)
