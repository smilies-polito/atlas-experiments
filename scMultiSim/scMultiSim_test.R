#######################
# simulate scATAC-seq and scRNA-seq data using scMultiSim
#######################
#if(!requireNamespace("BiocManager", quietly=TRUE))
#  install.packages("BiocManager")
#BiocManager::install("scMultiSim")
set.seed(0)

library("scMultiSim")
library(dplyr)

# define utility function to modify a list
list_modify <- function(curr_list, ...){
  args <- list(...)
  for (i in names(args)){ curr_list[[i]] <- args[[i]] }
  curr_list
}


data(GRN_params_100)
GRN_params <- GRN_params_100


options <- list(rand.seed = 0, GRN = GRN_params, num.cells = 1000, num.cifs=50, cif.sigma=0.5, tree=Phyla3()
                , diff.cif.fraction = 0.8, do.velocity=TRUE)
results <- sim_true_counts(options)
results$pseudotime <- (results$cell_time - min(results$cell_time))/(max(results$cell_time) - min(results$cell_time))

rownames(results$atacseq_data) <- paste0("peak_", 1:dim(results$atac_counts)[1])
colnames(results$atacseq_data) <- results$cell_meta$cell_id
rownames(results$counts) <- paste0("gene_", 1:dim(results$counts)[1])
colnames(results$counts) <- results$cell_meta$cell_id
rownames(results$unspliced_counts) <- paste0("gene_", 1:dim(results$unspliced_counts)[1])
colnames(results$unspliced_counts) <- results$cell_meta$cell_id

write.table(results$atacseq_data, file = "atac.tsv", sep = "\t", row.names = TRUE, col.names = TRUE, quote = FALSE) 
write.table(results$counts, file = "spliced.tsv", sep = "\t", row.names = TRUE, col.names = TRUE, quote = FALSE)
write.table(results$unspliced_counts, file = "unspliced.tsv", sep = "\t", row.names = TRUE, col.names = TRUE, quote = FALSE)

metadata <- results$cell_meta
metadata$pseudotime <- results$pseudotime
write.table(metadata, file = "metadata.tsv", sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)


gene_activity <- function(gene.idx){
  peaks <- results$region_to_gene[, gene.idx]
  gene.activity <- colSums(results$atacseq_data * peaks)
  return(gene.activity)
}

activity <- lapply(1:results$num_genes, gene_activity) %>% do.call(cbind, .) %>% as.data.frame() %>% setNames(paste0("gene_", 1:results$num_genes))
write.table(activity, file = "activity.tsv", sep = "\t", row.names = TRUE, col.names = TRUE, quote = FALSE) 

