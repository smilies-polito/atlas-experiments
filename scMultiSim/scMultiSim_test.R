#########################################################################
# Simulate scATAC-seq and scRNA-seq data using scMultiSim + grid search #
#########################################################################

#if(!requireNamespace("BiocManager", quietly=TRUE))
#  install.packages("BiocManager")
#BiocManager::install("scMultiSim")


library("scMultiSim")
library(dplyr)

seed <- 0
set.seed(seed)

# define utility function to modify a list
list_modify <- function(curr_list, ...){
  args <- list(...)
  for (i in names(args)){ curr_list[[i]] <- args[[i]] }
  curr_list
}

# FUNCTION THAT COMPUTED GENE ACTIVITY 
gene_activity <- function(results, gene.idx){
  peaks <- results$region_to_gene[, gene.idx]
  gene.activity <- colSums(results$atacseq_data * peaks)
  return(gene.activity)
}


# SIMULATIONS AND RESULTS
simulate <- function(seed, GRN_params, tree, saving.folder, num.cells = 1000, num.cifs = 50, 
                     cif.sigma = 0.5, diff.cif.fraction=0.8, do.velocity = TRUE){

  file.specifics = paste(diff.cif.fraction, cif.sigma, sep="_")
  options <- list(rand.seed = seed, GRN = GRN_params, num.cells = num.cells, num.cifs= num.cifs, 
                  cif.sigma= cif.sigma, tree= tree, diff.cif.fraction = diff.cif.fraction, do.velocity=do.velocity)
  results <- sim_true_counts(options)
  results$pseudotime <- (results$cell_time - min(results$cell_time))/(max(results$cell_time) - min(results$cell_time))

  rownames(results$atacseq_data) <- paste0("peak_", 1:dim(results$atac_counts)[1])
  colnames(results$atacseq_data) <- results$cell_meta$cell_id
  rownames(results$counts) <- paste0("gene_", 1:dim(results$counts)[1])
  colnames(results$counts) <- results$cell_meta$cell_id
  rownames(results$unspliced_counts) <- paste0("gene_", 1:dim(results$unspliced_counts)[1])
  colnames(results$unspliced_counts) <- results$cell_meta$cell_id

  write.table(results$atacseq_data, 
              file = file.path(saving.folder, paste(file.specifics, "atac.tsv", sep="_")),
              sep = "\t", row.names = TRUE, col.names = TRUE, quote = FALSE) 
  write.table(results$counts, 
              file = file.path(saving.folder, paste(file.specifics, "spliced.tsv", sep="_")),
              sep = "\t", row.names = TRUE, col.names = TRUE, quote = FALSE)
  write.table(results$unspliced_counts, 
              file = file.path(saving.folder, paste(file.specifics, "unspliced.tsv", sep="_")),
              sep = "\t", row.names = TRUE, col.names = TRUE, quote = FALSE)

  metadata <- results$cell_meta
  metadata$pseudotime <- results$pseudotime
  write.table(metadata, 
              file = file.path(saving.folder, paste(file.specifics, "metadata.tsv", sep="_")),
              sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)

  activity <- lapply(1:results$num_genes, function(idx) gene_activity(results, idx))  %>%
                        do.call(cbind, .) %>% as.data.frame() %>%
                       setNames(paste0("gene_", 1:results$num_genes))
  write.table(activity, 
              file = file.path(saving.folder, paste(file.specifics, "activity.tsv", sep="_")),
              sep = "\t", row.names = TRUE, col.names = TRUE, quote = FALSE) 
}


# MAIN TEXT
data(GRN_params_100)
GRN_params <- GRN_params_100
cif.sigma.list <- c(0.1, 0.3, 0.5, 0.7, 0.9)
diff.cif.fraction <- c(0.1, 0.3, 0.5, 0.7, 0.9)
param.grid <- expand.grid(cif.sigma = cif.sigma.list, diff.cif = diff.cif.fraction)
saving.folder <- "/Users/lrcq/Documents/devtraj/scvemo/scMultiSim/simulation"
num.cells <- 1000
num.cifs <- 50
do.velocity <- TRUE

lapply(1: nrow(param.grid), function(i){
  simulate(seed = seed, GRN_params <- GRN_params, tree <- Phyla3(), saving.folder <- saving.folder,
           num.cells = num.cells, num.cifs=num.cifs, cif.sigma = param.grid$cif.sigma[i], 
           diff.cif.fraction = param.grid$diff.cif[i], do.velocity = do.velocity)
})
