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

# Minimal input: differentiation tree as a R pyhlo object and controls cell population 
# structure. Each node represents a cell type and connected nodes represent differentiation
# replationship between cell types. 
# Già previsti da scMultiSim ci sono i seguenti
Phyla5(plotting=TRUE) # 5 stati terminali e 4 nodi interni 
Phyla3(plotting=TRUE) # 3 stati terminali e 2 nodi interni 
Phyla1() # 1 branch che connette 2 nodi 
# Altrimenti bisogna creare un oggetto vero e proprio con ape::read.tree() o ape::rtree() 


# Optional input 1/2: GRN. Dataframe con 3 colonne <target> <regulator> <effect>. <target> e <regulator>
# sono nomi di geni, mentre <effect> numero che indica l'azione del regulator sul target. 
# \Qui usiamo un esempio di GRN sample con 1139 geni 
data(GRN_params_100)
GRN_params <- GRN_params_100
head(GRN_params)

# Optional input 2/2: cell-cell interactions, soprattutto per simulare spatial data


# ESEMPIO: simulare 1000 cells con 50 CIFs 
# num.genes parametro che indica quanti geni considerare. Se non settato è numero di geni differenti nel GRN
# mentre noi qui usiamo 100 geni con 10% geni in più che non sono regolati in alcun modo. 

# cif.sigma controlla la variance di CIF e più piccolo è cif.sigma più la traiettoria dei dati è definita. 
# diff.cif.fraction indica quanto diff-CIF impattano su creazione dei dati rispetto a non-diff-CIF

# do.velocity simula RNA velocity data
options <- list(rand.seed = 0, GRN = GRN_params, num.cells = 1000, num.cifs=50, cif.sigma=0.5, tree=Phyla5()
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

