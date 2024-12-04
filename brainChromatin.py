#######################
# Brain Chromatin Preprocessing
# Data from https://github.com/GreenleafLab/brainchromatin/blob/main/links.txt
####################### 

import os
import anndata
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
from muon import atac as ac

def preprocess_names(multimodalData, experiment_string):
	for modality in ["rna", "atac"]:
		multimodalData[modality].obs_names = multimodalData[modality].obs_names.str.replace("-1","",regex=False)
		multimodalData[modality].obs_names = experiment_string + "_" + multimodalData[modality].obs_names
		multimodalData[modality].obs["sample"] = experiment_string

if __name__ == "__main__":
	os.chdir() #set working dir
	data_path=os.getcwd() #add frolder where data is stored
	metadata_path = os.path.join(os.getcwd(), "multiome_cell_metadata.txt") # add path to multiome_cell_metadata.txt file
	cluster_path = os.path.join(os.getcwd(), "multiome_cluster_names.txt") # add path to multiome_cluster_names.txt file
    
	data = mu.read_10x_h5(os.path.join(data_path, "filtered_feature_bc_matrix.h5"))
	data["rna"].var_names_make_unique()

	preprocess_names(data, "hft_ctx_w21_dc1r3_r1") #add experiment name either hft_ctx_w21_dc1r3_r1, hft_ctx_w21_dc2r2_r1, hft_ctx_w21_dc2r2_r2 
	data.update_obs()

	# Subset for cells in metadata
	metadata = pd.read_csv(metadata_path, sep="\t", header=0)
	mu.pp.filter_obs(data, var=metadata["Cell.ID"])
	
	# Add cluster names and clusters 
	clusters = pd.read_csv(cluster_path, sep="\t", header=0)
	rna_cls = pd.merge(metadata[["Cell.ID", "seurat_clusters"]], clusters[clusters["Assay"]=="Multiome RNA"], how="left", left_on="seurat_clusters", right_on="Cluster.ID")
	atac_cls = pd.merge(metadata[["Cell.ID", "ATAC_cluster"]], clusters[clusters["Assay"]=="Multiome ATAC"], how="left", left_on = "ATAC_cluster", right_on = "Cluster.ID")
	data["rna"].obs.merge(rna_cls[["Cell.ID","Cluster.Name"]], left_index=True, right_on="Cell.ID", how="left")
	data["atac"].obs.merge(atac_cls[["Cell.ID", "Cluster.Name"]], left_index=True, right_on="Cell.ID", how="left")
	data.update()
	mu.pp.intersect_obs(data)

	# RNA preprocessing
	sc.pp.normalize_per_cell(data["rna"])
	sc.pp.log1p(data["rna"])
	sc.pp.highly_variable_genes(data["rna"], n_top_genes=2000)
	sc.pp.scale(data["rna"])

	sc.tl.pca(data["rna"]) # 50 components
	sc.pl.pca_variance_ratio(data["rna"], n_pcs=50, save="RNA.png")

	# ATAC preprocessing
	#ac.pp.tfidf(data["atac"])
	#ac.tl.lsi(data["atac"])
	# remove first lsi component since highly correlated with sequencing depth 
	#data["atac"].obsm["X_lsi"]=data["atac"].obsm["X_lsi"][:,1:]
	#data["atac"].varm["LSI"]= data["atac"].varm["LSI"][:, 1:]
	#data["atac"].uns["lsi"]["stdev"] = data["atac"].uns["lsi"]["stdev"][1:]

	# 5. compute nearest neighbor graph + umap
	#k = 30
	#pca = 30
	#lsi = 10
	#sc.pp.neighbors(data["rna"], n_neighbors=k, n_pcs = pca, use_rep="X_pca", random_state=seed)
	#sc.pp.neighbors(data["atac"], n_neighbors=k, n_pca=lsi, use_rep="X_lsi", random_state=seed)
	#mu.pp.neighbors(data, n_neighbors=30, key_added="wnn")
	#mu.tl.umap(data, neighbors_key = "wnn", random_state=seed)
	#mu.pl.embedding(data, basis = "X_umap", color=["rna:Cluster.Name", "atac:Cluster.Name"], save="multimodal.png")


# 7. plot expression of genes of interest
# 8. read and subset spliced and unspliced counts
# 9. velocyto 

