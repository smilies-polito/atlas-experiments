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
	seed=52
	os.chdir("/Users/lrcq/Documents/devtraj/preprocessing/brainChromatinGreenLeaf") #set working dir
	data_path=os.path.join(os.getcwd(), "dc1r3_r1") #add folder where data is stored
	metadata_path = os.path.join(os.getcwd(), "multiome_cell_metadata.txt") # add path to multiome_cell_metadata.txt file
	cluster_path = os.path.join(os.getcwd(), "multiome_cluster_names.txt") # add path to multiome_cluster_names.txt file
    
	data = mu.read_10x_h5(os.path.join(data_path, "filtered_feature_bc_matrix.h5"))
	data["rna"].var_names_make_unique()

	preprocess_names(data, "hft_ctx_w21_dc1r3_r1") #add experiment name either hft_ctx_w21_dc1r3_r1, hft_ctx_w21_dc2r2_r1, hft_ctx_w21_dc2r2_r2 
	data.update_obs()

	# Subset for cells in metadata
	metadata = pd.read_csv(metadata_path, sep="\t", header=0)
	metadata = metadata[metadata["Cell.ID"].str.startswith("hft_ctx_w21_dc1r3_r1")]
	mu.pp.filter_obs(data, var=metadata["Cell.ID"])

	# Add cluster names and clusters 
	clusters = pd.read_csv(cluster_path, sep="\t", header=0)
	rna_cls = pd.merge(metadata[["Cell.ID", "seurat_clusters"]], clusters[clusters["Assay"]=="Multiome RNA"], how="left", left_on="seurat_clusters", right_on="Cluster.ID")
	rna_cls = rna_cls.set_index("Cell.ID")
	atac_cls = pd.merge(metadata[["Cell.ID", "ATAC_cluster"]], clusters[clusters["Assay"]=="Multiome ATAC"], how="left", left_on = "ATAC_cluster", right_on = "Cluster.ID")
	atac_cls = atac_cls.set_index("Cell.ID")
	data["rna"].obs = pd.merge(data["rna"].obs, rna_cls["Cluster.Name"], left_index=True, right_index = True, how="left")
	data["atac"].obs= pd.merge(data["atac"].obs, atac_cls["Cluster.Name"], left_index=True, right_index= True, how="left")
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
	ac.pp.tfidf(data["atac"])
	ac.tl.lsi(data["atac"])
	# remove first lsi component since highly correlated with sequencing depth 
	data["atac"].obsm["X_lsi"]=data["atac"].obsm["X_lsi"][:,1:]
	data["atac"].varm["LSI"]= data["atac"].varm["LSI"][:, 1:]
	data["atac"].uns["lsi"]["stdev"] = data["atac"].uns["lsi"]["stdev"][1:]

	# 5. compute nearest neighbor graph + umap
	k = 30
	pca = 30
	lsi = 10
	sc.pp.neighbors(data["rna"], n_neighbors=k, n_pcs = pca, use_rep="X_pca", random_state=seed)
	sc.pp.neighbors(data["atac"], n_neighbors=k, n_pcs=lsi, use_rep="X_lsi", random_state=seed)
	mu.pp.neighbors(data, n_neighbors=30, key_added="wnn")
	mu.tl.umap(data, neighbors_key = "wnn", random_state=seed)
	mu.pl.embedding(data, basis = "X_umap", color=["rna:Cluster.Name", "atac:Cluster.Name"], save="multimodal.png")

	print(data["rna"].var['highly_variable'].head(5))

# 7. plot expression of genes of interest
	# Read and subset spliced and unspliced counts 
	spliced_path = os.path.join(os.getcwd(), "multiome_spliced_rna_counts.tsv")
	#unspliced_path = os.path.join(os.getcwd(), "multiome_unspliced_rna_counts.tsv")
	spliced_counts = pd.read_csv(spliced_path, header=0, sep="\t")
	#unspliced_counts = pd.read_csv(unspliced_path, header=0, sep="\t")
	valid_columns = (spliced_counts.columns.str.startswith("hft_ctx_w21_dc1r3_r1")) | (spliced_counts.columns=="gene")	
	spliced_counts = spliced_counts.loc[:, valid_columns]

	
	

# 9. velocyto 

