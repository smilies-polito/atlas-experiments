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

def read_multiomics(experiment):
	data = mu.read_10x_h5(os.path.join(os.getcwd(), experiment, "filtered_feature_bc_matrix.h5"))
	data["rna"].var_names_make_unique()
	data.update()
	for mod in data.mod.keys():
		data[mod].obs_names = data[mod].obs_names.str.replace("-1","",regex=False)
		data[mod].obs_names = "hft_ctx_w21_" + experiment + "_" + data[mod].obs_names
		data[mod].obs["sample"] = experiment
	data.update_obs()
	return data

def subset_cells(data, metadata, experiment):
	subset_metadata = metadata[metadata["Cell.ID"].str.startswith("hft_ctx_w21_" + experiment)]
	mu.pp.filter_obs(data, var=subset_metadata["Cell.ID"])
	
def add_clusters(data, rna_cls, atac_cls):
	data["rna"].obs = pd.merge(data["rna"].obs, rna_cls["Cluster.Name"], left_index = True, right_index=True, how="left")
	data["atac"].obs = pd.merge(data["atac"].obs, atac_cls["Cluster.Name"], left_index=True, right_index=True, how="left")
	data.update()
	mu.pp.intersect_obs(data)
	

if __name__ == "__main__":
	seed=52

	os.chdir("...")
	dc1r3_r1 = read_multiomics("dc1r3_r1")
	dc2r2_r1 = read_multiomics("dc2r2_r1")
	dc2r2_r2 = read_multiomics("dc2r2_r2")

	metadata_path = os.path.join(os.getcwd(), "multiome_cell_metadata.txt") # add path to multiome_cell_metadata.txt file
	cluster_path = os.path.join(os.getcwd(), "multiome_cluster_names.txt") # add path to multiome_cluster_names.txt file

	# Subset for cells in metadata
	metadata = pd.read_csv(metadata_path, sep="\t", header=0)
	subset_cells(dc1r3_r1, metadata, "dc1r3_r1")
	subset_cells(dc2r2_r1, metadata, "dc2r2_r1")
	subset_cells(dc2r2_r2, metadata, "dc2r2_r2")

	
	# Add cluster names and clusters 
	clusters = pd.read_csv(cluster_path, sep="\t", header=0)
	rna_cls = pd.merge(metadata[["Cell.ID", "seurat_clusters"]], clusters[clusters["Assay"]=="Multiome RNA"], how="left", left_on="seurat_clusters", right_on="Cluster.ID")
	rna_cls = rna_cls.set_index("Cell.ID")
	atac_cls = pd.merge(metadata[["Cell.ID", "ATAC_cluster"]], clusters[clusters["Assay"]=="Multiome ATAC"], how="left", left_on = "ATAC_cluster", right_on = "Cluster.ID")
	atac_cls = atac_cls.set_index("Cell.ID")
	
	add_clusters(dc1r3_r1, rna_cls, atac_cls)
	add_clusters(dc2r2_r1, rna_cls, atac_cls)
	add_clusters(dc2r2_r2, rna_cls, atac_cls)
	
	# merge atac into single AnnData
	atac = sc.concat([dc1r3_r1["atac"], dc2r2_r1["atac"], dc2r2_r2["atac"]], join='outer', label="experiment")

	# ATAC preprocessing
	ac.pp.tfidf(atac)
	ac.tl.lsi(atac)
	atac.obsm["X_lsi"]=atac.obsm["X_lsi"][:,1:]
	atac.varm["LSI"]= atac.varm["LSI"][:, 1:]
	atac.uns["lsi"]["stdev"] = atac.uns["lsi"]["stdev"][1:]

	# Compute nearest neighbor graph + umap
	k = 30
	lsi = 10
	sc.pp.neighbors(atac, n_neighbors=k, n_pcs=lsi, use_rep="X_lsi", random_state=seed)
	sc.tl.umap(atac, random_state=seed)
	sc.pl.embedding(atac, basis="umap", color="sample")

	sc.external.pp.harmony_integrate(atac, key="sample", basis ="X_lsi", adjusted_basis="X_lsi_harmony")
	sc.pl.embedding(atac, basis = "X_lsi_harmony", color="sample")
