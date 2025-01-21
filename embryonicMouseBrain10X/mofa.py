########################
# Embryonic Mouse Brain Preprocessing: MOFA
# ATAC data from data from https://www.10xgenomics.com/datasets/fresh-embryonic-e-18-mouse-brain-5-k-1-standard-1-0-0 
# RNA and velocities from Multivelo github page: https://github.com/welch-lab/MultiVelo/blob/main/Examples/velocyto/10X_multiome_mouse_brain.loom
# cell_annotations.tsv from Multivelo github page: https//github.com/welch-lab/MultiVelo/blob/main/Examples/cell_annotations.tsv
#######################

import os
import anndata
import muon as mu
import numpy as np
import scvelo as scv
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from muon import atac as ac
from muon import MuData



if __name__ == "__main__":
	seed = 52
	data_path = "/Users/lrcq/Documents/devtraj/preprocessing/embryonicMouseBrain10X/"
	rna = sc.read_loom(os.path.join(data_path, "multivelo.loom"))
	rna.obs_names = [cell.split(":")[1][:-1] + "-1" for cell in rna.obs_names]
	rna.var_names_make_unique()
	
	sc.pp.filter_cells(rna, min_counts=1000)
	sc.pp.filter_cells(rna, max_counts=20000)
	scv.pp.filter_and_normalize(rna, min_shared_counts=10, n_top_genes=2000)

	atac = sc.read_10x_mtx(os.path.join(data_path, "filtered_feature_bc_matrix"), var_names = "gene_symbols", gex_only= False)
	atac = atac[:, atac.var["feature_types"]=="Peaks"]
	sc.pp.filter_cells(atac, min_counts =1000)
	sc.pp.filter_cells(atac, max_counts=20000)
	ac.pp.tfidf(atac, scale_factor=1e4)
	sc.pp.highly_variable_genes(atac, min_mean=0.05, max_mean=0.1, min_disp=0.5)	
	print(f"Number of highly varaible peaks: {np.sum(atac.var.highly_variable)}")	
	
	# Find common barcodes
	intersection = set(rna.obs_names).intersection(set(atac.obs_names))
	atac = atac[atac.obs_names.isin(intersection), :]
	rna = rna[rna.obs_names.isin(intersection), :]
	
	# Add cell types 
	annotations = pd.read_csv(os.path.join(data_path, "cell_annotations.tsv"), sep="\t", header=0, index_col=0)
	union = pd.merge(rna.obs, annotations, how="left", left_index = True, right_index = True)
	non_developmental_celltype = ["Interneurons1", "Interneurons2", "Interneurons3", "Cajal-Retzius", "Microglia"]	
	rna.obs = union
	
	data = mu.MuData({"rna": rna, "atac":atac})
	
	# Subset for non developmental cell types
	data.obs["is_developmental"] = ~data.obs["rna:celltype"].isin(non_developmental_celltype)
	mu.pp.filter_obs(data, "is_developmental")
	print(f"Number of cells retained = {data.shape[0]}") 
	
	# Perform MOFA
	n_pcs = 30
	sc.pp.pca(data["rna"], random_state = seed)
	ac.tl.lsi(data["atac"])
	# remove first LSI dimension since highly correlated with read sequencing
	data["atac"].obsm["X_lsi"] = data["atac"].obsm["X_lsi"][:, 1:]
	data["atac"].varm["LSI"] = data["atac"].varm["LSI"][:, 1:]
	data["atac"].uns["lsi"]["stdev"] = data["atac"].uns["lsi"]["stdev"][1:]

	# Neighbors
	n_pcs_rna = 30
	n_lsi_atac = 10
	knn_rna = 30
	knn_atac = 10
	sc.pp.neighbors(data["rna"], n_neighbors= knn_rna, n_pcs= n_pcs_rna, random_state=seed)
	sc.pp.neighbors(data["atac"], n_neighbors = knn_atac, n_pcs =n_lsi_atac, use_rep="X_lsi", random_state = seed)

	# MOFA
	mu.tl.mofa(data, n_factors =20, outfile=os.path.join(data_path, "mofa.hdf5"), gpu_mode = True)
	sc.pp.neighbors(data, use_rep="X_mofa", random_state= seed)
	sc.tl.umap(data, random_state = seed)
	sc.tl.leiden(data)
	mu.pl.embedding(data, basis="X_umap", color=["rna:celltype", "leiden"])

	data.write(os.path.join(data_path, "mofa.h5mu"))
