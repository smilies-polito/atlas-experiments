#######################
# Embryonic Mouse Brain Preprocessing 
# ATAC data from https://www.10xgenomics.com/datasets/fresh-embryonic-e-18-mouse-brain-5-k-1-standard-1-0-0
# RNA and velocities from Multivelo github page: https://github.com/welch-lab/MultiVelo/blob/main/Examples/velocyto/10X_multiome_mouse_brain.loom
# cell_annotations.tsv from Multivelo github page: https://github.com/welch-lab/MultiVelo/blob/main/Examples/cell_annotations.tsv
####################### 

import os
import scvi
import anndata
import muon as mu
import numpy as np
import scvelo as scv
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from muon import atac as ac
from muon import MuData


def create_feature_map(rna, strand:bool = False):
	features = rna.var.copy()
	columns = ["Chromosome", "Start", "End"]
	if strand:
		columns.append("Strand")
	features = features[columns]
	features = features.loc[~features.Start.isnull()]
	features.Start = features.Start.astype(int)
	features.End = features.End.astype(int) 
	features.Chromosome = ["chr" + chrom if chrom!='JH584304.1' else chrom for chrom in features.Chromosome]
	return features
			

if __name__ == "__main__":
	seed=52

	data_path = ...
	rna = sc.read_loom(os.path.join(data_path, "multivelo.loom"))
	rna.obs_names = [cell.split(":")[1][:-1] + "-1" for cell in rna.obs_names]
	rna.var_names_make_unique()

	sc.pp.filter_cells(rna, min_counts=1000)
	sc.pp.filter_cells(rna, max_counts=20000)

	scv.pp.filter_and_normalize(rna, min_shared_counts=10, n_top_genes=2000)
	
	atac = sc.read_10x_mtx(os.path.join(data_path, "filtered_feature_bc_matrix"), var_names = "gene_symbols", gex_only = False)
	atac = atac[:, atac.var["feature_types"]=="Peaks"]
	sc.pp.filter_cells(atac, min_counts = 2000)
	sc.pp.filter_cells(atac, max_counts = 60000) 
	
	# find common barcodes
	intersection = set(rna.obs_names).intersection(set(atac.obs_names))
	atac = atac[atac.obs_names.isin(intersection), :]
	rna = rna[rna.obs_names.isin(intersection), :]

	mu.atac.tl.locate_file(data=atac, file=os.path.join(data_path, "fragments.tsv.gz"), key="fragments")
	features = create_feature_map(rna, strand=True)
	activity = mu.atac.tl.count_fragments_features(data = atac, features = features, stranded=True)

	# Add celltypes
	annotations = pd.read_csv(os.path.join(data_path, "cell_annotations.tsv"), sep="\t", header=0, index_col=0)
	union = pd.merge(rna.obs, annotations, how = "left", left_index=True, right_index=True)
	non_developmental_celltype = ["Interneurons1", "Interneurons2", "Interneurons3", "Cajal-Retzius", "Microglia"]
	rna.obs = union

	data = mu.MuData({"rna":rna, "activity":activity})
	# Subset for non developmental cell types 
	data.obs["is_developmental"] = ~data.obs["rna:celltype"].isin(non_developmental_celltype)
	mu.pp.filter_obs(data, "is_developmental")

	sc.pp.normalize_total(data["activity"])

	# PCA 
	n_pcs = 30
	sc.pp.pca(data["rna"], random_state = seed)
	sc.pp.pca(data["activity"], random_state = seed)
	# print plot variance ratio
#	sc.pl.pca_variance_ratio(data["rna"])
#	sc.pl.pca_variance_ratio(data["activity"])
	
	# NEIGHBORS
	n_pcs_rna = 30
	knn_rna = 30
	n_pcs_activity = 30
	knn_activity = 30
	
	sc.pp.neighbors(data["rna"], n_neighbors = knn_rna, n_pcs = n_pcs_rna, random_state = seed)  
	sc.pp.neighbors(data["activity"], n_neighbors = knn_activity, n_pcs= n_pcs_activity, random_state = seed)
	
	# Used for differential expreession analysis and gene annotation later
	# sc.tl.umap(data["rna"], random_state = seed)
	# sc.tl.leiden(data["rna"])
	# print("RNA embedding alone")
	# sc.pl.embedding(data["rna"], basis="X_umap", color="leiden")

	# sc.tl.umap(data["activity"],  random_state = seed)
	# sc.tl.leiden(data["activity"])
	# print("Activity embedding alone")
	# sc.pl.embedding(data["activity"], basis="X_umap", color="leiden")

	# Shared Nearest Neighbors
	mu.pp.neighbors(data, key_added="wnn", n_neighbors=30)
	mu.tl.umap(data, neighbors_key="wnn", random_state = seed)
	mu.tl.leiden(data, random_state = seed)
	mu.tl.louvain(data, random_state = seed)
	
	mu.pl.embedding(data, basis="X_umap", color=["leiden", "louvain", "rna:celltype"])
	
	#sc.tl.rank_genes_groups(data["rna"], 'leiden', method='t-test')
	#result = data["rna"].uns['rank_genes_groups']
	#groups = result['names'].dtype.names
	#genes = pd.DataFrame({group + '_' + key[:1]: result[key][group]for group in groups for key in ['names', 'pvals']}).head(10)
	#genes.to_csv("geni.csv")

	data.write(os.path.join(data_path, "data.h5mu"))	

	# MOFA
	mu.tl.mofa(data, n_factors=10, outfile=os.path.join(data_path, "mofa.hdf5"), gpu_mode=True)
	sc.pp.neighbors(data, use_rep="X_mofa")
	sc.tl.umap(data, random_state=seed)	
	sc.tl.leiden(data)
	mu.pl.embedding(data, basis="X_umap", color=["rna:celltype", "leiden"])
	
