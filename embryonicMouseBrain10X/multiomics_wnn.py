#######################
# Embryonic Mouse Brain Preprocessing 
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
	seed=42
	np.random.seed(seed)
	data_path = #INSERT PATH 
	rna = sc.read_loom(os.path.join(data_path, "multivelo.loom"))
	rna.obs_names = [cell.split(":")[1][:-1] + "-1" for cell in rna.obs_names]
	rna.var_names_make_unique()

	features = create_feature_map(rna, strand=True)

	sc.pp.calculate_qc_metrics(rna, inplace=True)
	sc.pp.filter_cells(rna, min_counts=1000)
	sc.pp.filter_cells(rna, max_counts=10000)
	scv.pp.filter_and_normalize(rna, min_shared_counts=10, n_top_genes=2000)

	atac = sc.read_10x_mtx(os.path.join(data_path, "filtered_feature_bc_matrix"), var_names = "gene_symbols", gex_only = False)
	atac = atac[:, atac.var["feature_types"]=="Peaks"]

	mu.atac.tl.locate_file(data=atac, file=os.path.join(data_path, "fragments.tsv.gz"), key="fragments")
	
	sc.pp.calculate_qc_metrics(atac, inplace=True)
	ac.tl.nucleosome_signal(atac)
	ac.tl.tss_enrichment(atac, features)
	
	low_tss, high_tss = np.percentile(atac.obs["tss_score"], 5), np.percentile(atac.obs["tss_score"], 95)
	low_nucleo, high_nucleo = np.percentile(atac.obs["nucleosome_signal"], 5), np.percentile(atac.obs["nucleosome_signal"], 95)
	low_atac, high_atac = np.percentile(atac.obs["total_counts"], 5), np.percentile(atac.obs["total_counts"], 95)

	sc.pp.filter_cells(atac, min_counts = low_atac)
	sc.pp.filter_cells(atac, max_counts = high_atac) 
	conditions = (atac.obs.nucleosome_signal > low_nucleo) & (atac.obs.nucleosome_signal < high_nucleo) & (atac.obs.tss_score > low_tss)
	atac = atac[conditions, :]
	
	intersection = set(rna.obs_names).intersection(set(atac.obs_names))
	atac = atac[atac.obs_names.isin(intersection), :]
	rna = rna[rna.obs_names.isin(intersection), :]

	features = create_feature_map(rna, strand=True) #considering only contribution of 2000 HVGs
	activity = mu.atac.tl.count_fragments_features(data = atac, features = features, stranded=True)

	# Add celltypes
	annotations = pd.read_csv(os.path.join(data_path, "cell_annotations.tsv"), sep="\t", header=0, index_col=0)
	union = pd.merge(rna.obs, annotations, how = "left", left_index=True, right_index=True)
	non_developmental_celltype = ["Interneurons1", "Interneurons2", "Interneurons3", "Cajal-Retzius", "Microglia", "Ependymal cells"]
	rna.obs = union

	# Create multimodal dataset
	data = mu.MuData({"rna":rna, "activity":activity})
	data = data[~data.obs["rna:celltype"].isin(non_developmental_celltype)] # removing non developmental cell_types	
	
	sc.pp.normalize_total(data["activity"])

	# PCA 
	sc.pp.pca(data["rna"], random_state = seed)
	sc.pp.pca(data["activity"], random_state = seed)
	
	# NEIGHBORS
	n_pcs_rna = 20
	knn_rna = 30
	n_pcs_activity = 10
	knn_activity = 30
	
	sc.pp.neighbors(data["rna"], n_neighbors = knn_rna, n_pcs = n_pcs_rna, random_state = seed)  
	sc.pp.neighbors(data["activity"], n_neighbors = knn_activity, n_pcs= n_pcs_activity, random_state = seed)

		
	# Shared Nearest Neighbors
	mu.pp.neighbors(data, key_added="wnn", n_neighbors=30)
	mu.tl.umap(data, neighbors_key="wnn", random_state = seed)
	mu.tl.leiden(data, random_state = seed)
	mu.tl.louvain(data, random_state = seed)
	mu.pl.embedding(data, basis="X_umap", color="rna:celltype")
	
	data.write(os.path.join(data_path, "data.h5mu"))

	
