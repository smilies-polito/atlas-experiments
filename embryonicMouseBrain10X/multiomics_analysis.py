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
	data_path = # INSERT PATH  
	rna = sc.read_loom(os.path.join(data_path, "multivelo.loom"))
	rna.obs_names = [cell.split(":")[1][:-1] + "-1" for cell in rna.obs_names]
	rna.var_names_make_unique()

	features = create_feature_map(rna, strand=True)

	# Preprocessing QC metrics: rna + filtering	
	print("QC RNA")
	sc.pp.calculate_qc_metrics(rna, inplace=True)
	sc.pl.violin(rna, ["n_genes_by_counts", "total_counts"])
	sc.pp.filter_cells(rna, min_counts=1000)
	sc.pp.filter_cells(rna, max_counts=10000)
	scv.pp.filter_and_normalize(rna, min_shared_counts=10, n_top_genes=2000)

	# Read ATAC	
	atac = sc.read_10x_mtx(os.path.join(data_path, "filtered_feature_bc_matrix"), var_names = "gene_symbols", gex_only = False)
	atac = atac[:, atac.var["feature_types"]=="Peaks"]

	mu.atac.tl.locate_file(data=atac, file=os.path.join(data_path, "fragments.tsv.gz"), key="fragments")
	
	# Preprocessing QC metrics: atac + filtering
	print("QC ATAC")
	sc.pp.calculate_qc_metrics(atac, inplace=True)
	ac.tl.nucleosome_signal(atac)
	ac.tl.tss_enrichment(atac, features)
	sc.pl.violin(atac, "nucleosome_signal")
	sc.pl.violin(atac, "tss_score")
	sc.pl.violin(atac, "total_counts")
	
	low_tss, high_tss = np.percentile(atac.obs["tss_score"], 5), np.percentile(atac.obs["tss_score"], 95)
	low_nucleo, high_nucleo = np.percentile(atac.obs["nucleosome_signal"], 5), np.percentile(atac.obs["nucleosome_signal"], 95)
	low_atac, high_atac = np.percentile(atac.obs["total_counts"], 5), np.percentile(atac.obs["total_counts"], 95)
	print(low_tss, high_tss, low_nucleo, high_nucleo, low_atac, high_atac)

	sc.pp.filter_cells(atac, min_counts = low_atac)
	sc.pp.filter_cells(atac, max_counts = high_atac) 
	conditions = (atac.obs.nucleosome_signal > low_nucleo) & (atac.obs.nucleosome_signal < high_nucleo) & (atac.obs.tss_score > low_tss)
	atac = atac[conditions, :]
	
	# Find common barcodes
	print("INTERSECT FOR COMMON BARCODES AFTER QC")
	intersection = set(rna.obs_names).intersection(set(atac.obs_names))
	atac = atac[atac.obs_names.isin(intersection), :]
	rna = rna[rna.obs_names.isin(intersection), :]

	print("COMPUTING ACTIVITY FOR 2000 VAR GENES")
	features = create_feature_map(rna, strand=True) #considering only contribution of 2000 HVGs
	activity = mu.atac.tl.count_fragments_features(data = atac, features = features, stranded=True)

	# Add celltypes
	print("ADD CELL TYPES")
	annotations = pd.read_csv(os.path.join(data_path, "cell_annotations.tsv"), sep="\t", header=0, index_col=0)
	union = pd.merge(rna.obs, annotations, how = "left", left_index=True, right_index=True)
	union_activity= pd.merge(activity.obs, annotations, how="left", left_index=True, right_index=True)
	non_developmental_celltype = ["Interneurons1", "Interneurons2", "Interneurons3", "Cajal-Retzius", "Microglia", "Ependymal cells"]
	rna.obs = union
	activity.obs = union_activity

	# Create multimodal dataset
	print("CREATE MULTIMODAL MUON.MUDATA")
	data = mu.MuData({"rna":rna, "activity":activity})
	data = data[~data.obs["rna:celltype"].isin(non_developmental_celltype)] # removing non developmental cell_types
	print("NORMALIZING ACTIVITY")
	sc.pp.normalize_total(data["activity"])

	# PCA 
	print("PCA")
	sc.pp.pca(data["rna"], random_state = seed)
	sc.pp.pca(data["activity"], random_state = seed)
	# print plot variance ratio
	sc.pl.pca_variance_ratio(data["rna"])
	sc.pl.pca_variance_ratio(data["activity"])
	
	# NEIGHBORS
	print("SINGLE MODALITY NEIGHBORS")
	n_pcs_rna = 20
	knn_rna = 30
	n_pcs_activity = 10
	knn_activity = 30
	
	sc.pp.neighbors(data["rna"], n_neighbors = knn_rna, n_pcs = n_pcs_rna, random_state = seed)  
	sc.pp.neighbors(data["activity"], n_neighbors = knn_activity, n_pcs= n_pcs_activity, random_state = seed)
	
	# Single modality leiden and umap
	print("SINGLE MODALITIES UMAP + LEIDEN CLUSTERING")
	sc.tl.umap(data["rna"], random_state = seed)
	sc.tl.leiden(data["rna"])
	sc.pl.embedding(data["rna"], basis="X_umap", color=["celltype","leiden"])

	sc.tl.umap(data["activity"],  random_state = seed)
	sc.tl.leiden(data["activity"])
	sc.pl.embedding(data["activity"], basis="X_umap", color=["celltype", "leiden"])

	# Shared Nearest Neighbors
	print("MULTIMODAL NEIGHBORS + UMAP + LEIDEN CLUSTERING")
	mu.pp.neighbors(data, key_added="wnn", n_neighbors=30)
	mu.tl.umap(data, neighbors_key="wnn", random_state = seed)
	mu.tl.leiden(data, random_state = seed)
	mu.tl.louvain(data, random_state = seed)
	mu.pl.embedding(data, basis="X_umap", color="rna:celltype")
	print("MODALITY WEIGHTS")
	mu.pl.embedding(data, basis="X_umap", color=["rna:mod_weight", "activity:mod_weight"])
	
	print("IDENTIFY CELLS FOR WHICH ACTIVITY_MOD_WEIGHTS > RNA_MOD_WEIGHTS (OUTLIERS IN UMAP PLOT)")
	data.obs["high:activityW_low:rnaW"] = (data.obs["rna:mod_weight"] - data.obs["activity:mod_weight"]) < 0
	print("Number of outliers {}".format(np.sum(data.obs["high:activityW_low:rnaW"])))
	mu.pl.embedding(data, basis="X_umap", color="high:activityW_low:rnaW")
	print("SAVING OUTLIERS INTO CSV FILE")
	outliers = pd.Series(data.obs[data.obs["high:activityW_low:rnaW"]==True].index)
	outliers.to_csv(os.path.join(data_path, "outliers.csv"))
	
	print("PCA_i VS PCA_j FOR RNA AND ACTIVITY WITH OUTLIERS IDENTIFICATION")
	data["activity"].obs = pd.merge(data["activity"].obs, data.obs["high:activityW_low:rnaW"], how="left", left_index=True, right_index=True)
	data["rna"].obs = pd.merge(data["rna"].obs, data.obs["high:activityW_low:rnaW"], how="left", left_index=True, right_index=True)
	sc.pl.pca(data["rna"], color=["celltype", "high:activityW_low:rnaW"], components=["1,2", "2,3", "3,4"])
	sc.pl.pca(data["activity"], color=["celltype", "high:activityW_low:rnaW"], components=["1,2", "2,3", "3,4"])


	print("GEX TOTAL FOR EACH CELL (LOG-NORMALIZED GEX)")
	data["rna"].obs["gex"] = data["rna"].X.sum(axis=1)
	mu.pl.embedding(data, basis = "X_umap", color=["rna:total_counts"], title="Normal GEX")
	data["activity"].obs["ga"] = data["activity"].X.sum(axis=1)
	data.update()
	mu.pl.embedding(data, basis="X_umap", color=["activity:ga"], title="Normal GA")


	
