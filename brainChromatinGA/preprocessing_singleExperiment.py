#######################
# Brain Chromatin Preprocessing
# Data from https://github.com/GreenleafLab/brainchromatin/blob/main/links.txt
####################### 

import os
import anndata
import muon as mu
import numpy as np
import scvelo as scv
import pandas as pd
import scanpy as sc
from muon import atac as ac
from muon import MuData

def preprocess_names(modality, experiment_string):
	modality.obs_names = modality.obs_names.str.replace("-1","",regex=False)
	modality.obs_names = experiment_string + "_" + modality.obs_names
	modality.obs["sample"] = experiment_string

def preprocess_count_matrix(counts, experiment_string):
	valid_columns = (counts.columns.str.startswith(experiment_string)) | (counts.columns=="gene")
	counts = counts.loc[:, valid_columns]
	columns =list(counts.columns[1:].str.replace(f"{experiment_string}_", "", regex=False) + "-1")
	counts.columns = ["gene"] + columns
	counts = counts.drop_duplicates(subset="gene")
	return counts

if __name__ == "__main__":
	seed=52

	os.chdir("/Users/lrcq/Documents/devtraj/preprocessing/brainChromatinGreenLeaf")
	experiment_folder = "dc2r2_r2" #add experiment
	data_path=os.path.join(os.getcwd(), experiment_folder) #add folder where data is stored
	metadata_path = os.path.join(os.getcwd(), "multiome_cell_metadata.txt") # add path to multiome_cell_metadata.txt file
	cluster_path = os.path.join(os.getcwd(), "multiome_cluster_names.txt") # add path to multiome_cluster_names.txt file
	spliced_path = os.path.join(os.getcwd(), "multiome_spliced_rna_counts.tsv")
	unspliced_path = os.path.join(os.getcwd(), "multiome_unspliced_rna_counts.tsv")

	experiment = "hft_ctx_w21_" + experiment_folder

	data = mu.read_10x_h5(os.path.join(data_path, "filtered_feature_bc_matrix.h5"))
		
	# Read spliced and unspliced counts + subset for the cell
	spliced_counts = pd.read_csv(spliced_path, header=0, sep="\t")
	unspliced_counts = pd.read_csv(unspliced_path, header=0, sep="\t")
	spliced_counts = preprocess_count_matrix(spliced_counts, experiment)
	unspliced_counts = preprocess_count_matrix(unspliced_counts, experiment)
			
	data["rna"].var_names_make_unique()
	data.update_var()

	# Subset for common genes 
	intersection = set(data["rna"].var_names).intersection(set(spliced_counts["gene"]))
	spliced_counts = spliced_counts[spliced_counts.gene.isin(intersection)]
	mu.pp.filter_var(data["rna"], var=intersection)
	data.update_var()

	# Subset for cells in metadata
	metadata = pd.read_csv(metadata_path, sep="\t", header=0)
	metadata = metadata[metadata["Cell.ID"].str.startswith(experiment)]
	metadata["Cell.ID"] = metadata["Cell.ID"].str.replace(f"{experiment}_", "", regex=False)
	metadata["Cell.ID"] = metadata["Cell.ID"] + "-1"
	
	mu.pp.filter_obs(data, var=metadata["Cell.ID"])

	data["rna"].layers["spliced"] = spliced_counts.iloc[:, 1:].T
	data["rna"].layers["unspliced"] = unspliced_counts.iloc[:, 1:].T
	
	# Add cluster names and clusters 
	clusters = pd.read_csv(cluster_path, sep="\t", header=0)
	rna_cls = pd.merge(metadata[["Cell.ID", "ATAC_cluster"]], clusters[clusters["Assay"]=="Multiome ATAC"], how="left", left_on="ATAC_cluster", right_on="Cluster.ID")
	rna_cls = rna_cls.set_index("Cell.ID")
	data["rna"].obs = pd.merge(data["rna"].obs, rna_cls["Cluster.Name"], left_index=True, right_index = True, how="left")
	data.update()
	mu.pp.intersect_obs(data)

	# Remove non developmental lineages
	atac_non_developmental = ["IN1", "IN2", "IN3", "IN4", "MG/EC/Peric."]
	mu.pp.filter_obs(data, var=~data.obs["rna:Cluster.Name"].isin(atac_non_developmental))
	
	# Gene Activity
	# Get features for aggregation in a format compatible with muon 
	features = pd.DataFrame([s.replace(":", "-", 1).split("-") for s in data["rna"].var.interval])
	features.columns = ["Chromosome", "Start", "End"]
	features["gene_id"] = data["rna"].var.gene_ids.values
	features["gene_name"] = data["rna"].var.index.values
	features.index = data["rna"].var.index
	features = features.loc[~features.Start.isnull()]
	features.Start = features.Start.astype(int)
	features.End = features.End.astype(int) 	
	features = features[features.Chromosome!="KI270726.1"] #remove since not found among peaks 
	gene_activity = mu.atac.tl.count_fragments_features(data, features)
	gene_activity.X = gene_activity.X.astype(float)	


	# RNA preprocessing
	sc.pp.normalize_per_cell(data["rna"])
	sc.pp.log1p(data["rna"])
	sc.pp.highly_variable_genes(data["rna"], n_top_genes=2000, subset=True)
	data.update()
	sc.pp.scale(data["rna"])
	sc.tl.pca(data["rna"]) # 50 components
	sc.pl.pca_variance_ratio(data["rna"], n_pcs=50, save="RNA.png")
	
	# Gene Activity preprocessing
	sc.pp.normalize_per_cell(gene_activity)
	mu.pp.filter_var(gene_activity, var=data["rna"].var_names)
	sc.tl.pca(gene_activity) 
	sc.pl.pca_variance_ratio(gene_activity, n_pcs=50, save=f"{experiment}_gene_activity.png")
	
	sc.pp.neighbors(data["rna"], n_neighbors=30, n_pcs=30, use_rep="X_pca")
	sc.pp.neighbors(gene_activity, n_neighbors=30, n_pcs= 30, use_rep="X_pca")
	
	data = MuData({
		'rna': data.mod["rna"].copy(),
		'activity': gene_activity.copy()
	})
	mu.pp.neighbors(data, n_neighbors=30, random_state=seed, key_added = "wnn")
	mu.tl.umap(data, neighbors_key = "wnn", random_state=seed)
	mu.pl.embedding(data, basis = "X_umap", color=["rna:Cluster.Name"], save=f"{experiment}_multimodal.png")

	# VelocitieATAC_clusters
	scv.pp.moments(data["rna"])	
	scv.tl.recover_dynamics(data["rna"], n_jobs=-1)
	scv.tl.velocity(data["rna"], mode="dynamical")
	scv.tl.velocity_graph(data["rna"])
	scv.tl.velocity_pseudotime(data["rna"])
	mu.pp.intersect_obs(data)
	
	mu.pl.embedding(data, color='rna:velocity_pseudotime', basis="X_umap", color_map='gnuplot', size=40, save=f"{experiment}_pseudotime.png")	

	data.write_h5mu(os.path.join(data_path, "data.h5mu"))
