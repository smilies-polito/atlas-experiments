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

def read_mudata(path):
	data = mu.read_10x_h5(path)
	data["rna"].var_names_make_unique()
	data.update_var()
	return data

def subset_metadata(metadata, experiment):
	meta = metadata[metadata.index.str.startswith(experiment)]
	meta.index = meta.index.str.replace(f"{experiment}_", "", regex=False)
	meta.index = meta.index + "-1"
	return meta

def filter_cell_metadata(data, metadata, experiment):
	meta = subset_metadata(metadata, experiment)
	mu.pp.filter_obs(data, var=meta.index)
	
def add_clusters(data, clusters, experiment, cluster_key="Cluster.Name", modality="rna"):
	clu = subset_metadata(clusters, experiment)
	data[modality].obs = pd.merge(data[modality].obs, clu[cluster_key], left_index=True, right_index = True, how="left")
	data.update()
	mu.pp.intersect_obs(data)

def add_rna_counts(data, spliced, unspliced, experiment):
	cols = (spliced.columns == "gene") | (spliced.columns.str.startswith(experiment))
	spliced = spliced.loc[:, cols]
	unspliced = unspliced.loc[:, cols]
	cols = ["gene"] + [s.replace(f"{experiment}_", "") + "-1" for s in spliced.columns if s != "gene"]
	spliced.columns = cols
	unspliced.columns = cols
	
	intersection = set(data["rna"].var_names).intersection(set(spliced.gene))
	mu.pp.filter_var(data["rna"], var=intersection)

	spliced = spliced[spliced.gene.isin(intersection)]
	unspliced = unspliced[unspliced.gene.isin(intersection)]
	data["rna"].layers["spliced"] = spliced.iloc[:, 1:].T
	data["rna"].layers["unspliced"] = unspliced.iloc[:, 1:].T
	data.update()

def create_features(data):
	features = pd.DataFrame([s.replace(":", "-", 1).split("-") for s in data.var.interval])
	features.columns = ["Chromosome", "Start", "End"]
	features["gene_id"] = data.var.gene_ids.values
	features["gene_name"] = data.var.index.values	
	features.index = data.var.index
	features = features.loc[~features.Start.isnull()]
	features.Start = features.Start.astype(int)
	features.End = features.End.astype(int)
	return(features)

def compute_gene_activity(data):
	features = create_features(data["rna"])
	gene_activity = mu.atac.tl.count_fragments_features(data, features)
	gene_activity.X = gene_activity.X.astype(float)
	return(gene_activity)	

def update_modality_obs_names(data, experiment):
	data.obs_names = experiment + "_" + data.obs_names.values 

def update_obs_names(data, experiment):
	for mod in data.mod.keys():
		update_modality_obs_names(data[mod], experiment)
	data.update_obs()
	


if __name__ == "__main__":
	seed=52

	os.chdir("/Users/lrcq/Documents/devtraj/preprocessing/brainChromatinGreenLeaf")
	metadata_path = os.path.join(os.getcwd(), "multiome_cell_metadata.txt") # add path to multiome_cell_metadata.txt file
	cluster_path = os.path.join(os.getcwd(), "multiome_cluster_names.txt") # add path to multiome_cluster_names.txt file
	spliced_path = os.path.join(os.getcwd(), "multiome_spliced_rna_counts.tsv")
	unspliced_path = os.path.join(os.getcwd(), "multiome_unspliced_rna_counts.tsv")

	prefix = "hft_ctx_w21_"

	dc1r3_r1 = read_mudata(os.path.join(os.getcwd(), "dc1r3_r1", "filtered_feature_bc_matrix.h5"))
	dc2r2_r1 = read_mudata(os.path.join(os.getcwd(), "dc2r2_r1", "filtered_feature_bc_matrix.h5"))
	dc2r2_r2 = read_mudata(os.path.join(os.getcwd(), "dc2r2_r2", "filtered_feature_bc_matrix.h5"))
		
	# Read spliced and unspliced counts + subset for the cell
	spliced_counts = pd.read_csv(spliced_path, header=0, sep="\t")
	unspliced_counts = pd.read_csv(unspliced_path, header=0, sep="\t")
	spliced_counts = spliced_counts.drop_duplicates(subset="gene") 
	unspliced_counts = unspliced_counts.drop_duplicates(subset="gene")
			
	# Subset for cells in metadata
	metadata = pd.read_csv(metadata_path, sep="\t", header=0)
	metadata = metadata.set_index("Cell.ID")
	filter_cell_metadata(dc1r3_r1, metadata, f"{prefix}dc1r3_r1")
	filter_cell_metadata(dc2r2_r1, metadata, f"{prefix}dc2r2_r1")
	filter_cell_metadata(dc2r2_r2, metadata, f"{prefix}dc2r2_r2")
	
	# Add cluster names and clusters 
	clusters = pd.read_csv(cluster_path, sep="\t", header=0)
	clusters = metadata["ATAC_cluster"].reset_index().merge(clusters[clusters.Assay=="Multiome ATAC"], how="left", left_on = "ATAC_cluster", right_on = "Cluster.ID").set_index("Cell.ID")
	add_clusters(dc1r3_r1, clusters, f"{prefix}dc1r3_r1")	
	add_clusters(dc2r2_r1, clusters, f"{prefix}dc2r2_r1")
	add_clusters(dc2r2_r2, clusters, f"{prefix}dc2r2_r2")

	add_rna_counts(dc1r3_r1, spliced_counts, unspliced_counts, f"{prefix}dc1r3_r1")
	add_rna_counts(dc2r2_r2, spliced_counts, unspliced_counts, f"{prefix}dc2r2_r2")
	add_rna_counts(dc2r2_r1, spliced_counts, unspliced_counts, f"{prefix}dc2r2_r1")

	# Remove non developmental lineages
	atac_non_developmental = ["IN1", "IN2", "IN3", "IN4", "MG/EC/Peric."]
	mu.pp.filter_obs(dc1r3_r1, var=~dc1r3_r1.obs["rna:Cluster.Name"].isin(atac_non_developmental))
	mu.pp.filter_obs(dc2r2_r1, var=~dc2r2_r1.obs["rna:Cluster.Name"].isin(atac_non_developmental))	
	mu.pp.filter_obs(dc2r2_r2, var=~dc2r2_r2.obs["rna:Cluster.Name"].isin(atac_non_developmental))
		
	# Gene Activity
	dc1r3_r1_GA = compute_gene_activity(dc1r3_r1)
	dc2r2_r1_GA = compute_gene_activity(dc2r2_r1)
	dc2r2_r2_GA = compute_gene_activity(dc2r2_r2)	

	# Create unique df
	update_obs_names(dc1r3_r1, f"{prefix}dc1r3_r1")
	update_obs_names(dc2r2_r1, f"{prefix}dc2r2_r1")
	update_obs_names(dc2r2_r2, f"{prefix}dc2r2_r2")
	
	rna = anndata.concat([dc1r3_r1["rna"], dc2r2_r1["rna"], dc2r2_r2["rna"]], label="batch", keys=["dc1r3_r1", "dc2r2_r1", "dc2r2_r2"])
	
	# RNA reprocessing
	sc.pp.normalize_per_cell(rna)
	sc.pp.log1p(rna)
	sc.pp.highly_variable_genes(rna, n_top_genes=2000, subset=True)
	sc.pp.scale(rna)
	sc.tl.pca(rna) # 50 components
	sc.pl.pca_variance_ratio(rna, n_pcs=50, save="RNA.png")
	
	update_modality_obs_names(dc1r3_r1_GA, f"{prefix}dc1r3_r1")
	update_modality_obs_names(dc2r2_r1_GA, f"{prefix}dc2r2_r1")
	update_modality_obs_names(dc2r2_r2_GA, f"{prefix}dc2r2_r2")
	
	activity = anndata.concat([dc1r3_r1_GA, dc2r2_r1_GA, dc2r2_r2_GA], label="batch", keys=["dc1r3_r1", "dc2r2_r1", "dc2r2_r2"])

	# Gene Activity preprocessing
	sc.pp.normalize_per_cell(activity)
	mu.pp.filter_var(activity, var=rna.var_names)
	sc.tl.pca(activity) 
	sc.pl.pca_variance_ratio(activity, n_pcs=50, save=f"_gene_activity.png")
	
	sc.external.pp.bbknn(rna, batch_key="batch", neighbors_within_batch=10, n_pcs=30, use_rep="X_pca")
	sc.tl.umap(rna)
	sc.pl.embedding(rna, basis="X_umap", color="batch", save = "_rna_batch.png")
	
	sc.pp.neighbors(activity, n_neighbors=30, n_pcs= 30, use_rep="X_pca")
	sc.tl.umap(activity)
	sc.pl.embedding(activity, basis="X_umap", color="batch", save="_gene_activity_batch.png")
	
	data = MuData({'rna': rna.copy(), 'activity': activity.copy()})
	mu.pp.neighbors(data, n_neighbors=30, random_state=seed, key_added = "wnn")
	mu.tl.umap(data, neighbors_key = "wnn", random_state=seed)
	mu.pl.embedding(data, basis = "X_umap", color=["rna:Cluster.Name"], save=f"_multimodal.png")

	scv.pp.moments(data["rna"])	
	scv.tl.recover_dynamics(data["rna"], n_jobs=-1)
	scv.tl.velocity(data["rna"], mode="dynamical")
	scv.tl.velocity_graph(data["rna"])
	scv.tl.velocity_pseudotime(data["rna"])
	mu.pp.intersect_obs(data)
	
	mu.pl.embedding(data, color='rna:velocity_pseudotime', basis="X_umap", color_map='gnuplot', size=40, save=f"_pseudotime.png")	

	data.write_h5mu(os.path.join(os.getcwd(), "data.h5mu"))
