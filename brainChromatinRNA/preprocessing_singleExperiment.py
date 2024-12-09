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

def preprocess_names(modality, experiment_string):
	modality.obs_names = modality.obs_names.str.replace("-1","",regex=False)
	modality.obs_names = experiment_string + "_" + modality.obs_names
	modality.obs["sample"] = experiment_string

def preprocess_count_matrix(counts, experiment_string):
	valid_columns = (counts.columns.str.startswith(experiment_string)) | (counts.columns=="gene")
	counts = counts.loc[:, valid_columns]
	counts = counts.drop_duplicates(subset="gene")
	return counts

if __name__ == "__main__":
	seed=52

	os.chdir("...")
	experiment_folder = "" #add experiment
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
	unspliced_counts = unspliced_counts[unspliced_counts.gene.isin(intersection)]
	mu.pp.filter_var(data["rna"], var=intersection)

	# Chamge to choerent cell name representation across files
	preprocess_names(data["rna"], experiment)
	preprocess_names(data["atac"], experiment)
	data.update_obs()

	# Subset for cells in metadata
	metadata = pd.read_csv(metadata_path, sep="\t", header=0)
	metadata = metadata[metadata["Cell.ID"].str.startswith(experiment)]
	mu.pp.filter_obs(data, var=metadata["Cell.ID"])
	
	# Add spliced and unspliced layers
	data["rna"].layers["spliced"] = spliced_counts.iloc[:, 1:].T
	data["rna"].layers["unspliced"] = unspliced_counts.iloc[:, 1:].T

	# Add cluster names and clusters 
	clusters = pd.read_csv(cluster_path, sep="\t", header=0)
	atac_cls = pd.merge(metadata[["Cell.ID", "ATAC_cluster"]], clusters[clusters["Assay"]=="Multiome ATAC"], how="left", left_on = "ATAC_cluster", right_on = "Cluster.ID")
	atac_cls = atac_cls.set_index("Cell.ID")
	data["rna"].obs = pd.merge(data["rna"].obs, atac_cls["Cluster.Name"], left_index=True, right_index = True, how="left")

	# Remove non developmental lineages
	atac_non_developmental = ["IN1", "IN2", "IN3", "IN4", "MG/EC/Peric."]
	cells_to_keep = data["rna"][~data["rna"].obs["Cluster.Name"].isin(atac_non_developmental)].obs_names
	mu.pp.filter_obs(data["rna"], var=cells_to_keep)

	# RNA preprocessing
	sc.pp.normalize_per_cell(data["rna"])
	sc.pp.log1p(data["rna"])
	sc.pp.highly_variable_genes(data["rna"], n_top_genes=2000, subset=True)
	sc.pp.scale(data["rna"])

	sc.tl.pca(data["rna"]) # 50 components
	sc.pl.pca_variance_ratio(data["rna"], n_pcs=50, save="RNA.png")

	# Compute nearest neighbor graph + umap
	k = 30
	pca = 30
	sc.pp.neighbors(data["rna"], n_neighbors=k, n_pcs = pca, use_rep="X_pca", random_state=seed)
	sc.tl.umap(data["rna"], random_state = seed)
	sc.tl.leiden(data["rna"], random_state=seed, key_added="leiden")
	sc.tl.louvain(data["rna"], random_state = seed, key_added = "louvain")
	sc.pl.embedding(data["rna"], color = ["Cluster.Name", "louvain", "leiden"], save = f"{experiment}_clusters.png", basis="X_umap")


	# Scvelo RNA velocities computations  
	scv.pp.moments(data["rna"])
	scv.tl.recover_dynamics(data["rna"], n_jobs=-1)
	scv.tl.velocity(data["rna"], mode="dynamical")
	scv.tl.velocity_graph(data["rna"], n_jobs=-1)
	scv.tl.latent_time(data["rna"])

	scv.pl.scatter(data["rna"], color='velocity_pseudotime', color_map='gnuplot', size=40, save="pseudotime.png")
	
	data["rna"].write(os.path.join(data_path, "rna", "data.h5ad"))
	
