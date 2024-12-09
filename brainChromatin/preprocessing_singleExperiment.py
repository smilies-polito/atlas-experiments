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
	peak_path = os.path.join(data_path, "atac_peak_annotation.tsv")

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
	data.update_var()

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
	rna_cls = pd.merge(metadata[["Cell.ID", "seurat_clusters"]], clusters[clusters["Assay"]=="Multiome RNA"], how="left", left_on="seurat_clusters", right_on="Cluster.ID")
	rna_cls = rna_cls.set_index("Cell.ID")
	atac_cls = pd.merge(metadata[["Cell.ID", "ATAC_cluster"]], clusters[clusters["Assay"]=="Multiome ATAC"], how="left", left_on = "ATAC_cluster", right_on = "Cluster.ID")
	atac_cls = atac_cls.set_index("Cell.ID")
	data["rna"].obs = pd.merge(data["rna"].obs, rna_cls["Cluster.Name"], left_index=True, right_index = True, how="left")
	data["atac"].obs= pd.merge(data["atac"].obs, atac_cls["Cluster.Name"], left_index=True, right_index= True, how="left")
	data.update()
	mu.pp.intersect_obs(data)

	# Remove non developmental lineages
	atac_non_developmental = ["IN1", "IN2", "IN3", "IN4", "MG/EC/Peric."]
	mu.pp.filter_obs(data, var=~data.obs["atac:Cluster.Name"].isin(atac_non_developmental))
	

	# RNA preprocessing
	sc.pp.normalize_per_cell(data["rna"])
	sc.pp.log1p(data["rna"])
	sc.pp.highly_variable_genes(data["rna"], n_top_genes=2000, subset=True)
	data.update()
	sc.pp.scale(data["rna"])

	sc.tl.pca(data["rna"]) # 50 components
	sc.pl.pca_variance_ratio(data["rna"], n_pcs=50, save="RNA.png")

	# ATAC preprocessing
	ac.pp.tfidf(data["atac"])
	ac.tl.lsi(data["atac"])
	data["atac"].obsm["X_lsi"]=data["atac"].obsm["X_lsi"][:,1:]
	data["atac"].varm["LSI"]= data["atac"].varm["LSI"][:, 1:]
	data["atac"].uns["lsi"]["stdev"] = data["atac"].uns["lsi"]["stdev"][1:]

	# Compute nearest neighbor graph + umap
	k = 30
	pca = 30
	lsi = 10
	sc.pp.neighbors(data["rna"], n_neighbors=k, n_pcs = pca, use_rep="X_pca", random_state=seed)
	sc.pp.neighbors(data["atac"], n_neighbors=k, n_pcs=lsi, use_rep="X_lsi", random_state=seed)
	mu.pp.neighbors(data, n_neighbors=30, key_added="wnn")
	mu.tl.umap(data, neighbors_key = "wnn", random_state=seed)
	mu.pl.embedding(data, basis = "X_umap", color=["rna:Cluster.Name", "atac:Cluster.Name"], save="multimodal.png")
	mu.tl.leiden(data, random_state=seed, key_added="leiden_clusters")
	mu.tl.louvain(data, random_state = seed, key_added="louvain_clusters")
	mu.pl.embedding(data, basis="X_umap", color=["leiden_clusters", "louvain_clusters"], save=f"{experiment}_clusters.png")

	# Scvelo RNA velocities computations  
	scv.pp.moments(data["rna"])
	scv.tl.recover_dynamics(data["rna"], n_jobs=-1)
	scv.tl.velocity(data["rna"], mode="dynamical")
	scv.tl.velocity_graph(data["rna"], n_jobs=-1)
	scv.tl.latent_time(data["rna"])
	data.update_obs()
	data.update_var()

	scv.pl.scatter(data, color='rna:velocity_pseudotime', color_map='gnuplot', size=40, save="pseudotime.png")
	
	# Check for promoter genes and peaks
	peaks = pd.read_csv(peak_path, sep="\t", header=0)
	promoters = peaks[peaks["peak_type"]=="promoter"]
	renamed_peaks = promoters["peak"].str.split("_")
	promoters["peak"] = renamed_peaks.map(lambda x: x[0] + ":" + x[1] + "-" + x[2])

	promoters = promoters[(promoters.gene.isin(data["rna"].var_names)) & (promoters.peak.isin(data["atac"].var_names))]
	gene_intersection = set(promoters.gene).intersection(set(data["rna"].var_names))
	peak_intersection = set(promoters.peak).intersection(set(data["atac"].var_names))
	promoters = promoters[(promoters.gene.isin(gene_intersection)) & (promoters.peak.isin(peak_intersection))]
	mu.pp.filter_var(data["rna"], var=gene_intersection)
	mu.pp.filter_var(data["atac"], var=peak_intersection)
	data.update_var()

	data.write(os.path.join(data_path, "data.h5mu"))
	promoters.to_csv(os.path.join(data_path, "promoter_annotations.tsv"), sep="\t", header=True)
	
