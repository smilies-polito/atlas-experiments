#######################
# Brain Chromatin Preprocessing
# Data from https//github.com/GreenleafLab/brainchromatin/blob/main/links.txt
####################### 

import os
import anndata
import muon as mu
import numpy as np
import scvelo as scv
import pandas as p
import scanpy as sc
from muon import atac as ac
from muon import MuData


def create_feature_map(rna, strand:bool = False):
	features = rna.var.copy()
	features[['Chromosome', 'Start', 'End']] = features['interval'].str.extract(r'(?P<Chromosome>[^:]+):(?P<Start>\d+)-(?P<End>\d+)')
	features = features.loc[~features.Start.isnull()]
	features.Start = features.Start.astype(int)
	features.End = features.End.astype(int) 
	features = features[features.Start>2e3]
	features = features[features.End>0]
	return features[["Start", "Chromosome", "End", "interval"]] 
			
def add_rna_counts(activity, multiomics, rna):
	"""
	Subset for common genes, add spliced and unsplcied acountsand return new mudata with activituies and RNA
	"""	
	pass
		
	


if __name__ == "__main__":
	seed=52

	os.chdir("")
	rna = sc.read_loom(os.path.join(os.getcwd(), "e18_mouse_brain.loom"))
	multiomics = mu.read_10x_h5(os.path.join(os.getcwd(), "e18_mouse_brain_fresh_5k_filtered_feature_bc_matrix.h5"))
	
	rna.obs_names = [s.split(":")[1][:-1]  + "-1" for s in rna.obs_names]
	mu.pp.filter_obs(multiomics, var=rna.obs_names)
	
	features = create_feature_map(multiomics.mod["rna"])
	mu.atac.tl.locate_fragments(multiomics, os.path.join(os.getcwd(), "e18_mouse_brain_fresh_5k_atac_fragments.tsv.gz"))
	
	# not present in iterator: to be removed 
	features = features[~features.Chromosome.isin(["GL456219.1", "JH584293.1", "JH584294.1", "JH584294.1", "JH584298.1", "JH584303.1"])]
	activity = mu.atac.tl.count_fragments_features(multiomics, features, stranded=False)
	print(activity)
	print(rna)
	print(multiomics)
	
	add_rna_counts(activity, multiomics, rna)

#	# Add cluster names and clusters 
#	clusters = pd.read_csv(cluster_path, sep="\t", header=0)
#	clusters_atac = metadata["ATAC_cluster"].reset_index().merge(clusters[clusters.Assay=="Multiome ATAC"], how="left", left_on = "ATAC_cluster", right_on = "Cluster.ID")
#	clusters_rna = metadata["seurat_clusters"].reset_index().merge(clusters[clusters.Assay=="Multiome RNA"], how="left", left_on="seurat_clusters", right_on = "Cluster.ID")
#	clusters = pd.merge(clusters_atac, clusters_rna, how = "left", left_on = "Cell.ID", right_on="Cell.ID").set_index("Cell.ID")
#	clusters = clusters.loc[:, ["Cluster.Name_x", "Cluster.Name_y"]]
#	clusters.columns = ["ATAC_Clusters", "RNA_Clusters"]

#	add_clusters(dc1r3_r1, clusters, f"{prefix}dc1r3_r1")	
#	add_clusters(dc2r2_r1, clusters, f"{prefix}dc2r2_r1")
#	add_clusters(dc2r2_r2, clusters, f"{prefix}dc2r2_r2")
#
#	add_rna_counts(dc1r3_r1, spliced_counts, unspliced_counts, f"{prefix}dc1r3_r1")
#	add_rna_counts(dc2r2_r2, spliced_counts, unspliced_counts, f"{prefix}dc2r2_r2")
#	add_rna_counts(dc2r2_r1, spliced_counts, unspliced_counts, f"{prefix}dc2r2_r1")
#
#	# Remove non developmental lineages
#	atac_non_developmental = ["IN1", "IN2", "IN3", "IN4", "MG/EC/Peric."]
#	mu.pp.filter_obs(dc1r3_r1, var=~dc1r3_r1.obs["rna:ATAC_Clusters"].isin(atac_non_developmental))
#	mu.pp.filter_obs(dc2r2_r1, var=~dc2r2_r1.obs["rna:ATAC_Clusters"].isin(atac_non_developmental))	
#	mu.pp.filter_obs(dc2r2_r2, var=~dc2r2_r2.obs["rna:ATAC_Clusters"].isin(atac_non_developmental))
#		
#	# Gene Activity
#	dc1r3_r1_GA = compute_gene_activity(dc1r3_r1)
#	dc2r2_r1_GA = compute_gene_activity(dc2r2_r1)
#	dc2r2_r2_GA = compute_gene_activity(dc2r2_r2)	
#
#	# Create unique df
#	update_obs_names(dc1r3_r1, f"{prefix}dc1r3_r1")
#	update_obs_names(dc2r2_r1, f"{prefix}dc2r2_r1")
#	update_obs_names(dc2r2_r2, f"{prefix}dc2r2_r2")
#	
#	rna = anndata.concat([dc1r3_r1["rna"], dc2r2_r1["rna"], dc2r2_r2["rna"]], label="batch", keys=["dc1r3_r1", "dc2r2_r1", "dc2r2_r2"])
#	
#	# RNA reprocessing
#	sc.pp.normalize_per_cell(rna)
#	sc.pp.log1p(rna)
#	sc.pp.highly_variable_genes(rna, n_top_genes=2000, subset=True)
#	sc.pp.scale(rna)
#	sc.tl.pca(rna) # 50 components
#	sc.pl.pca_variance_ratio(rna, n_pcs=50, save="RNA.png")
#	
#	update_modality_obs_names(dc1r3_r1_GA, f"{prefix}dc1r3_r1")
#	update_modality_obs_names(dc2r2_r1_GA, f"{prefix}dc2r2_r1")
#	update_modality_obs_names(dc2r2_r2_GA, f"{prefix}dc2r2_r2")
#	
#	activity = anndata.concat([dc1r3_r1_GA, dc2r2_r1_GA, dc2r2_r2_GA], label="batch", keys=["dc1r3_r1", "dc2r2_r1", "dc2r2_r2"])
#
#	# Gene Activity preprocessing
#	sc.pp.normalize_per_cell(activity)
#	mu.pp.filter_var(activity, var=rna.var_names)
#	sc.pp.log1p(activity)
#	sc.tl.pca(activity) 
#	sc.pl.pca_variance_ratio(activity, n_pcs=50, save=f"_gene_activity.png")
#	
#	sc.external.pp.bbknn(rna, batch_key="batch", neighbors_within_batch=10, n_pcs=30, use_rep="X_pca")
#	sc.tl.umap(rna)
#	sc.pl.embedding(rna, basis="X_umap", color="batch", save = "_rna_batch.png")
#	
#	sc.pp.neighbors(activity, n_neighbors=30, n_pcs= 30, use_rep="X_pca")
#	sc.tl.umap(activity)
#	sc.pl.embedding(activity, basis="X_umap", color="batch", save="_gene_activity_batch.png")
#	
#	data = MuData({'rna': rna.copy(), 'activity': activity.copy()})
#	mu.pp.neighbors(data, n_neighbors=30, random_state=seed, key_added = "wnn")
#	mu.tl.umap(data, neighbors_key = "wnn", random_state=seed)
#	mu.pl.embedding(data, basis = "X_umap", color=["rna:RNA_Clusters", "rna:ATAC_Clusters"], save=f"_multimodal.png")
#
#	scv.pp.moments(data["rna"])	
#	scv.tl.recover_dynamics(data["rna"], n_jobs=-1)
#	scv.tl.velocity(data["rna"], mode="dynamical")
#	scv.tl.velocity_graph(data["rna"])
#	scv.tl.velocity_pseudotime(data["rna"])
#	mu.pp.intersect_obs(data)
#	
#	mu.pl.embedding(data, color='rna:velocity_pseudotime', basis="X_umap", color_map='gnuplot', size=40, save=f"_pseudotime.png")	
#
#	data.write_h5mu(os.path.join(os.getcwd(), "data.h5mu"))
