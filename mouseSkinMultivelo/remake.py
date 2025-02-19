import os
import muon as mu
import scanpy as sc
import pandas as pd
import numpy as np
from palantirModel.palantir_wrapper import PalantirWrapper 
from palantirModel.palantir_plots import plot_palantir_results
from palantirModel.palantir_utils import PalantirComparator


if __name__=="__main__":
	seed=42
	np.random.seed(seed)
	
	data_path = ... 
	saving_path_multiomics = os.path.join(os.getcwd(), "palantir_results_multiomics")
	saving_path_rna = os.path.join(os.getcwd(), "palantir_results_rna")

	rna = sc.read_h5ad(os.path.join(data_path, "adata_postpro.h5ad"))
	activity = sc.read_h5ad(os.path.join(data_path, "adata_atac_postpro.h5ad"))

	# RNA dataset post processing already has PCA
	sc.pl.pca_variance_ratio(rna)
	sc.pp.normalize_total(activity)
	sc.pp.pca(activity, random_state=seed, use_highly_variable=False)
	sc.pl.pca_variance_ratio(activity)
	
	data = mu.MuData({"rna":rna, "activity":activity})
	
	pcs_rna = 30
	pcs_activity = 30
	knn_rna = 30
	knn_activity = 30

	sc.pp.neighbors(data["rna"], n_pcs = pcs_rna, n_neighbors=knn_rna, random_state=seed)
	sc.pp.neighbors(data["activity"], n_pcs = pcs_activity, n_neighbors=knn_activity, random_state=seed)
	mu.pp.neighbors(data, random_state = seed, n_neighbors=30, key_added="wnn")
	mu.tl.umap(data, random_state = seed, neighbors_key="wnn")
	mu.pl.embedding(data, basis="X_umap", color="rna:celltype")

	# Palantir run
	if not os.path.exists(saving_path_multiomics):
		os.mkdir(saving_path_multiomics)
	if not os.path.exists(saving_path_rna):
		os.mkdir(saving_path_rna)

	early_cell = np.random.choice(data.obs_names[data.obs["rna:celltype"]=="TAC-1"])
	medulla_cell = np.random.choice(data.obs_names[data.obs["rna:celltype"]=="Medulla"])
	irs_cell = np.random.choice(data.obs_names[data.obs["rna:celltype"]=="IRS"])
	terminal_states = [medulla_cell, irs_cell] 
	

	fix_terminal = False
	pw = PalantirWrapper()

	# Multiomics run
	pw.compute_kernel(data)
	pw.run_diffusion_maps(data)
	pw.determine_multiscale_space(data)
	if fix_terminal:
		pw.run_palantir(data, early_cell = early_cell, num_waypoints = 500, terminal_states = terminal_states)
	else:
		pw.run_palantir(data, early_cell = early_cell, num_waypoints=500)
	
	data.obs["kl_divergence"] = pw.compute_priming_degree(data, entropy_type = "kl_divergence")
	plot_palantir_results(data, saving_path = saving_path_multiomics, entropy_key=["palantir_entropy","kl_divergence"])
	# RNA run
	pw.compute_kernel(data["rna"], knn_key="neighbors", distance_key="distances")
	pw.run_diffusion_maps(data["rna"])
	pw.determine_multiscale_space(data["rna"])
	if fix_terminal:
		pw.run_palantir(data["rna"], early_cell = early_cell, num_waypoints = 500, terminal_states=terminal_states)
	else:
		pw.run_palantir(data["rna"], early_cell = early_cell, num_waypoints = 500)

	data["rna"].obs["kl_divergence"] = pw.compute_priming_degree(data, entropy_type = "kl_divergence")
	plot_palantir_results(data, modality_key = "rna", saving_path = saving_path_rna, entropy_key=["palantir_entropy", "kl_divergence"])

	if fix_terminal:
		pc = PalantirComparator()
		pc.save_palantir_matrix(data, data["rna"], "rna:celltype", saving_path = os.path.join(saving_path_multiomics, "fates.tsv"))
		pc.save_palantir_matrix(data, data["rna"], "rna:celltype", is_fate=False, key1 = "palantir_entropy", key2="palantir_entropy", saving_path = os.path.join(saving_path_multiomics, "entropy.tsv"))
		pc.save_palantir_matrix(data, data["rna"], "rna:celltype", is_fate=False, key1="kl_divergence", key2="kl_divergence", saving_path=os.path.join(saving_path_multiomics, "kl_divergence.tsv"))	

	
