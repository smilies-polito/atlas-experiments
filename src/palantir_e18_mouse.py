import os
import json
import muon as mu
import scanpy as sc
import pandas as pd
import numpy as np
from src.palantir_wrapper import PalantirWrapper
from src.plots import plot_palantir_results, plot_heatmap 


if __name__=="__main__":
	seed = 42
	np.random.seed(seed)

	working_dir = "/scvemo"
	data_path = os.path.join(working_dir, "data", "e18_mouse")
	results_folder = os.path.join(working_dir, "output", "e18_mouse")
	early_cell_path = os.path.join(data_path, "selected_cells_palantir.json")

	with open(early_cell_path, "r") as f:
		early_cell = json.load(f)
		f.close()
	early_cell = early_cell["initial"]["rg"]

	#path declaration + additional files
	data = mu.read_h5mu(os.path.join(data_path, "data.h5mu"))
	data["rna"].obsm["X_umap"] = data.obsm["X_umap"]
	data["rna"].obs = data.obs[["celltype"]]

	rna_folder = os.path.join(results_folder, f"palantir_rna")
	if not os.path.exists(rna_folder):
		os.mkdir(rna_folder)

	multiomics_folder = os.path.join(results_folder, f"palantir_multiomics")
	if not os.path.exists(multiomics_folder):
		os.mkdir(multiomics_folder)


	#EXECUTE PALANTIR RNA
	print("EXECUTING PALANTIR - RNA MODEL")
	pw = PalantirWrapper()
	pw.compute_kernel(data["rna"], knn_key="neighbors", distance_key = "distances")
	pw.run_diffusion_maps(data["rna"], seed=seed)
	pw.determine_multiscale_space(data["rna"])

	plot_heatmap(data=data["rna"], similarity_key="DM_Similarity", group_key="celltype", subset_key=None, keep_subset=None, save=os.path.join(rna_folder, "similarity_celltypes.png"))
	plot_heatmap(data=data["rna"], similarity_key="DM_EigenVectors_multiscaled", group_key="celltype", subset_key=None, keep_subset=None, save=os.path.join(rna_folder, "diffusion_space_celltypes.png"))

	try:
		pw.run_palantir(data["rna"], early_cell=early_cell, seed = seed) 
		plot_palantir_results(data = data, modality_key = "rna", embedding_key = "X_umap", pseudo_time_key = "palantir_pseudotime", entropy_key = "palantir_entropy", fate_prob_key = "palantir_fate_probabilities", save= True, saving_path = rna_folder)

	except Exception as e:
		print(e)

	
	#EXECUTE PALANTIR MULTIOMICS
	print("EXECUTING PALANTIR - MULTIOMICS MODEL")
	pw = PalantirWrapper()
	pw.compute_kernel(data)
	pw.run_diffusion_maps(data, seed=seed)
	pw.determine_multiscale_space(data)
	
	plot_heatmap(data=data, similarity_key="DM_Similarity", group_key="celltype", subset_key=None, keep_subset=None, save=os.path.join(multiomics_folder, "similarity_celltypes.png"))
	plot_heatmap(data=data, similarity_key="DM_EigenVectors_multiscaled", group_key="celltype", subset_key=None, keep_subset=None, save=os.path.join(multiomics_folder, "diffusion_space_celltypes.png"))

	try:
		pw.run_palantir(data, early_cell=early_cell, seed = seed) 
		plot_palantir_results(data = data, modality_key = None, embedding_key = "X_umap", pseudo_time_key = "palantir_pseudotime", entropy_key = "palantir_entropy", fate_prob_key = "palantir_fate_probabilities", save= True, saving_path = multiomics_folder)

	except Exception as e:
		print(e)

	data.write(os.path.join(data_path, "palantir_data.h5mu"))
