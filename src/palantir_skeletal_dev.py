import os
import json
import argparse
import muon as mu
import scanpy as sc
import pandas as pd
import numpy as np
from muon import MuData
from anndata import AnnData
from typing import Union, Optional
from src.palantir_wrapper import PalantirWrapper
from src.plots import plot_palantir_results, plot_heatmap, plot_eigenvalues 

if __name__=="__main__":
	seed = 42
	np.random.seed(seed)

	working_dir = "/scvemo"
	data_path = os.path.join(working_dir, "data", "skeletalDev")

	parser = argparse.ArgumentParser()
	parser.add_argument("--site", type=str)
	parser.add_argument("--lineage", type=str)
	args = parser.parse_args()
	lineage = args.lineage
	site = args.site

	results_folder = os.path.join(working_dir, "output", "skeletalDev", f"bbknn_{site}_{lineage}")
	if not os.path.exists(results_folder):
		os.mkdir(results_folder)
 
	early_cell_path_suffix = f"selected_cells_palantir_bbknn_{site}.json" if lineage == "whole" else f"selected_cells_palantir_bbknn_{site}_{lineage}.json"
	early_cell_path = os.path.join(data_path, early_cell_path_suffix)

	with open(early_cell_path, "r") as f:
		early_cell = json.load(f)
		f.close()
	early_cell = early_cell["initial"]["initial"]

	#path declaration + additional files
	data_suffix = f"bbknn_{site}.h5mu" if lineage == "whole" else f"bbknn_{site}_{lineage}.h5mu"
	data = mu.read_h5mu(os.path.join(data_path, data_suffix))

	data["rna"].obsm["X_umap"] = data.obsm["X_umap"]

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

	if lineage != "whole":
		plot_heatmap(data=data["rna"], similarity_key="DM_Similarity", group_key=["Celltype_fig1"], subset_key =None, keep_subset=None, save = os.path.join(rna_folder, f"similarity_celltypes.png"))
	else:
		lineages = list(data.obs["rna:lineage"].unique())
		for ln in lineages:
			keep_subset=list(data.obs[data.obs["rna:lineage"]==ln]["rna:Celltype_fig1"].unique())
			plot_heatmap(data= data["rna"], similarity_key = "DM_Similarity", group_key=["Celltype_fig1"], subset_key="Celltype_fig1", keep_subset=keep_subset, save= os.path.join(rna_folder, f"similarity_{ln}_celltypes.png"))

	plot_heatmap(data=data["rna"], similarity_key="DM_EigenVectors_multiscaled", group_key=["Celltype_fig1"], subset_key =None, keep_subset=None, save = os.path.join(rna_folder, f"diffusion_space_celltypes.png"))
	
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

	if lineage != "whole":
		plot_heatmap(data=data, similarity_key="DM_Similarity", group_key=["rna:Celltype_fig1"], subset_key =None, keep_subset=None, save = os.path.join(multiomics_folder, f"similarity_celltypes.png"))
	else:
		lineages = list(data.obs["rna:lineage"].unique())
		for ln in lineages:
			keep_subset=list(data.obs[data.obs["rna:lineage"]==ln]["rna:Celltype_fig1"].unique())
			plot_heatmap(data= data, similarity_key = "DM_Similarity", group_key=["rna:Celltype_fig1"], subset_key="rna:Celltype_fig1", keep_subset=keep_subset, save= os.path.join(multiomics_folder, f"similarity_{ln}_celltypes.png"))

	plot_heatmap(data=data, similarity_key="DM_EigenVectors_multiscaled", group_key=["rna:Celltype_fig1"], subset_key =None, keep_subset=None, save = os.path.join(multiomics_folder, f"diffusion_space_celltypes.png"))

	try:
		pw.run_palantir(data, early_cell=early_cell, seed = seed) 
		plot_palantir_results(data = data, modality_key = None, embedding_key = "X_umap", pseudo_time_key = "palantir_pseudotime", entropy_key = "palantir_entropy", fate_prob_key = "palantir_fate_probabilities", save= True, saving_path = multiomics_folder)

	except Exception as e:
		print(e)

	saving_eigenvalues = os.path.join(results_folder, "eigenvalues.png")	
	plot_eigenvalues(data, saving_path=saving_eigenvalues)

	data_suffix = f"palantir_bbknn_{site}.h5mu" if lineage=="whole" else f"palantir_bbknn_{site}_{lineage}.h5mu"
	data.write_h5mu(os.path.join(data_path, data_suffix))
