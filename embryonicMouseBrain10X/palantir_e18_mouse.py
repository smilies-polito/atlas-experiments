import os
import json
import time
import muon as mu
import scanpy as sc
import pandas as pd
import numpy as np
from itertools import product
from utils import aggregate_lineage_fate
from core.palantirModel.plots import plot_palantir_results, plot_similarity_matrix, plot_diffusion_space
from core.palantirModel.palantir_wrapper import PalantirWrapper 
from core.palantirModel.utils import _save_results
from core.metrics import compute_correlation, compute_f1
from core.utils import save_run_results


if __name__=="__main__":
	seed = 42
	np.random.seed(seed)

	working_dir = "/home/scvemo"
	data_path = os.path.join(working_dir, "data", "e18_mouse")
	results_folder = os.path.join(working_dir, "results", "e18_mouse")


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

	# select initial cell 
	early_cell = np.random.choice(data[data.obs["celltype"]=="RG, Astro, OPC"].obs_names)
	print(type(early_cell))

	#EXECUTE PALANTIR MULTIOMICS
	pw = PalantirWrapper()
	pw.compute_kernel(data["rna"], knn_key="neighbors", distance_key = "distances")
	pw.run_diffusion_maps(data["rna"], seed=seed)
	pw.determine_multiscale_space(data["rna"])

	try:
		pw.run_palantir(data["rna"], early_cell=early_cell, seed = seed) 
		plot_palantir_results(data = data, modality_key = "rna", embedding_key = "X_umap", pseudo_time_key = "palantir_pseudotime", entropy_key = "palantir_entropy", fate_prob_key = "palantir_fate_probabilities", save= True, saving_path = rna_folder)

	except Exception as e:
		print(e)

	#EXECUTE PALANTIR MULTIOMICS
	pw = PalantirWrapper()
	pw.compute_kernel(data)
	pw.run_diffusion_maps(data, seed=seed)
	pw.determine_multiscale_space(data)

	try:
		pw.run_palantir(data, early_cell=early_cell, seed = seed) 
		plot_palantir_results(data = data, modality_key = None, embedding_key = "X_umap", pseudo_time_key = "palantir_pseudotime", entropy_key = "palantir_entropy", fate_prob_key = "palantir_fate_probabilities", save= True, saving_path = multiomics_folder)

	except Exception as e:
		print(e)
