import os
import json
import argparse
import argparse
import cellrank
import muon as mu
import scanpy as sc
import pandas as pd
import numpy as np
from src.plots import plot_entropy 
from src.palantir_wrapper import PalantirWrapper 
from src.pseudokernel import PseudotimeKernelMuon, analysis_matrix


if __name__=="__main__":
	seed = 42
	np.random.seed(seed)

	parser = argparse.ArgumentParser()
	parser.add_argument("--site", type=str)
	parser.add_argument("--lineage", type=str)
	args = parser.parse_args()
	lineage = args.lineage
	site = args.site
	
	working_dir = "/scvemo"
	data_path = os.path.join(working_dir, "data", "skeletalDev")
	results_folder = os.path.join(working_dir, "output", "skeletalDev",f"bbknn_{site}_{lineage}")

	macrostates_to_eval = list(range(2,10,2))

	#path declaration + additional files
	data_suffix = f"bbknn_{site}.h5mu" if lineage == "whole" else f"bbknn_{site}_{lineage}.h5mu"
	data = mu.read_h5mu(os.path.join(data_path, data_suffix))
	data["rna"].obsm["X_umap"] = data.obsm["X_umap"]

	rna_folder = os.path.join(results_folder, f"pseudotime_rna")
	if not os.path.exists(rna_folder):
		os.mkdir(rna_folder)

	multiomics_folder = os.path.join(results_folder, f"pseudotime_multiomics")
	if not os.path.exists(multiomics_folder):
		os.mkdir(multiomics_folder)

	# select initial cell 
	early_cell_path_suffix = f"selected_cells_palantir_bbknn_{site}.json" if lineage == "whole" else f"selected_cells_palantir_bbknn_{site}_{lineage}.json"
	early_cell_path = os.path.join(data_path, early_cell_path_suffix)
	with open(early_cell_path, "r") as f:
		early_cell = json.load(f)
		f.close()
	early_cell = early_cell["initial"]["initial"]


	print("EXECUTING PSEUDOTIME RNA")
	pw = PalantirWrapper()
	pw.compute_kernel(data["rna"], knn_key="neighbors", distance_key = "distances")
	pw.run_diffusion_maps(data["rna"], seed=seed)
	pw.determine_multiscale_space(data["rna"])

	try:
		pw.run_palantir(data["rna"], early_cell=early_cell, seed = seed) 
		kernel = PseudotimeKernelMuon(data=data, modality_key = "rna", embedding_key = "X_umap", connectivity_key="connectivities", pseudotime_key="palantir_pseudotime", group_key="Celltype_fig1")
		kernel.compute_transition_matrix(threshold_scheme="hard")
		
		matrix_path = os.path.join(rna_folder, "matrix_analysis.json")
		matrix_analysis = analysis_matrix(kernel.kernel.transition_matrix)
		with open(matrix_path, "w") as f:
			json.dump(matrix_analysis, f)
			f.close()

		g = cellrank.estimators.GPCCA(kernel.kernel)
		g.compute_schur()
		for n_macrostates in macrostates_to_eval:
			results_folder = os.path.join(rna_folder, f"{n_macrostates}")
			if not os.path.exists(results_folder):
				os.mkdir(results_folder)
			g.compute_macrostates(n_states=n_macrostates, cluster_key="Celltype_fig1")
			g.predict_terminal_states()
			g.predict_initial_states(allow_overlap=True)
			plot_path = f"macrostate_composition.png" 
			title = f"Macrostate Composition {n_macrostates}" 
			g.plot_macrostate_composition(key="Celltype_fig1", show=False, save=os.path.join(results_folder, plot_path), title=title)
			plot_path = f"coarseT.png" 
			title = f"Corse Grained TM {n_macrostates}"
			g.plot_coarse_T(annotate=True, save = os.path.join(results_folder, plot_path), title= title)
			g.compute_fate_probabilities(tol=1e-10, preconditioner="ilu", n_jobs=-1, use_petsc=True, show_progress_bar=False, backend="threading")
			plot_path = f"fateProbs.png"
			title = f"Fate Probabilities {n_macrostates}" 
			g.plot_fate_probabilities(same_plot=True, save=os.path.join(results_folder, plot_path), title=title, show=False, color="Celltype_fig1")

			dataframe = pd.DataFrame(data["rna"].obs["Celltype_fig1"])
			dataframe["entropy"] = g.compute_lineage_priming(method="entropy")
			dataframe["KL"] = g.compute_lineage_priming(method="kl_divergence")
			plot_entropy(dataframe["entropy"], data.obsm["X_umap"], save=True, saving_path = results_folder)
			plot_entropy(dataframe["KL"], data.obsm["X_umap"], save=True, saving_path = results_folder)
			file_name = f"bbknn_{site}_rna_{n_macrostates}.csv" if lineage=="whole" else f"bbknn_{site}_{lineage}_rna_{n_macrostates}.csv"
			pd.DataFrame(g.fate_probabilities.X , columns=g.fate_probabilities.names, index=data.obs_names).to_csv(os.path.join(data_path, file_name), sep=",", header=True, index=True)

	except Exception as e:
		print(e)

	#EXECUTE PALANTIR MULTIOMICS
	print("EXECUTING PSEUDOTIME MULTIOMICS")
	pw = PalantirWrapper()
	pw.compute_kernel(data)
	pw.run_diffusion_maps(data, seed=seed)
	pw.determine_multiscale_space(data)

	try:
		pw.run_palantir(data, early_cell=early_cell, seed = seed) 
		kernel = PseudotimeKernelMuon(data=data, modality_key = None, embedding_key = "X_umap", connectivity_key="wnn_connectivities", pseudotime_key="palantir_pseudotime", group_key="rna:Celltype_fig1")
		kernel.compute_transition_matrix(threshold_scheme="hard")

		matrix_path = os.path.join(multiomics_folder, "matrix_analysis.json")
		matrix_analysis = analysis_matrix(kernel.kernel.transition_matrix)
		with open(matrix_path, "w") as f:
			json.dump(matrix_analysis, f)
			f.close()

		g = cellrank.estimators.GPCCA(kernel.kernel)
		g.compute_schur()
		for n_macrostates in macrostates_to_eval:
			results_folder = os.path.join(multiomics_folder, f"{n_macrostates}")
			if not os.path.exists(results_folder):
				os.mkdir(results_folder)
			g.compute_macrostates(n_states=n_macrostates, cluster_key="rna:Celltype_fig1")
			g.predict_terminal_states()
			g.predict_initial_states(allow_overlap=True)
			plot_path = f"macrostate_composition.png" 
			title = f"Macrostate Composition {n_macrostates}" 
			g.plot_macrostate_composition(key="rna:Celltype_fig1", show=False, save=os.path.join(results_folder, plot_path), title=title)
			plot_path = f"coarseT.png" 
			title = f"Corse Grained TM {n_macrostates}"
			g.plot_coarse_T(annotate=True, save = os.path.join(results_folder, plot_path), title= title)
	
			g.compute_fate_probabilities(tol=1e-10, preconditioner="ilu", n_jobs=-1, use_petsc=True, show_progress_bar=False, backend="threading")
			plot_path = f"fateProbs.png"
			title = f"Fate Probabilities {n_macrostates}" 
			g.plot_fate_probabilities(same_plot=True, save=os.path.join(results_folder, plot_path), title=title, show=False, color="rna:Celltype_fig1")

			dataframe = pd.DataFrame(data.obs["rna:Celltype_fig1"])
			dataframe["entropy"] = g.compute_lineage_priming(method="entropy")
			dataframe["KL"] = g.compute_lineage_priming(method="kl_divergence")
			plot_entropy(dataframe["entropy"], data.obsm["X_umap"], save=True, saving_path = results_folder)
			plot_entropy(dataframe["KL"], data.obsm["X_umap"], save=True, saving_path = results_folder)
			file_name = f"bbknn_{site}_multiomics_{n_macrostates}.csv" if lineage=="whole" else f"bbknn_{site}_{lineage}_multiomics_{n_macrostates}.csv"
			pd.DataFrame(g.fate_probabilities.X , columns=g.fate_probabilities.names, index=data.obs_names).to_csv(os.path.join(data_path, file_name), sep=",", header=True, index=True)


	except Exception as e:
		print(e)

