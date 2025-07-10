import os
import json
import time
import cellrank
import argparse
import muon as mu
import scanpy as sc
import pandas as pd
import numpy as np
from itertools import product
from utils import aggregate_lineage_fate 
from core.palantirModel.plots import plot_entropy
from core.cellrankPseudotime.pseudokernel import PseudotimeKernelMuon
from core.matrix_analysis import MatrixAnalyser
from core.palantirModel.plots import plot_palantir_results
from core.palantirModel.palantir_wrapper import PalantirWrapper 
from core.metrics import compute_correlation, compute_f1

if __name__=="__main__":
	seed = 42	
	np.random.seed(seed)

	working_directory= ...
	donor = ...
	model = "multiomics"
	fixed_terminal = ...
	data_path = os.path.join(working_directory, "data", "lineage_tracing",  f"{donor}")	
	results_folder = os.path.join(working_directory, "results", "lineage_tracing")
	
	threshold_scheme = "hard"
	n_macrostates = range(4,20,2)

	palantir_cells_file = os.path.join(data_path, "selected_cells_palantir.json")
	with open(palantir_cells_file, "r") as f:
		cells = json.load(f)
		f.close()
	palantir_initial_barcode = cells["initial"]["hsc"]

	results = {"donor": donor, "model": model, "fixed_terminal":fixed_terminal, "n_macrostates" : None, "pruning_type":"hard", "pearson_kl_statistics": None, "pearson_kl_pvalue":None, "pearson_entropy_statistics":None, "pearson_entropy_pvalue": None, "f1_cosine":None, "f1_euclidean":None, "n_terminal_states":None} 
	
	#path declaration + additional files
	tsv_path = os.path.join(results_folder, f"pseudoKernel_results.csv")
	failures_path = os.path.join(results_folder, f"pseudoKernel_failures.csv")

	saving_folder = os.path.join(results_folder, f"pseudoKernel_{donor}_{model}_{fixed_terminal}")
	if not os.path.exists(saving_folder):
		os.mkdir(saving_folder)

	# read selected cells 
	cell_path = os.path.join(data_path, f"selected_cells.json")
	with open(cell_path, "r") as f:
		cells = json.load(f)
		f.close()
	
	# read ground truth fates probabilities 
	truth_path = os.path.join(data_path, "branch_assignment.csv") 
	ground_truth = pd.read_csv(truth_path, sep= ",", index_col=0, header=0)

	# read dataset
	data = mu.read_h5mu(os.path.join(data_path, "data.h5mu"))

	pw = PalantirWrapper()
	pw.compute_kernel(data)
	pw.run_diffusion_maps(data, seed=seed)
	pw.determine_multiscale_space(data)
	
	try:
		pw.run_palantir(data, early_cell = palantir_initial_barcode, seed=seed)
		data.update()
		kernel = PseudotimeKernelMuon(data=data, modality_key=None, embedding_key="X_umap", connectivity_key = "wnn_connectivities", pseudotime_key = "palantir_pseudotime", group_key=["STD.CellType"])
		kernel.compute_transition_matrix(threshold_scheme=threshold_scheme)			
		g = cellrank.estimators.GPCCA(kernel.kernel)
		g.compute_schur()

		for nms in n_macrostates:
			if fixed_terminal :
				g.set_initial_states(cells["initial"])
				g.set_terminal_states(cells["terminal"])
			else:
				g.compute_macrostates(n_states = nms, cluster_key="STD.CellType")
				g.predict_terminal_states()
				g.predict_initial_states(allow_overlap=True)

			g.compute_fate_probabilities(tol = 1e-10, preconditioner="ilu", n_jobs=-1, use_petsc=True, show_progress_bar=False, backend="threading")
			dataframe = pd.DataFrame(g.fate_probabilities.X, columns = g.fate_probabilities.names, index=data.obs_names)
	
			# if fix terminal then compute f1 score	
			if fixed_terminal :		
				dataframe_filtered = dataframe[dataframe.index.isin(ground_truth.index.to_list())]
				aggregate_lineage_fate(dataframe_filtered, cells["terminal"], use_barcodes=False)
				keys = ["lymphoid", "megakaryocyte", "erythroid", "myeloid"]
				cosine, euclidean = compute_f1(dataframe_filtered[keys], ground_truth[keys], aggregate=True)
				results["f1_cosine"] = cosine
				results["f1_euclidean"] = euclidean
			else: 
				results["f1_cosine"] = None
				results["f1_euclidean"] = None

			dataframe["pseudotime"] = data.obs["palantir_pseudotime"]
			dataframe["celltype"] = data.obs["STD.CellType"]
			dataframe["entropy"] = g.compute_lineage_priming(method="entropy")
			dataframe["KL"] = g.compute_lineage_priming(method="kl_divergence")
			dataframe = dataframe[["entropy", "pseudotime", "KL", "celltype"]]

#	 		plot images
			plot_path = f"fateProbs.png" if fixed_terminal else f"fateProbs_{nms}.png"
			title = f"Fate Probabilities" + "" if fixed_terminal else f" {nms}"
			g.plot_fate_probabilities(same_plot=True, save=os.path.join(saving_folder, plot_path), title=title, show=False)
			
			plot_path = os.path.join(saving_folder, f"entropy.png" if fixed_terminal else f"entropy_{nms}.png")
			plot_entropy(dataframe["entropy"], data.obsm["X_umap"], save=True, saving_path = plot_path)
			plot_path = os.path.join(saving_folder, f"kl.png" if fixed_terminal else f"kl_{nms}.png")
			plot_entropy(dataframe["KL"], data.obsm["X_umap"], save=True, saving_path = plot_path)

			if not fixed_terminal :
				plot_path = f"macrostate_composition_{nms}.png" 
				title = f"Macrostate Composition {nms}" 
				g.plot_macrostate_composition(key="STD.CellType", show=False, save=os.path.join(saving_folder, plot_path), title=title)
				plot_path = f"coarseT_{nms}.png" 
				title = f"Corse Grained TM {nms}" 
				g.plot_coarse_T(annotate=True, save = os.path.join(saving_folder, plot_path), title= title)
				results["n_terminal_states"] = g.fate_probabilities.shape[1]

			# Compute correlations and save results
			statistics, pvalue= compute_correlation(dataframe, "pearson", "pseudotime", "KL")
			results["pearson_kl_statistics"] = statistics
			results["pearson_kl_pvalue"] = pvalue

			statistics, pvalue = compute_correlation(dataframe, "pearson", "pseudotime", "entropy")
			results["pearson_entropy_statistics"] = statistics 
			results["pearson_entropy_pvalue"] = pvalue
	
			pd.DataFrame(results, index=[0]).to_csv(tsv_path, sep=",", header=False, index=False, mode="a")	
		
			if fixed_terminal:
				break
			
			
	except Exception as e:
		print(e)
		pd.DataFrame([(donor, model, fixed_terminal, nms)], index=[0]).to_csv(failures_path, sep=",", header=False, index=False, mode="a")

