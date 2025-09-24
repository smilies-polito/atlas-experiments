import os
import json
import cellrank
import muon as mu
import scanpy as sc
import pandas as pd
import numpy as np
from src.utils import aggregate_lineage_fate 
from src.plots import plot_entropy
from src.pseudokernel import PseudotimeKernelMuon, analysis_matrix
from src.plots import plot_palantir_results
from src.palantir_wrapper import PalantirWrapper 
from src.metrics import compute_correlation, compute_f1

if __name__=="__main__":
	seed = 42	
	np.random.seed(seed)

	working_directory= "/scvemo"
	donor = "donor2"
	model = "rna"
	fixed_terminal = True
	data_path = os.path.join(working_directory, "data", "lineage_tracing",  donor)
	results_folder = os.path.join(working_directory, "output", "lineage_tracing")
	
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
	matrix_path = os.path.join(saving_folder, "matrix_analysis.json")

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
	data["rna"].obs = data.obs[["lineage", "STD.CellType"]]
	data["rna"].obsm["X_umap"] = data.obsm["X_umap"]

	pw = PalantirWrapper()
	pw.compute_kernel(data["rna"], knn_key ="neighbors", distance_key="connectivities")
	pw.run_diffusion_maps(data["rna"], seed=seed)
	pw.determine_multiscale_space(data["rna"])
	
	try:
		pw.run_palantir(data["rna"], early_cell = palantir_initial_barcode, seed=seed)
		data.update()
		kernel = PseudotimeKernelMuon(data=data, modality_key="rna", embedding_key="X_umap", connectivity_key = "connectivities", pseudotime_key = "palantir_pseudotime", group_key=["STD.CellType"])
		kernel.compute_transition_matrix(threshold_scheme=threshold_scheme)			
		if fixed_terminal:
			matrix_analysis = analysis_matrix(kernel.kernel.transition_matrix)
			with open(matrix_path, "w") as f:
				json.dump(matrix_analysis, f)
				f.close()

		g = cellrank.estimators.GPCCA(kernel.kernel)
		g.compute_schur()

		for n_macrostates in macrostates_to_eval:
			results_path = saving_folder if fixed_terminal else os.path.join(saving_folder, f"{n_macrostates}")
			if not os.path.exists(results_path):
				os.mkdir(results_path)

			if fixed_terminal:
				g.set_initial_states(cells["initial"])
				g.set_terminal_states(cells["terminal"])
			else:
				g.compute_macrostates(n_states = n_macrostates, cluster_key="STD.CellType")
				g.predict_terminal_states()
				g.predict_initial_states(allow_overlap=True)

			g.compute_fate_probabilities(tol = 1e-10, preconditioner="ilu", n_jobs=-1, use_petsc=True, show_progress_bar=False, backend="threading")
			dataframe = pd.DataFrame(g.fate_probabilities.X, columns = g.fate_probabilities.names, index=data.obs_names)
	
			# if fix terminal then compute f1 score	
			if fixed_terminal:		
				dataframe_filtered = dataframe[dataframe.index.isin(ground_truth.index.to_list())]
				aggregate_lineage_fate(dataframe_filtered, cells["terminal"], use_barcodes=False)
				keys = ["lymphoid", "megakaryocyte", "erythroid", "myeloid"]
				cosine, euclidean = compute_f1(dataframe_filtered[keys], ground_truth[keys], aggregate=True)
				results["f1_cosine"] = cosine
				results["f1_euclidean"] = euclidean
			else: 
				results["f1_cosine"] = None
				results["f1_euclidean"] = None

			dataframe["pseudotime"] = data.obs["rna:palantir_pseudotime"]
			dataframe["celltype"] = data.obs["STD.CellType"]
			dataframe["entropy"] = g.compute_lineage_priming(method="entropy")
			dataframe["KL"] = g.compute_lineage_priming(method="kl_divergence")
			dataframe = dataframe[["entropy", "pseudotime", "KL", "celltype"]]

#	 		plot images
			plot_path = os.path.join(results_path, f"fateProbs.png" )
			title = f"Fate Probabilities" + "" if fixed_terminal else f" {n_macrostates}"
			g.plot_fate_probabilities(same_plot=True, save=plot_path, title=title, show=False)
			
			plot_entropy(dataframe["KL"], data.obsm["X_umap"], save=True, saving_path = plot_path)

			if not fixed_terminal :
				plot_path = os.path.join(results_path, f"macrostate_composition_{n_macrostates}.png")
				title = f"Macrostate Composition {n_macrostates}" 
				g.plot_macrostate_composition(key="STD.CellType", show=False, save=plot_path, title=title)
				plot_path = os.path.join(results_path, f"coarseT_{n_macrostates}.png")
				title = f"Corse Grained TM {n_macrostates}" 
				g.plot_coarse_T(annotate=True, save = plot_path, title= title)
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
		pd.DataFrame([(donor, model, fixed_terminal, n_macrostates)], index=[0]).to_csv(failures_path, sep=",", header=False, index=False, mode="a")

