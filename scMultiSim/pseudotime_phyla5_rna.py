import os
import gc
import json
import muon as mu
import scanpy as sc
import pandas as pd
import cellrank
import numpy as np
from itertools import product
from core.cellrankPseudotime.pseudokernel import PseudotimeKernelMuon
from core.matrix_analysis import MatrixAnalyser
from core.palantirModel.plots import plot_palantir_results
from core.palantirModel.palantir_wrapper import PalantirWrapper 
from core.palantirModel.utils import _save_results
from core.metrics import compute_correlation, _check_macrostate_quality, compute_f1
from core.utils import save_run_results

if __name__=="__main__":
	seed = 42
	np.random.seed(seed)

	tsv_path = ...
	grid_scmultisim = {"diff_cif_fraction":[.1,.3,.5,.7,.9] , "sigma_cif" : [.1,.3,.5,.7,.9], "knn_rna":[30, 50, 70, 100], "knn_atac":[30,50,70,100], "wnn":[30,50,70,100]}
	grid_palantir = {"n_waypoints":[.25, .5, .75], "knn": [30,50,100]}
	threshold_scheme = "hard"
	n_macrostates = 5
	
	data_path =  ... 
	failures = []

	results = {"albero":"phyla5", "n_cellule":1000, "GRN_type":"GRN_100", "sigma_cif":None, "diff_cif_fraction":None, "modello":"rna", "fixed_terminal":False, "knn_atac":None, "knn_rna":None, "wnn":None, "algoritmo":"cellrank_pseudotime_kernel", "n_waypoints":None, "knn_waypoints":None, "n_macrostates":n_macrostates, "velocity_algorithm":None, "pruning_type":threshold_scheme, "pearson_kl_statistics":None, "pearson_kl_pvalue":None, "kendall_pseudotime_statistics":None, "kendall_pseudotime_pvalue":None, "pearson_entropy_statistics":None, "pearson_entropy_pvalue":None, "f1_cosine":None, "f1_euclidean":None}

	# read selected cells pseudotime kernel 
	cell_path = ... 
	with open(cell_path, "r") as f:
		cells = json.load(f)
		f.close()
	early_cell= "cell1"	

	# read ground truth fates probabilities 
	truth_path = ... 
	ground_truth = pd.read_csv(truth_path, sep= "\t", index_col=0, header=0)

	pw = PalantirWrapper()
	
	for diff_cif_fraction, sigma_cif, knn_rna, knn_atac, wnn in product(*grid_scmultisim.values()):	
		results["diff_cif_fraction"] = diff_cif_fraction
		results["sigma_cif"] = sigma_cif
		results["knn_rna"] = knn_rna
		results["knn_atac"] = knn_atac
		results["wnn"] = wnn 

		# read data and determine multiscale space
		data = mu.read_h5mu(os.path.join(data_path, f"{knn_rna}_{knn_atac}_{wnn}", f"{diff_cif_fraction}_{sigma_cif}_data.h5mu"))
		data["rna"].obsm["X_umap"] = data.obsm["X_umap"] #adjust to obtain comparable plots umap
		pw.compute_kernel(data["rna"], knn_key = "neighbors", distance_key = "distances")
		pw.run_diffusion_maps(data["rna"], seed=seed)
		pw.determine_multiscale_space(data["rna"])
	
		# create results folder
		saving_folder = os.path.join(os.getcwd(), "pseudotime_phyla5_rna", f"{diff_cif_fraction}_{sigma_cif}_{knn_rna}{knn_atac}{wnn}_rna")
		if not os.path.exists(saving_folder):
			os.mkdir(saving_folder)
		
		# iterate over palantir parameters
		for n_waypoints, knn in product(*grid_palantir.values()):
			n_waypoints = int(data.shape[0] * n_waypoints)
			results["n_waypoints"] = n_waypoints
			results["knn_waypoints"] = knn

		#### KERNEL + PALANTIR #############################################################################
			np.random.seed(seed)
			try:
			#use palantir to estimate pseudotime
				pw.run_palantir(data["rna"], early_cell = early_cell, num_waypoints = n_waypoints, knn=knn, seed=seed)	
				data.update()
				kernel = PseudotimeKernelMuon(data=data, modality_key="rna", embedding_key="X_umap", connectivity_key = "connectivities", pseudotime_key = "palantir_pseudotime", group_key=["pop"])
				kernel.compute_transition_matrix(threshold_scheme=threshold_scheme)
				analyser = MatrixAnalyser(kernel.kernel.transition_matrix, data, cluster_key="rna:pop", seed=seed)
				analyser._topology_analysis()					
			except:
				failures.append((diff_cif_fraction, sigma_cif, knn_rna, knn_atac, wnn, n_waypoints, knn, "kernel"))
				continue #go to next element
			
		#### NO TERMINAL STATES ###############################
			np.random.seed(seed)
			results["fixed_terminal"] = False
			g = cellrank.estimators.GPCCA(kernel.kernel)
			g.compute_schur()
			saving_folder_noterminal= os.path.join(saving_folder, f"no_terminal_{n_waypoints}_{knn}")
			if not os.path.exists(saving_folder_noterminal):
				os.mkdir(saving_folder_noterminal)
			try:
				g.compute_macrostates(n_states=n_macrostates, cluster_key="pop")
				g.predict_initial_states()
				g.predict_terminal_states(allow_overlap=True)
			#	g.plot_macrostate_composition(key="pop", show=False, save=os.path.join(saving_folder_noterminal, f"macrostate_composition_{n_macrostates}.png"), title=f"Macrostate Composition {n_macrostates}")
			#	g.plot_coarse_T(annotate=True, save=os.path.join(saving_folder_noterminal, f"coarseT_{n_macrostates}.png"), title =f"Macrostates {n_macrostates}")
				g.compute_fate_probabilities(tol=1e-10, use_petsc=True, preconditioner="ilu")
			#	g.plot_fate_probabilities(same_plot=True, save=os.path.join(saving_folder_noterminal, f"fateProb_no_fixed.png"), show=False)
				dataframe = pd.DataFrame(g.fate_probabilities.X, columns=g.fate_probabilities.names, index=data.obs_names)
				dataframe["entropy"] = g.compute_lineage_priming(method="entropy")	
				dataframe["KL"] = g.compute_lineage_priming(method="kl_divergence")
				dataframe["pseudotime"] = data.obs["rna:palantir_pseudotime"]
				dataframe["celltype"] = data.obs["rna:pop"]
				
				results["f1_cosine"] = None
				results["f1_euclidean"] = None

				dataframe = dataframe[["KL", "entropy", "pseudotime", "celltype"]]
				# Correlation entropy - kl -  pseudotime
				statistics, pvalue= compute_correlation(dataframe, "pearson", "pseudotime", "KL")
				results["pearson_kl_statistics"] = statistics
				results["pearson_kl_pvalue"] = pvalue

				statistics, pvalue = compute_correlation(dataframe, "pearson", "pseudotime", "entropy")
				results["pearson_entropy_statistics"] = statistics 
				results["pearson_entropy_pvalue"] = pvalue

				pd.DataFrame(results, index=[0]).to_csv(tsv_path, sep=",", header=False, index=False, mode="a")		
	
			except:
				failures.append((diff_cif_fraction, sigma_cif, knn_rna, knn_atac, wnn, n_waypoints, knn, "noterminal"))
				

		#### SET TERMINAL STATES ############################################################################
			np.random.seed(seed)
			results["fixed_terminal"] = True
			g = cellrank.estimators.GPCCA(kernel.kernel)
			g.compute_schur()
			saving_folder_terminal= os.path.join(saving_folder, f"terminal_{n_waypoints}_{knn}")
			if not os.path.exists(saving_folder_terminal):
				os.mkdir(saving_folder_terminal)
			try:
				g.set_initial_states(cells["initial"])
				g.set_terminal_states(cells["terminal"])	
				g.compute_fate_probabilities(tol=1e-10, use_petsc=True, preconditioner="ilu")
			#	g.plot_fate_probabilities(same_plot=True, save=os.path.join(saving_folder_terminal, f"fateProb_fixed.png"), title="Fate Probabilities", show=False)
				dataframe = pd.DataFrame(g.fate_probabilities.X, columns = g.fate_probabilities.names, index=data.obs_names)

				cosine, euclidean = compute_f1(dataframe[list(cells["terminal"].keys())], ground_truth[list(cells["terminal"].keys())], aggregate=True)
				results["f1_cosine"] = cosine
				results["f1_euclidean"] = euclidean
				dataframe["pseudotime"] = data.obs["rna:palantir_pseudotime"]
				dataframe["celltype"] = data.obs["rna:pop"]
				dataframe["entropy"] = g.compute_lineage_priming(method="entropy")	
				dataframe["KL"] = g.compute_lineage_priming(method="kl_divergence")
				
				dataframe = dataframe[["entropy", "pseudotime", "KL", "celltype"]]

				# Correlazione entropy - pseudotime - true pseudotime
				statistics, pvalue= compute_correlation(dataframe, "pearson", "pseudotime", "KL")
				results["pearson_kl_statistics"] = statistics
				results["pearson_kl_pvalue"] = pvalue

				statistics, pvalue = compute_correlation(dataframe, "pearson", "pseudotime", "entropy")
				results["pearson_entropy_statistics"] = statistics 
				results["pearson_entropy_pvalue"] = pvalue

				pd.DataFrame(results, index=[0]).to_csv(tsv_path, sep=",", header=False, index=False, mode="a")	

			except:
				failures.append((diff_cif_fraction, sigma_cif, knn_rna, knn_atac, wnn, n_waypoints, knn, "terminal"))

	failure_path = ... 
	pd.DataFrame(failures, columns = ["rd", "sigma", "knn_rna", "knn_atac", "wnn", "n_waypoints", "knn_waypoints", "run_type"]).to_csv(failure_path, sep="\t", index=False, header=True)
