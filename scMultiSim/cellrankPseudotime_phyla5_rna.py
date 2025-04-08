import os
import gc
import muon as mu
import json
import cellrank
import numpy as np
import pandas as pd
from itertools import product
from core.cellrankPseudotime.pseudokernel import PseudotimeKernelMuon
from core.matrix_analysis import MatrixAnalyser
from core.metrics import _check_macrostate_quality, compute_correlation, plot_branch_correlation, compute_f1

if __name__ == "__main__":
	seed= 42
	np.random.seed(seed)
	results = {}
	diff_cif_fraction, sigma_cif = .9, .9
	data_path = os.path.join(os.getcwd(), "phyla5")

	# read selected cells
	cell_path = os.path.join(data_path, "selected_cells_phyla5.json")
	with open(cell_path, "r") as f:
		cells = json.load(f)
		f.close()

	# read ground truth fates probabilities
	truth_path = os.path.join(data_path, "branch_assignment.tsv")
	ground_truth = pd.read_csv(truth_path, sep="\t", index_col =0, header=0)

	# create selected folder
	saving_folder = os.path.join(os.getcwd(), "results", f"{diff_cif_fraction}_{sigma_cif}_rna") 
	if not os.path.exists(saving_folder):
		os.mkdir(saving_folder)
	
	#read data a1nche compute kernel
	data = mu.read_h5mu(os.path.join(data_path, f"{diff_cif_fraction}_{sigma_cif}_data.h5mu"))
	kernel = PseudotimeKernelMuon(data = data, modality_key = "rna", embedding_key = "X_umap", connectivity_key = "connectivities", pseudotime_key = "pseudotime", group_key = "pop")
	kernel.compute_transition_matrix(threshold_scheme="hard")

	# transition matrix analysis
	analyser = MatrixAnalyser(kernel.kernel.transition_matrix, data["rna"], cluster_key= "pop", seed=seed)
	analyser._topology_analysis()
	results["transition_matrix"] = analyser.get_params()

	#### SET STATES ########################################
	np.random.seed(seed)
	g = cellrank.estimators.GPCCA(kernel.kernel)
	g.compute_schur()
	g.set_initial_states(cells["initial"])
	g.set_terminal_states(cells["terminal"])
	g.compute_fate_probabilities(tol=1e-10, use_petsc=True, preconditioner="ilu")
	g.plot_fate_probabilities(same_plot=True, save=os.path.join(saving_folder, f"fateProb_fixed.png"), title="Fate Probabilities", show=False)
	
	df = pd.DataFrame(g.fate_probabilities.X, columns = g.fate_probabilities.names, index=data.obs_names)
	df['entropy'] = g.compute_lineage_priming(method="entropy")
	df['KL'] = g.compute_lineage_priming(method="kl_divergence")
	df["pseudotime"] = data.obs["rna:pseudotime"].copy()

	# compute correlations pseudotime - entropy - kl
	results["si_terminal"] = {}
	results["si_terminal"]["spearman_entropy"] = compute_correlation(df, "pearson","pseudotime", "entropy")
	results["si_terminal"]["kendall-tau_entropy"] = compute_correlation(df, "kendall_tau", "pseudotime", "entropy")	
	
	results["si_terminal"]["spearman_KL"] = compute_correlation(df, "pearson", "pseudotime", "KL")
	results["si_terminal"]["kendall-tau_KL"] = compute_correlation(df, "kendall_tau", "pseudotime", "KL")

	# compute f1 score
	results["f1"] = {}
	for terminal in cells["terminal"].keys():
		results["f1"][terminal] = compute_f1(df[terminal], ground_truth[terminal])

	results_df = df[["entropy", "KL", "pseudotime"]]
	results_df.columns = ["fixed_entropy", "fixed_kl", "pseudotime"]
	
	#### NO SET STATES #####################################
	n_states = 5
	np.random.seed(seed)
	g = cellrank.estimators.GPCCA(kernel.kernel)
	g.compute_schur()
	g.compute_macrostates(n_states=n_states, cluster_key="pop")
	g.predict_initial_states()
	g.predict_terminal_states(allow_overlap=True)
	g.plot_macrostate_composition(key="pop", show=False, save = os.path.join(saving_folder, f"macrostate_composition_{n_states}.png"), title= f"Macrostate Composition {n_states}")
	g.plot_coarse_T(annotate=True, save = os.path.join(saving_folder, f"coarseT_{n_states}.png"), title= f"Coarse Transition Matrix {n_states}")
	g.plot_macrostates(which="all", legend_loc="right", s=100, show=False, save=os.path.join(saving_folder, f"macrostates_no_fixed_{n_states}.png"), title=f"Macrostates {n_states}")
	g.compute_fate_probabilities(tol=1e-10, use_petsc = True, preconditioner="ilu")
	g.plot_fate_probabilities(same_plot=True, save=os.path.join(saving_folder, f"fateProb_no_fixed.png"), title="Fate Probabilities", show=False)

	# RESULTS
	results["macrostate_quality"] = _check_macrostate_quality(g, n_states)

	df = pd.DataFrame(g.fate_probabilities.X, columns = g.fate_probabilities.names, index=data.obs_names)
	df['entropy'] = g.compute_lineage_priming(method="entropy")
	df['KL'] = g.compute_lineage_priming(method="kl_divergence")
	df["pseudotime"] = data.obs["rna:pseudotime"].copy()

	results_df["entropy"] = df["entropy"]
	results_df["kl"] = df["KL"]
	results_df["celltype"] = data.obs["rna:pop"]
	
	# compute correlations pseudotime - entropy - kl
	results["no_terminal"] = {}
	results["no_terminal"]["spearman_entropy"] = compute_correlation(df, "pearson","pseudotime", "entropy")
	results["no_terminal"]["kendall-tau_entropy"] = compute_correlation(df, "kendall_tau", "pseudotime", "entropy")	
	
	results["no_terminal"]["spearman_KL"] = compute_correlation(df, "pearson", "pseudotime", "KL")
	results["no_terminal"]["kendall-tau_KL"] = compute_correlation(df, "kendall_tau", "pseudotime", "KL")

	path = os.path.join(saving_folder, "results.json")
	with open(path, "w") as f:
		json.dump(results, f)
		f.close() 

	path = os.path.join(saving_folder, "results.tsv")
	results_df.to_csv(path, sep="\t", header=True, index=True)	
