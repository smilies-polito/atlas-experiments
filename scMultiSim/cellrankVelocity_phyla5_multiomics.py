import os
import json
import muon as mu 
import numpy as np
import pandas as pd
import scvelo as scv
from cellrank.kernels import PrecomputedKernel
from cellrank.estimators import GPCCA
from core.cellrankVelocity import TransitionMatrix
from core.matrix_analysis import MatrixAnalyser
from core.metrics import _check_macrostate_quality, compute_correlation, compute_f1

if __name__=="__main__":
	seed = 42
	np.random.seed(seed)
	matrix_key = "transition_matrix"
	results = {}
	diff_cif_fraction, sigma_cif = ...

	data_path = ... 
	
	#read selected cells 
	cell_path = ... 
	with open(cell_path, "r") as f:
		cells = json.load(f)
		f.close()

	# read ground truth fates probabilities
	truth_path = ... 
	ground_truth = pd.read_csv(truth_path, sep="\t", index_col=0, header=0)

	#create saving folder
	saving_folder = ... 
	if not os.path.exists(saving_folder):
		os.mkdir(saving_folder)

	# read data and compute velocity
	data = mu.read_h5mu(os.path.join(data_path, f"{diff_cif_fraction}_{sigma_cif}_data.h5mu"))
	
	scv.pp.moments(data["rna"])
	scv.tl.recover_dynamics(data["rna"], n_jobs=-1)
	scv.tl.velocity(data["rna"], mode="dynamical", n_jobs=-1)
	scv.tl.velocity_graph(data["rna"])
	data["rna"].obsm["X_umap"] = data.obsm["X_umap"] 
	scv.pl.velocity_embedding_stream(data["rna"], color="pop", show=False, save=f"{diff_cif_fraction}_{sigma_cif}.png")
	
	
	# compute transition matrix + analysis 
	tm = TransitionMatrix(data, rna_key = "rna", atac_key="activity", velocity_key="velocity")
	tm.compute_transition_matrix()
	transition_matrix = tm.transition_matrix

	analyser = MatrixAnalyser(transition_matrix, data, cluster_key="rna:pop", seed = seed)
	analyser._topology_analysis()
	results["transition_matrix"] = analyser.get_params()

	#### SET STATES #########################################
	np.random.seed(seed)
	data["rna"].obsp[matrix_key]= transition_matrix
	gpcca = GPCCA(PrecomputedKernel(data["rna"], obsp_key= matrix_key))
	gpcca.compute_schur()
	gpcca.set_initial_states(cells["initial"])
	gpcca.set_terminal_states(cells["terminal"])
	gpcca.compute_fate_probabilities(tol = 1e-3, use_petsc=True, preconditioner = "ilu")
	gpcca.plot_fate_probabilities(same_plot=True, save = os.path.join(saving_folder, f"fateProb_fixed.png"), title="Fate Probabilities", show = False)

	df = pd.DataFrame(gpcca.fate_probabilities.X, columns = gpcca.fate_probabilities.names, index = data.obs_names)
	df["entropy"] = gpcca.compute_lineage_priming(method="entropy")
	df["KL"] = gpcca.compute_lineage_priming(method="kl_divergence")
	df["pseudotime"] = data.obs["rna:pseudotime"].copy()

	#compute correlations pseudotime - entropy - kl
	results["si_terminal"] = {}
	results["si_terminal"]["pearson_entropy"] = compute_correlation(df, "pearson", "pseudotime", "entropy")
	results["si_terminal"]["kendall-tau_entropy"] = compute_correlation(df, "kendall_tau", "pseudotime", "entropy")
	results["si_terminal"]["pearson_KL"] = compute_correlation(df, "pearson", "pseudotime", "KL")
	results["si_terminal"]["kendall-tau_KL"] = compute_correlation(df, "kendall_tau", "pseudotime", "KL")

	# compute f1 score
	results["f1"] = {}
	for terminal in cells["terminal"].keys():
		results["f1"][terminal] = compute_f1(df[terminal], ground_truth[terminal])

	results_df = df[["entropy", "KL", "pseudotime"]]
	results_df.columns = ["fixed_entropy", "fixed_kl", "pseudotime"]

	### NO SET STATES #################################
	n_states = 5
	np.random.seed(seed)
	gpcca = GPCCA(PrecomputedKernel(data["rna"], obsp_key= matrix_key))
	gpcca.compute_schur()
	gpcca.compute_macrostates(n_states = n_states, cluster_key = "pop")
	gpcca.predict_initial_states()
	gpcca.predict_terminal_states(allow_overlap=True)
	gpcca.plot_macrostate_composition(key="pop", show=False, save = os.path.join(saving_folder, f"macrostate_composition_{n_states}.png"), title=f"Macrostate Composition {n_states}")
	gpcca.plot_coarse_T(annotate=True, save = os.path.join(saving_folder, f"coarseT_{n_states}.png"), title=f"Coarse Transition Matrix {n_states}")
	gpcca.plot_macrostates(which="all", legend_loc="right", s=100, show=False, save=os.path.join(saving_folder, f"macrostates_no_fixed_{n_states}.png"), title=f"Macrostates {n_states}")
	gpcca.compute_fate_probabilities(tol=1e-10, use_petsc=True, preconditioner="ilu")
	gpcca.plot_fate_probabilities(same_plot=True, save=os.path.join(saving_folder, f"fateProb_no_fixed.png"), title="Fate Probabilities", show=False)
	
	# RESULTS
	results["macrostate_quality"] = _check_macrostate_quality(gpcca, n_states)
	
	df = pd.DataFrame(gpcca.fate_probabilities.X, columns = gpcca.fate_probabilities.names, index = data.obs_names)
	df["entropy"] = gpcca.compute_lineage_priming(method="entropy")
	df["KL"]=gpcca.compute_lineage_priming(method="kl_divergence")
	df["pseudotime"] = data.obs["rna:pseudotime"].copy()

	results_df["entropy"] = df["entropy"]
	results_df["kl"] = df["KL"]
	results_df["celltype"] = data.obs["rna:pop"]
	
	# compute correlations pseudotime - entropy - kl
	results["no_terminal"] = {}
	results["no_terminal"]["pearson_entropy"] = compute_correlation(df, "pearson", "pseudotime", "entropy")
	results["no_terminal"]["kendall-tau_entropy"]=compute_correlation(df, "kendall_tau", "pseudotime", "entropy")
	
	results["no_terminal"]["pearson_KL"] = compute_correlation(df, "pearson", "pseudotime", "KL")
	results["no_terminal"]["kendall-tau_KL"] = compute_correlation(df, "kendall_tau", "pseudotime", "KL")
	
	path = os.path.join(saving_folder, "results.json")
	with open(path, "w") as f:
		json.dump(results,f)
		f.close()

	path = os.path.join(saving_folder, "results.tsv")
	results_df.to_csv(path, sep="\t", header=True, index=True)
	
