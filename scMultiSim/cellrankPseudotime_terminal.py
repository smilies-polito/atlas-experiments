import os
import gc
import muon 
import cellrank
import numpy as np
import pandas as pd
from cellrankModel.pseudokernel import PseudotimeKernelMuon
from cellrankModel.matrix_analysis import MatrixAnalyser
from cellrankModel.utils import _check_conjugate, _compute_macrostates, _plot_quality_gpcca, _save_probabilities


if __name__ == "__main__":
	seed= 42
	np.random.seed(seed)
	quality_dict = {}

	saving_folder_multiomics = os.path.join(os.getcwd(), "results_pseudotime_multiomics")
	saving_folder_rna = os.path.join(os.getcwd(), "results_pseudotime_rna")
	if not os.path.exists(saving_folder_multiomics):
		os.mkdir(saving_folder_multiomics)
	if not os.path.exists(saving_folder_rna):
		os.mkdir(saving_folder_rna)

	data_path = ... 
	data = muon.read_h5mu(data_path)

	initial_state = {"4_1_1" : np.random.choice(data.obs_names[(data.obs["rna:pop"]=="4_1") & (data.obs["rna:pseudotime"]<0.2)], 30).tolist()}
#	data.obs["is_initial"] = data.obs_names.isin(initial_state)	
	terminal41 = np.random.choice(data.obs_names[(data.obs["rna:pop"]=="4_1") & (data.obs["rna:pseudotime"]>0.8)], 30).tolist()
	terminal52 = np.random.choice(data.obs_names[(data.obs["rna:pop"]=="5_2") & (data.obs["rna:pseudotime"]>0.8)], 30).tolist()
	terminal53 = np.random.choice(data.obs_names[(data.obs["rna:pop"]=="5_3") & (data.obs["rna:pseudotime"]>0.8)], 30).tolist()
	terminal_states = {"4_1_2" : terminal41, "5_2": terminal52, "5_3": terminal53}
#	data.obs["is_terminal"] = data.obs_names.isin(terminal_states)
# 	muon.pl.embedding(data, basis="X_umap", color=["is_initial", "is_terminal"], save = "selectedcells.png", title="Selected Cells", show=False)

	# MULTIOMICS RUN
	kernel = PseudotimeKernelMuon(data = data, modality_key = None, embedding_key = "X_umap", connectivity_key = "wnn_connectivities", pseudotime_key = "rna:pseudotime", group_key = ["rna:pop"])
	kernel.compute_transition_matrix(threshold_scheme="hard")

	analyser = MatrixAnalyser(kernel.kernel.transition_matrix, data, cluster_key= "rna:pop", seed=seed)
	analyser._topology_analysis()
	analyser._condensation_graph(saving_folder = saving_folder_multiomics) 
	analyser._save_params(os.path.join(saving_folder_multiomics, "matrix_analysis.csv"))

	g = cellrank.estimators.GPCCA(kernel.kernel)
	g.compute_schur()
	g.set_initial_states(initial_state)
	g.set_terminal_states(terminal_states)
	g.compute_fate_probabilities(tol = 1e-10, use_petsc=True, preconditioner="ilu")
	g.plot_fate_probabilities(same_plot = True, save= os.path.join(saving_folder_multiomics, f"fateProb_set.png"), title = "Fate Probabilities")
	
	_save_probabilities(g, barcodes=data.obs_names, saving_path = saving_folder_multiomics)
		
	# RNA RUN
	kernel = PseudotimeKernelMuon(data=data, modality_key = "rna", embedding_key = "X_umap", connectivity_key = "connectivities", pseudotime_key= "pseudotime", group_key = "pop")
	kernel.compute_transition_matrix(threshold_scheme = "hard")
	
	analyser = MatrixAnalyser(kernel.kernel.transition_matrix, data, cluster_keyt = "pop", seed = seed)
	analyser._topology_analysis()
	analyser._condensation_graph(saving_folder = saving_folder_rna)
	analyser._save_params(os.path.join(saving_folder_rna, "matrix_analysis.csv"))

	g = cellrank.estimators.GPCCA(kernel.kernel)
	g.compute_schur()
	g.set_initial_states(initial_state)
	g.set_terminal_states(terminal_states)
	g.compute_fate_probabilities()
	g.plot_fate_probabilities(same_plot=True, save=os.path.join(saving_folder_rna, f"fateProb_set.png"))
	_save_probabilities(g, barcodes = data["rna"].obs_names, saving_path = saving_folder_rna)

		
	
