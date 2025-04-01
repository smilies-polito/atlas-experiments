import os
import gc
import muon as mu
import cellrank
import numpy as np
import pandas as pd
from itertools import product
from core.cellrankPseudotime.pseudokernel import PseudotimeKernelMuon
from core.cellrankPseudotime.matrix_analysis import MatrixAnalyser
from core.cellrankPseudotime.utils import _check_conjugate, _compute_macrostates, _plot_quality_gpcca


def _helper(g, quality_dict, model, barcodes, cell_type_key, saving_path, plot=True, save_fate=True):
	failed = []
	eigenvalues = g.eigendecomposition["D"]
	idx = 3
	while idx < 7:
		idx = _check_conjugate(idx, eigenvalues)
		try:
			quality_dict[(model, idx)] = _compute_macrostates(n_states = idx, gpcca = g, plot=plot, save_fate= save_fate,
									barcodes = barcodes, cell_type_key = cell_type_key, saving_path = saving_path)

		except Exception as e:
			failed.append((idx,e))
	return failed	
	

if __name__ == "__main__":
	seed= 42
	np.random.seed(seed)
	quality_dict = {}
	grid = {"diff_cif_fraction":[.1,.3,.5,.7,.9], "cif_sigma":[.1,.3,.5,.7,.9]}
 	
	saving_folder = os.path.join(os.getcwd(), "results")
	data_path = ...

	for values in product(*grid.values()):
		diff_cif_fraction, cif_sigma = values
		data = mu.read_h5mu(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_data.h5mu"))
			
		save_multiomics = os.path.join(saving_folder, f"{diff_cif_fraction}_{cif_sigma}_multiomics")
		save_rna = os.path.join(saving_folder, f"{diff_cif_fraction}_{cif_sigma}_rna")
		if not os.path.exists(save_multiomics):
			os.mkdir(save_multiomics)
		if not os.path.exists(save_rna):
			os.mkdir(save_rna)

		# MULTIOMICS RUN
		kernel = PseudotimeKernelMuon(data = data, modality_key = None, embedding_key = "X_umap", connectivity_key = "wnn_connectivities", pseudotime_key = "rna:pseudotime", group_key = ["rna:pop"])
		kernel.compute_transition_matrix(threshold_scheme="hard")

		analyser = MatrixAnalyser(kernel.kernel.transition_matrix, data, cluster_key= "rna:pop", seed=seed)
		analyser._topology_analysis()
		analyser._condensation_graph(saving_folder = save_multiomics) 
		analyser._save_params(os.path.join(save_multiomics, "matrix_analysis.csv"))

		g = cellrank.estimators.GPCCA(kernel.kernel)
		g.compute_schur()
#		g.plot_spectrum(real_only=True)
		eigenvalues = g.eigendecomposition["D"]
		multiomics_failed = _helper(g=g, quality_dict=quality_dict, model="multiomics", barcodes = data.obs_names, cell_type_key = "rna:pop", saving_path = save_multiomics)

		# RNA RUN
		kernel = PseudotimeKernelMuon(data=data, modality_key = "rna", embedding_key = "X_umap", connectivity_key = "connectivities", pseudotime_key= "pseudotime", group_key = "pop")
		kernel.compute_transition_matrix(threshold_scheme = "hard")
	
		analyser = MatrixAnalyser(kernel.kernel.transition_matrix, data, cluster_keyt = "pop", seed = seed)
		analyser._topology_analysis()
		analyser._condensation_graph(saving_folder = save_rna)
		analyser._save_params(os.path.join(save_rna, "matrix_analysis.csv"))

		g = cellrank.estimators.GPCCA(kernel.kernel)
		g.compute_schur()
#		g.plot_spectrum(real_only=True)
		eigenvalues = g.eigendecomposition["D"]
		rna_failed = _helper(g=g, quality_dict=quality_dict, model="rna", barcodes = data["rna"].obs_names, cell_type_key="pop", saving_path = save_rna)
	
		df = pd.DataFrame(quality_dict, columns = ["spectral_gap", "minChi", "crispness"])
		_plot_quality_gpcca(df, saving_path = save_multiomics)
		print(diff_cif_fraction, cif_sigma, multiomics_failed, rna_failed)
		gc.collect()	
	
