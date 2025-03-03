import os, muon 
import cellrank
import numpy as np
import pandas as pd
from cellrankModel.pseudokernel import PseudotimeKernelMuon
from cellrankModel.matrix_analysis import MatrixAnalyser
from cellrankModel.utils import _check_conjugate, _compute_macrostates, _plot_quality_gpcca

if __name__ == "__main__":
	seed= 42
	_quality_dict = {}
	data_results = os.path.join(os.getcwd(), "test_results")
	data = muon.read_h5mu(os.path.join(os.getcwd(), "palantir_500_30.h5mu"))
	kernel = PseudotimeKernelMuon(data = data, modality_key = None, embedding_key = "X_umap", connectivity_key = "wnn_connectivities", pseudotime_key = "palantir_pseudotime", group_key = ["rna:pop"])
	kernel.compute_transition_matrix(threshold_scheme="hard")


	analyser = MatrixAnalyser(kernel.kernel.transition_matrix, data, cluster_key= "rna:pop", seed=seed)
	analyser._topology_analysis()
	analyser._condensation_graph(saving_folder = data_results) 
	analyser._save_params(os.path.join(data_results, "matrix.csv"))

	g = cellrank.estimators.GPCCA(kernel.kernel)
	g.compute_schur()
	g.plot_spectrum(real_only=True)
	eigenvalues = g.eigendecomposition["D"]
	print(eigenvalues)

	ns = 2
	idx = _check_conjugate(ns, eigenvalues)
	try:
		_quality_dict[idx] = _compute_macrostates(n_states = idx, gpcca = g, plot=True, save_fate=True,
					 barcodes = data.obs_names, cell_type_key = "rna:pop", saving_path = data_results)
	except Exception as e:
		print(e)

	df = pd.DataFrame(_quality_dict, index = ["spectral_gap", "minChi", "crispness"]).T
	_plot_quality_gpcca(df, saving_path = data_results)
	
