import os
import muon as mu 
import numpy as np
import scvelo as scv
from cellrank.kernels import PrecomputedKernel
from cellrank.estimators import GPCCA
from core.cellrankVelocity import TransitionMatrix
from core.matrix_analysis import MatrixAnalyser

if __name__=="__main__":
	seed = 42
	np.random.seed(seed)
	data_path = os.path.join(os.getcwd(), "phyla5", "0.5_0.5_data.h5mu")
	data = mu.read_h5mu(data_path)

	n_states = 6
	matrix_key = "transition_matrix"

	results = {}
	results_path = os.path.join(os.getcwd(), "prova")
	if not os.path.exists(results_path):
		os.mkdir(results_path)

	# Compute Velocity
	scv.pp.moments(data["rna"])
	scv.tl.recover_dynamics(data["rna"])
	scv.tl.velocity(data["rna"], mode="dynamical")
	scv.tl.velocity_graph(data["rna"])
	# Update UMAP
	data["rna"].obsm["X_umap"] = data.obsm["X_umap"]
	scv.pl.velocity_embedding_stream(data["rna"], color="pop")
	
	
################ MULTIOMCS ##########################
	# COMPUTE TRANSITION MATRIX 
	tm = TransitionMatrix(data, rna_key = "rna", atac_key="activity", velocity_key="velocity")
	tm.compute_transition_matrix()
	transition_matrix = tm.transition_matrix

	# CHECK STATUS TRANSITION MATRIX
	analyser = MatrixAnalyser(transition_matrix, data, cluster_key="rna:pop", seed = seed)
	analyser._topology_analysis()
	analyser._condensation_graph()
	results["transition_matrix"] = analyser.get_params()

	# GPCCA
	data["rna"].obsp[matrix_key]= transition_matrix
	gpcca = GPCCA(PrecomputedKernel(data["rna"], obsp_key= matrix_key))
	gpcca.compute_schur()
	gpcca.plot_spectrum(real_only=True)
	gpcca.compute_macrostates(n_states = n_states, cluster_key = "pop")
	gpcca.plot_macrostates(which="all")
	gpcca.plot_macrostate_composition(key="pop")
	gpcca.plot_coarse_T()
	gpcca.predict_terminal_states()
	gpcca.predict_initial_states(allow_overlap=True)
	gpcca.compute_fate_probabilities()
	gpcca.plot_fate_probabilities(same_plot=True)
