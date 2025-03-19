import os
import gc
import muon as mu
import json
import cellrank
import numpy as np
import pandas as pd
from itertools import product
from core.cellrankPseudotime.pseudokernel import PseudotimeKernelMuon
from core.cellrankPseudotime.matrix_analysis import MatrixAnalyser
from core.cellrankPseudotime.utils import _check_conjugate, _compute_macrostates, _plot_quality_gpcca


if __name__ == "__main__":
	seed= 42
	np.random.seed(seed)
	quality_dict = {}
	grid = {"diff_cif_fraction":[.5], "cif_sigma":[.1]}
	
	cell_path = os.path.join(os.getcwd(), "scMultiSim", "phyla3", "phyla3_cells.json")
	with open(cell_path, "r") as f:
		cells = json.load(f)
		f.close()
	failures = [] 	
	saving_folder = os.path.join(os.getcwd(), "results")
	data_path = os.path.join(os.getcwd(), "scMultiSim", "phyla3")

	for values in product(*grid.values()):
		diff_cif_fraction, cif_sigma = values
		data = mu.read_h5mu(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_data.h5mu"))
			
		save_multiomics = os.path.join(saving_folder, f"{diff_cif_fraction}_{cif_sigma}_multiomics")
		save_rna = os.path.join(saving_folder, f"{diff_cif_fraction}_{cif_sigma}_rna")
		if not os.path.exists(save_multiomics):
			os.mkdir(save_multiomics)
		if not os.path.exists(save_rna):
			os.mkdir(save_rna)
#		try:
		# MULTIOMICS RUN
#		print("MULTIOMICS RUN")
#		kernel = PseudotimeKernelMuon(data = data, modality_key = None, embedding_key = "X_umap", connectivity_key = "wnn_connectivities", pseudotime_key = "rna:pseudotime", group_key = ["rna:pop"])
#		kernel.compute_transition_matrix(threshold_scheme="hard")
#
#		analyser = MatrixAnalyser(kernel.kernel.transition_matrix, data, cluster_key= "rna:pop", seed=seed)
#		analyser._topology_analysis()
#		analyser._condensation_graph(saving_folder = save_multiomics) 
#		analyser._save_params(os.path.join(save_multiomics, "matrix_analysis.csv"))

#		g = cellrank.estimators.GPCCA(kernel.kernel)
#		g.compute_schur()
#		g.set_initial_states(cells["initial"])
#		g.set_terminal_states(cells["terminal"])
#		g.compute_fate_probabilities(tol=1e-10, use_petsc=True, preconditioner="ilu")
#		g.plot_fate_probabilities(same_plot=True, save=os.path.join(save_multiomics, f"fateProb_set.png"), title="Fate Probabilities")
	
#		df = pd.DataFrame(g.fate_probabilities.X, columns = g.fate_probabilities.names, index=data.obs_names)
#		df['entropy'] = g.compute_lineage_priming(method="entropy")
#		df['KL'] = g.compute_lineage_priming(method="kl_divergence")
#		df["pseudotime"] = data.obs["rna:pseudotime"].copy()
#		df["celltype"] = data.obs["rna:pop"].copy()
#		df.to_csv(os.path.join(save_multiomics, f"terminal.tsv"), sep="\t", header=True, index=True)
#
			# RNA RUN
		print("RNA RUN")
		kernel = PseudotimeKernelMuon(data=data, modality_key = "rna", embedding_key = "X_umap", connectivity_key = "connectivities", pseudotime_key= "pseudotime", group_key = "pop")
		kernel.compute_transition_matrix(threshold_scheme = "hard")
	
		analyser = MatrixAnalyser(kernel.kernel.transition_matrix, data, cluster_keyt = "pop", seed = seed)
		analyser._topology_analysis()
		analyser._condensation_graph(saving_folder = save_rna)
		analyser._save_params(os.path.join(save_rna, "matrix_analysis.csv"))
	
		g = cellrank.estimators.GPCCA(kernel.kernel)
		g.compute_schur()
		g.set_initial_states(cells["initial"])
		g.set_terminal_states(cells["terminal"])
		g.compute_fate_probabilities(tol=1e-10, use_petsc=True, preconditioner="ilu")
		g.plot_fate_probabilities(same_plot=True, save=os.path.join(save_rna, f"fateProb_set.png"), title="Fate Probabilities")
		
		df = pd.DataFrame(g.fate_probabilities.X, columns = g.fate_probabilities.names, index=data.obs_names)
		df['entropy'] = g.compute_lineage_priming(method="entropy")
		df['KL'] = g.compute_lineage_priming(method="kl_divergence")
		df["pseudotime"] = data["rna"].obs["pseudotime"].copy()
		df["celltype"] = data["rna"].obs["pop"].copy()
		df.to_csv(os.path.join(save_rna, f"terminal.tsv"), sep="\t", header=True, index=True)

#		except:
#			failures.append((diff_cif_fraction, cif_sigma))
		gc.collect()
	print(failures)	
	
