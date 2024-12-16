import os
import muon as mu
import scanpy as sc
import numpy as np
import pandas as pd
import cellrank as cr 
from scipy.sparse import csr_matrix
from transitionMatrix import TransitionMatrix #note that in gitlab transitionMatrix.py not in same folder as this file. Should be moved


def check_conjugate(eigenvalue):
	return np.iscomplex(eig)


if __name__=="__main__":
	seed = 52
	
	os.chdir("/Users/lrcq/Documents/devtraj/preprocessing/brainChromatinGreenLeaf")
	experiment = "dc2r2_r2" #set experiment
	tm_key = "transition_matrix"
	data = mu.read_h5mu("/Users/lrcq/Documents/devtraj/preprocessing/brainChromatinGreenLeaf/dc1r3_r1/data.h5mu")	
	data["rna"].obsm["X_umap"] = data.obsm["X_umap"] #brutto ma anche qui necessario	
	tm = TransitionMatrix(data=data, velocity_key= "velocity") 
	tm.compute_transition_matrix()
	data.obsp[tm_key] = tm.transition_matrix
	
	kernel = cr.kernels.PrecomputedKernel(object=tm.transition_matrix, adata=data["rna"])
	start_ixs = data.obs[data.obs["rna:Cluster.Name"]=="RG"].index
	kernel.plot_random_walks(start_ixs = start_ixs, n_sims = 1, seed=seed, save = f"rw_{experiment}_1.png")
	kernel.plot_random_walks(start_ixs = start_ixs, n_sims = 1, seed=seed+50, save = f"rw_{experiment}_2.png")
	kernel.plot_random_walks(start_ixs = start_ixs, n_sims = 1, seed=seed//2, save = f"rw_{experiment}_3.png")


	g = cr.estimators.GPCCA(kernel)
	g.compute_schur()
	
	cell_type_key = "Cluster.Name"
	eigenvalues = g.eigendecomposition["D"]
		
	for ns in range(2,5):
		try:	
			g.compute_macrostates(n_states = ns, cluster_key=cell_type_key)
			g.predict_initial_states()
			g.predict_terminal_states(allow_overlap=True)
			g.plot_macrostate_composition(key = cell_type_key, show=False, title=f"{experiment}, {ns} macrostates",
					save =f"composition_{experiment}_{ns}.png")
			g.plot_macrostates(which="initial", legend_loc="right", s=100, show=False,
					title = f"Initial {experiment}, {ns} macrostates",
					save = f"initialstates_{experiment}_{ns}.png")
			g.plot_macrostates(which="terminal", legend_loc="right", s=100, show=False,
					title = f"Terminal {experiment}, {ns} macrostates", 
					save = f"terminalstates_{experiment}_{ns}.png")
			g.compute_fate_probabilities(tol=1e-10, use_petsc=True, preconditioner = "ilu")
			g.plot_fate_probabilities(same_plot= True, title=f"{experiment}, {ns} macrostates", save=f"fate_{experiment}_{ns}.png", show=False)
			
			fate_path = os.path.join(os.getcwd(), experiment, f"fates_{ns}.tsv")
			df = pd.DataFrame(g.fate_probabilities.X, columns =g.fate_probabilities.names, index=data.obs_names)
			df["entropy"] = g.compute_lineage_priming(method="entropy")
			df["KL div"] = g.compute_lineage_priming(method="kl_divergence")
			# aggiungere colonna pseudotime 
			# df.to_csv(fate_path)
		except ValueError as e:
			print(e)	
