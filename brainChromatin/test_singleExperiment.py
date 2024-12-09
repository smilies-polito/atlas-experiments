import os
import muon as mu 
import numpy as np
import pandas as pd
import cellrank as cr 
from scipy.sparse import csr_matrix
from transitionMatrix import TransitionMatrix #note that in gitlab transitionMatrix.py not in same folder as this file. Should be moved


def check_conjugate(eigenvalue):
	return np.iscomplex(eig)


if __name__=="__main__":
	seed = 52
	
	os.chdir("...")
	experiment = "" #set experiment
	tm_key = "transition_matrix"
	data = mu.read_h5mu(os.path.join(os.getcwd(), experiment, "dev_clusters", "data.h5mu"))
	annotations = pd.read_csv(os.path.join(os.getcwd(), experiment,"dev_clusters", "promoter_annotations.tsv"), sep="\t", header=0, index_col=0)

	tm = TransitionMatrix(data=data, annotations=annotations, velo_modality = "rna", velocity_key= "velocity", gene_key="gene", peak_key="peak")
	tm.compute_transition_matrix()
	data.obsp[tm_key] = tm.transition_matrix
	
	# Brutto ma necessario dato che posso soltanto usare AnnData object con CellRank. TO DO: implementare MultiOmic Kernel
	data["atac"].obsm["X_umap"] = data.obsm["X_umap"]

	kernel = cr.kernels.PrecomputedKernel(object=tm.transition_matrix, adata=data["atac"])
	start_ixs = data.obs[data.obs["atac:Cluster.Name"]=="RG"].index
	kernel.plot_random_walks(start_ixs = start_ixs, n_sims = 1, seed=seed, save = f"rw_{experiment}.png")

	print(data.obs.columns)

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
			g.plot_macrostates(which="initial", legend_loc="right", s=100, show = True, 
					title = f"{experiment}, {ns} macrostates",
					save = f"states_{experiment}_{ns}.png")
			g.plot_macrostates(which="terminal", legend_loc="right", s=100)
			g.compute_fate_probabilities(tol=1e-10, use_petsc=True, preconditioner = "ilu")
			g.plot_fate_probabilities(same_plot= True, title=f"{experiment}, {ns} macrostates", save=f"fate_{experiment}_{ns}.png", show=True)
			
			fate_path = os.path.join(os.getcwd(), experiment, f"fates_{ns}.tsv")
			df = pd.DataFrame(g.fate_probabilities.X, columns =g.fate_probabilities.names, index=data.obs_names)
			df["entropy"] = g.compute_lineage_priming(method="entropy")
			df["KL div"] = g.compute_lineage_priming(method="kl_divergence")
			# aggiungere colonna pseudotime 
			df.to_csv(fate_path)
		except ValueError as e:
			print(e)	
