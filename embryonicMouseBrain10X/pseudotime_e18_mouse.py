import os
import argparse
import muon as mu
import scanpy as sc
import pandas as pd
import numpy as np
import cellrank
from core.palantirModel.palantir_wrapper import PalantirWrapper 
from core.palantirModel.plots import plot_entropy 
from core.palantirModel.utils import _save_results
from core.cellrankPseudotime.pseudokernel import PseudotimeKernelMuon


if __name__=="__main__":
	seed = 42
	np.random.seed(seed)

	working_dir = "/home/scvemo"
	data_path = os.path.join(working_dir, "data", "e18_mouse")
	results_folder = os.path.join(working_dir, "results", "e18_mouse")

	#path declaration + additional files
	data = mu.read_h5mu(os.path.join(data_path, "data.h5mu"))
	data["rna"].obsm["X_umap"] = data.obsm["X_umap"]
	data["rna"].obs = data.obs[["celltype"]]

	rna_folder = os.path.join(results_folder, f"pseudotime_rna")
	if not os.path.exists(rna_folder):
		os.mkdir(rna_folder)

	multiomics_folder = os.path.join(results_folder, f"pseudotime_multiomics")
	if not os.path.exists(multiomics_folder):
		os.mkdir(multiomics_folder)

	# select initial cell 
	early_cell = np.random.choice(data[data.obs["celltype"]=="RG, Astro, OPC"].obs_names)

	# n_macrostates
	parser = argparse.ArgumentParser()
	parser.add_argument("--n_macrostates", type=int, help ="number of macrostates cellrank pseudotime")
	args = parser.parse_args()
	n_macrostates = args.n_macrostates

	#EXECUTE PALANTIR MULTIOMICS
	pw = PalantirWrapper()
	pw.compute_kernel(data["rna"], knn_key="neighbors", distance_key = "distances")
	pw.run_diffusion_maps(data["rna"], seed=seed)
	pw.determine_multiscale_space(data["rna"])

	try:
		pw.run_palantir(data["rna"], early_cell=early_cell, seed = seed) 
		kernel = PseudotimeKernelMuon(data=data, modality_key = "rna", embedding_key = "X_umap", connectivity_key="connectivities", pseudotime_key="palantir_pseudotime", group_key="celltype")
		kernel.compute_transition_matrix(threshold_scheme="hard")
		g = cellrank.estimators.GPCCA(kernel.kernel)
		g.compute_schur()
		g.compute_macrostates(n_states=n_macrostates, cluster_key="celltype")
		g.predict_terminal_states()
		g.predict_initial_states(allow_overlap=True)
		plot_path = f"macrostate_composition_{n_macrostates}.png" 
		title = f"Macrostate Composition {n_macrostates}" 
		g.plot_macrostate_composition(key="celltype", show=False, save=os.path.join(rna_folder, plot_path), title=title)
		plot_path = f"coarseT_{n_macrostates}.png" 
		title = f"Corse Grained TM {n_macrostates}"
		g.plot_coarse_T(annotate=True, save = os.path.join(rna_folder, plot_path), title= title)
		g.compute_fate_probabilities(tol=1e-10, preconditioner="ilu", n_jobs=-1, use_petsc=True, show_progress_bar=False, backend="threading")
		plot_path = f"fateProbs_{n_macrostates}.png"
		title = f"Fate Probabilities {n_macrostates}" 
		g.plot_fate_probabilities(same_plot=True, save=os.path.join(rna_folder, plot_path), title=title, show=False, color="celltype")

		dataframe = pd.DataFrame(data.obs["celltype"])
		dataframe["entropy"] = g.compute_lineage_priming(method="entropy")
		dataframe["KL"] = g.compute_lineage_priming(method="kl_divergence")
		plot_path = os.path.join(rna_folder, f"entropy_{n_macrostates}.png")
		plot_entropy(dataframe["entropy"], data.obsm["X_umap"], save=True, saving_path = plot_path)
		plot_path = os.path.join(rna_folder, f"KL_{n_macrostates}.png")
		plot_entropy(dataframe["KL"], data.obsm["X_umap"], save=True, saving_path = plot_path)

	except Exception as e:
		print(e)

	#EXECUTE PALANTIR MULTIOMICS
	pw = PalantirWrapper()
	pw.compute_kernel(data)
	pw.run_diffusion_maps(data, seed=seed)
	pw.determine_multiscale_space(data)

	try:
		pw.run_palantir(data, early_cell=early_cell, seed = seed) 
		kernel = PseudotimeKernelMuon(data=data, modality_key = None, embedding_key = "X_umap", connectivity_key="wnn_connectivities", pseudotime_key="palantir_pseudotime", group_key="celltype")
		kernel.compute_transition_matrix(threshold_scheme="hard")
		g = cellrank.estimators.GPCCA(kernel.kernel)
		g.compute_schur()
		g.compute_macrostates(n_states=n_macrostates, cluster_key="celltype")
		g.predict_terminal_states()
		g.predict_initial_states(allow_overlap=True)
		plot_path = f"macrostate_composition_{n_macrostates}.png" 
		title = f"Macrostate Composition {n_macrostates}" 
		g.plot_macrostate_composition(key="celltype", show=False, save=os.path.join(multiomics_folder, plot_path), title=title)
		plot_path = f"coarseT_{n_macrostates}.png" 
		title = f"Corse Grained TM {n_macrostates}"
		g.plot_coarse_T(annotate=True, save = os.path.join(multiomics_folder, plot_path), title= title)

		g.compute_fate_probabilities(tol=1e-10, preconditioner="ilu", n_jobs=-1, use_petsc=True, show_progress_bar=False, backend="threading")
		plot_path = f"fateProbs_{n_macrostates}.png"
		title = f"Fate Probabilities {n_macrostates}" 
		g.plot_fate_probabilities(same_plot=True, save=os.path.join(multiomics_folder, plot_path), title=title, show=False, color="celltype")

		dataframe = pd.DataFrame(data.obs["celltype"])
		dataframe["entropy"] = g.compute_lineage_priming(method="entropy")
		dataframe["KL"] = g.compute_lineage_priming(method="kl_divergence")
		plot_path = os.path.join(multiomics_folder, f"entropy_{n_macrostates}.png")
		plot_entropy(dataframe["entropy"], data.obsm["X_umap"], save=True, saving_path = plot_path)
		plot_path = os.path.join(multiomics_folder, f"KL_{n_macrostates}.png")
		plot_entropy(dataframe["KL"], data.obsm["X_umap"], save=True, saving_path = plot_path)

	except Exception as e:
		print(e)

