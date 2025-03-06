import os
import muon as mu
import numpy as np
import pandas as pd
from core.palantirModel.plots import plot_pseudotime, plot_entropy, plot_probabilities, plot_palantir_results, pseudotime_correlation, entropy_correlation
from core.palantirModel.utils import _check_keys
from core.utils import simple_scatter, simple_heatmap


if __name__=="__main__":

	
	# TESTING PLOTS ENTROPY ON EMBEDDING
	color = pd.Series(np.random.uniform(0, 1, 120))
	embedding = np.random.uniform(1, 100, (120,2))
	save = True
	saving_folder = os.path.join(os.getcwd(), "figure_pseudotime")
	if not os.path.exists(saving_folder):
		os.mkdir(saving_folder)

	plot_pseudotime(pseudotime = color, embedding=embedding, save=save, saving_path = saving_folder)

	# TESTING PLOT PSEUDOTIME ON EMBEDDING
	color = pd.Series(np.random.uniform(0, 1, 120))
	embedding = np.random.uniform(1, 100, (120,2))
	save = True
	saving_folder = os.path.join(os.getcwd(), "figure_entropy")
	if not os.path.exists(saving_folder):
		os.mkdir(saving_folder)

	plot_entropy(entropy = color, embedding=embedding, save=save, saving_path = saving_folder)

	# TESTING FATES ON EMBEDDING NO TERMINAL STATES
	saving_folder = os.path.join(os.getcwd(), "figure_fates")
	if not os.path.exists(saving_folder):
		os.mkdir(saving_folder)

	fates = pd.DataFrame({})
	terminal_states = fates.columns.values
	terminal_indices = np.where(fates.index.isin(terminal_states))
	terminal_locations = None
	plot_probabilities(fates=fates, terminal_states = terminal_states, terminal_locations = terminal_locations, embedding= embedding, save=save, saving_path = saving_folder)


	# TESTING FATES ON EMBEDDING TERMINAL STATES 
	fates = pd.DataFrame({"cell1": np.random.uniform(0,1,120)})
	fates["cell2"] = 1 - fates["cell1"] 
	fates.index = ["cell" + str(i) for i in range(120)]
	terminal_states = fates.columns
	terminal_indices = np.where(fates.index.isin(terminal_states))[0]
	terminal_locations = embedding[terminal_indices, :]
	plot_probabilities(fates = fates, terminal_states = terminal_states, terminal_locations = terminal_locations, embedding=embedding, save=save, saving_path= saving_folder)


	# TESTING PLOT_PALANTIR_RESULTS ON MUON DATA
	data = mu.read_h5mu( ... )
	modality_key = None
	pseudo_time_key = "palantir_pseudotime"
	embedding_key = "X_umap"
	entropy_key = "entropy"
	fate_prob_key = "fates"
	saving_folder = os.path.join(os.getcwd(), "palantir_muon")
	if not os.path.exists(saving_folder):
		os.mkdir(saving_folder)
	
	data.obs[pseudo_time_key]=np.random.uniform(0,1, len(data.obs_names))
	data.obs[entropy_key]=np.random.uniform(0,1, len(data.obs_names))
	cells = np.random.choice(data.obs_names, 2)
	fates = pd.DataFrame(np.random.uniform(0, 1, (len(data.obs_names), 2)), columns = cells, index = data.obs_names)
	data.obsm[fate_prob_key] = fates

	plot_palantir_results(data, modality_key = modality_key, embedding_key = embedding_key, pseudo_time_key = pseudo_time_key, entropy_key = entropy_key, fate_prob_key = fate_prob_key, save=save, saving_path = saving_folder)

	
	# TEST PLOT PSEUDOTIME CORRELATION ON MUON DATA
	pseudotime_correlation(data= data, modality_key=modality_key, true_key = "rna:pseudotime", infer_key = pseudo_time_key, save=True, saving_path = saving_folder, group_key = "rna:pop")

	# TEST PLOT ENTROPY CORRELATION ON MUON DATA
	entropy_correlation(data=data, modality_key=modality_key, pseudo_time_key = "rna:pseudotime", entropy_key = entropy_key, save=True, saving_path = saving_folder, group_key= "rna:pop")


	# TESTING PLOT_PALANTIR_RESULTS ON ANNDATA 
	data = mu.read_h5mu( ... )
	modality_key = "rna"
	pseudo_time_key = "palantir_pseudotime"
	embedding_key = "X_umap"
	entropy_key = "entropy"
	fate_prob_key = "fates"
	saving_folder = os.path.join(os.getcwd(), "palantir_rna")
	if not os.path.exists(saving_folder):
		os.mkdir(saving_folder)

	data[modality_key].obs[pseudo_time_key]=np.random.uniform(0,1, len(data.obs_names))
	data[modality_key].obs[entropy_key]=np.random.uniform(0,1, len(data.obs_names))
	cells = np.random.choice(data.obs_names, 2)
	fates = pd.DataFrame(np.random.uniform(0, 1, (len(data.obs_names), 2)), columns = cells, index = data.obs_names)
	data[modality_key].obsm[fate_prob_key] = fates
	data.obsm[fate_prob_key] = fates

	# TEST PLOT PSEUDOTIME CORRELATION ON ANNDATA
	pseudotime_correlation(data= data, modality_key=modality_key, true_key = "pseudotime", infer_key = pseudo_time_key, save=True, saving_path = saving_folder, group_key = "pop")

	# TEST PLOT ENTROPY CORRELATION ON ANNDATA
	entropy_correlation(data=data, modality_key=modality_key, pseudo_time_key = "pseudotime", entropy_key = entropy_key, save=True, saving_path = saving_folder, group_key= "pop")


	# TEST HEATMAP PLOTS
	saving_folder = os.path.join(os.getcwd(), "heatmaps")
	if not os.path.exists(saving_folder):
		os.mkdir(saving_folder)

	correlations1 = np.random.uniform(0, 1, (3,4))
	correlations2 = correlations1 + 1
	mask = np.random.choice([0,1], (3,4)).astype(bool)
	path1 = os.path.join(saving_folder, "heatmap1.png")
	path2 = os.path.join(saving_folder, "heatmap2.png")
	cmap_range = (0,2)	
	simple_heatmap(correlations1, mask, annot=True, saving_path = path1, cmap_range=cmap_range, title = "Heatmap1", xlabel= "X", ylabel="Y", cmap_label="label")
	simple_heatmap(correlations2, mask, annot=True, saving_path = path2, cmap_range= cmap_range) 





