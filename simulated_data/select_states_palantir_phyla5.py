import os
import json
import muon as mu 
import numpy as np

if __name__=="__main__":
	seed = 42
	np.random.seed(seed)

	working_directory = ... #set to repository
	data_path = os.path.join(working_directory, "data", "phyla5")
	saving_path = os.path.join(data_path, "selected_cells_palantir.json")
	data = mu.read_h5mu(os.path.join(data_path, "30_30_30", "0.5_0.5.h5mu"))

	initial_states = data.obs_names[data.obs["rna:pseudotime"].argmin()]
	data.obs["is_initial"] = data.obs_names == initial_states

	terminal9 = np.random.choice(data[(data.obs["rna:pop"].isin(["9_4", "9_5"])) & (data.obs["rna:pseudotime"]>.7)].obs_names)
	terminal61 = np.random.choice(data[(data.obs["rna:pop"]=="6_1") & (data.obs["rna:pseudotime"]>.9)].obs_names)
	terminal82 = np.random.choice(data[(data.obs["rna:pop"]=="8_2") & (data.obs["rna:pseudotime"]>.9)].obs_names)
	terminal83 = np.random.choice(data[(data.obs["rna:pop"]=="8_3") & (data.obs["rna:pseudotime"]>.9)].obs_names)
	terminal_states = [terminal9, terminal61, terminal82, terminal83]
	
	data.obs["is_terminal"] = data.obs_names.isin(terminal_states)
	mu.pl.embedding(data, basis="X_umap", color=["rna:pop", "is_initial", "is_terminal"])
	cells = {"initial": {"6-7-1" : initial_states}, "terminal":{"6-7-9": terminal9, "6-7-8-2":terminal82, "6-7-8-3":terminal83, "6-1":terminal61}}
	with open(saving_path, "w") as f:
		json.dump(cells, f)
			
	
