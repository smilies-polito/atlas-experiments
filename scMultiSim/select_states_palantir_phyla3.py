import os
import json
import muon as mu 
import numpy as np

if __name__=="__main__":
	seed = 42
	np.random.seed(seed)

	working_directory = ... #set to repository
	data_path = os.path.join(working_directory, "data", "phyla3")
	saving_path = os.path.join(data_path, "selected_cells_palantir.json")
	data = mu.read_h5mu(os.path.join(data_path, "30_30_30", "0.5_0.5.h5mu"))

	initial_states = data.obs_names[data.obs["rna:pseudotime"].argmin()]
	data.obs["is_initial"] = data.obs_names == initial_states

	terminal41 = np.random.choice(data[(data.obs["rna:pop"]=="4_1") & (data.obs["rna:pseudotime"]>.9)].obs_names)
	terminal52 = np.random.choice(data[(data.obs["rna:pop"]=="5_2") & (data.obs["rna:pseudotime"]>.9)].obs_names)
	terminal53 = np.random.choice(data[(data.obs["rna:pop"]=="5_3") & (data.obs["rna:pseudotime"]>.9)].obs_names)
	terminal_states = [terminal41, terminal52, terminal53]
	
	data.obs["is_terminal"] = data.obs_names.isin(terminal_states)
	mu.pl.embedding(data, basis="X_umap", color=["rna:pop", "is_initial", "is_terminal"])
	cells = {"initial": {"4-1-5" : initial_states}, "terminal":{"4-5-2":terminal52, "4-5-3":terminal53, "4-1":terminal41}}
	with open(saving_path, "w") as f:
		json.dump(cells, f)
			
	
