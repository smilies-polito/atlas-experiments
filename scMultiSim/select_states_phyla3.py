import os
import json
import muon as mu 
import numpy as np

if __name__=="__main__":
	seed = 42
	np.random.seed(seed)
	working_directory = ...
	data_path = os.path.join(working_directory, "data", "phyla3")
	saving_path = os.path.join(data_path, "selected_cells.json")
	data = mu.read_h5mu(os.paht.join(data_path, "30_30_30", "0.5_0.5.h5mu"))

	initial_states = data.obs["rna:pseudotime"].sort_values().index.tolist()[:30]
	data.obs["is_initial"] = data.obs_names.isin(initial_states)
	terminal41 = data[data.obs["rna:pop"]=="4_1"].obs["rna:pseudotime"].sort_values(ascending=False).index.tolist()[:30]
	terminal53 = data[data.obs["rna:pop"]=="5_3"].obs["rna:pseudotime"].sort_values(ascending=False).index.tolist()[:30] 
	terminal52 = data[data.obs["rna:pop"]=="5_2"].obs["rna:pseudotime"].sort_values(ascending=False).index.tolist()[:30] 
	terminal_states = terminal41 + terminal53 + terminal52
	data.obs["is_terminal"] = data.obs_names.isin(terminal_states)
	mu.pl.embedding(data, basis="X_umap", color=["rna:pop", "is_initial", "is_terminal"])
	cells = {"initial": {"4-1-5" : initial_states}, "terminal":{"4-1": terminal41, "4-5-2":terminal52, "4-5-3":terminal53}}
	with open(saving_path, "w") as f:
		json.dump(cells, f)
			
	
