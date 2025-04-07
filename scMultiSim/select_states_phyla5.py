import os
import json
import muon as mu 
import numpy as np

if __name__=="__main__":
	seed = 42
	data_path = ... 
	saving_path = ... 
	data = mu.read_h5mu(data_path)

	initial_states = data.obs["rna:pseudotime"].sort_values().index.tolist()[:30]
	data.obs["is_initial"] = data.obs_names.isin(initial_states)
	terminal9 = data[(data.obs["rna:pop"]=="9_4") | (data.obs["rna:pop"]=="9_5")].obs["rna:pseudotime"].sort_values(ascending=False).index.tolist()[:30]
	terminal61 = data[data.obs["rna:pop"]=="6_1"].obs["rna:pseudotime"].sort_values(ascending=False).index.tolist()[:30]
	terminal82 = data[data.obs["rna:pop"]=="8_2"].obs["rna:pseudotime"].sort_values(ascending=False).index.tolist()[:30]
	terminal83 = data[data.obs["rna:pop"]=="8_3"].obs["rna:pseudotime"].sort_values(ascending=False).index.tolist()[:30]
	terminal_states = terminal9 + terminal61 + terminal82 + terminal83
	data.obs["is_terminal"] = data.obs_names.isin(terminal_states)
	mu.pl.embedding(data, basis="X_umap", color=["rna:pop", "is_initial", "is_terminal"])
	cells = {"initial": {"6-7-1" : initial_states}, "terminal":{"6-7-9": terminal9, "6-7-8-2":terminal82, "6-7-8-3":terminal83, "6-1":terminal61}}
	with open(saving_path, "w") as f:
		json.dump(cells, f)
			
	
