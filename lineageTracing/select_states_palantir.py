import os
import json
import muon as mu
import numpy as np
import pandas as pd

if __name__=="__main__":
	seed = 42
	np.random.seed(seed)
	working_dir = ... # set repository 
	donor= ... #wither donor1 or donor2
	data_path = os.path.join(working_dir, "data", "lineage_tracing", f"{donor}", "data.h5mu")
	saving_path = os.path.join(working_dir, "data", "lineage_tracing", f"{donor}", "selected_cells_palantir.json")
	data = mu.read_h5mu(data_path)

	cells = {}
	
	# select initial cell 
	cells["initial"]= {"hsc" : np.random.choice(data.obs[data.obs["STD.CellType"]=="HSC"].index)}
	terminal_erythroid = np.random.choice(data.obs[data.obs["STD.CellType"] == "EryP"].index)
	terminal_megakaryocyte = np.random.choice(data.obs[data.obs["STD.CellType"] == "MKP"].index)
	terminal_monocyte = np.random.choice(data.obs[data.obs["STD.CellType"] == "Mono"].index)
	terminal_NK = np.random.choice(data.obs[data.obs["STD.CellType"] == "NK"].index)
	dendritic = ["cDC", "pDC"]
	terminal_dendritic = np.random.choice(data.obs[data.obs["STD.CellType"].isin(dendritic)].index)
	T_cells = ["CD4", "CD8"]
	terminal_T = np.random.choice(data.obs[data.obs["STD.CellType"].isin(T_cells)].index)
	B_cells = ["B"]
	terminal_B = np.random.choice(data.obs[data.obs["STD.CellType"].isin(B_cells)].index)

	cells["terminal"] = {"monocyte": terminal_monocyte,
			     "NK": terminal_NK,
			     "B": terminal_B,
			     "T": terminal_T, 
			     "erythroid": terminal_erythroid, 
			     "megakaryocyte": terminal_megakaryocyte, 
			     "dendritic": terminal_dendritic}

	data.obs["is_selected"] = data.obs_names.isin(list(cells["terminal"].values()) + list(cells["initial"].values()))

	mu.pl.embedding(data, basis="X_umap", color="is_selected")
	with open(saving_path, "w") as f:
		json.dump(cells,f)

