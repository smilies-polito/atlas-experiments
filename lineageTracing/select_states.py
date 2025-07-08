import os
import json 
import muon as mu
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

def get_cell_neighborhood(barcode_idx: str, connectivity_matrix: csr_matrix) -> list:
	neighbors= connectivity_matrix[barcode_idx].tocoo()
	neighbors_connections = neighbors.data
	neighbors_idx = neighbors.col 
	top_k = min(len(neighbors_connections), 30)
	sorted_idx =  np.argsort(neighbors_connections)[::-1]
	neighbors_idx = neighbors_idx[sorted_idx][:top_k]
	return neighbors_idx

def get_cells(data:mu.MuData, celltypes:list):
	barcode = np.random.choice(data.obs[data.obs["STD.CellType"].isin(celltypes)].index) # select initial cell
	barcode_idx =  data.obs_names.get_loc(barcode)
	neighbors_idx = get_cell_neighborhood(barcode_idx, data.obsp["wnn_connectivities"])
	return [barcode] + list(data.obs_names[neighbors_idx])

if __name__=="__main__":
	seed = 42
	np.random.seed(seed)

	working_dir = ... #repository path 
	donor =  ...  # either donor2 or donor1
	data_path = os.path.join(working_dir, "data", "lineage_tracing", f"{donor}")
	
	data = mu.read_h5mu(os.path.join(data_path, "data.h5mu"))
	connectivity_matrix = data.obsp["wnn_connectivities"]

	# SELECT INITIAL NEIGHBORHOOD	
	initial_states = get_cells(data, ["HSC", "Refined.HSC"])
	
	# SELECT ERYTHROID
	terminal_erythroid = get_cells(data, ["EryP"])
	terminal_megakaryocyte = get_cells(data, ["MKP"])
	terminal_monocyte = get_cells(data, ["Mono"])
	terminal_NK = get_cells(data, ["NK"])
	terminal_B = get_cells(data, ["B"])
	dendritic = ["cDC", "pDC"]
	terminal_dendritic= get_cells(data, dendritic)
	T_cells = ["CD4", "CD8"]
	terminal_T = get_cells(data, T_cells)

	cells = {"initial": {"hsc": initial_states}, "terminal": {"erythroid": terminal_erythroid, "megakaryocyte":terminal_megakaryocyte, "monocyte": terminal_monocyte, "T": terminal_T, "B": terminal_B, "NK": terminal_NK, "dendritic": terminal_dendritic}}
	with open(os.path.join(data_path, "selected_cells.json"), "w") as f:
		json.dump(cells,f)

