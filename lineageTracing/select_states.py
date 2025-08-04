import os
import json 
import muon as mu
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from core import search_cells

def get_state(x:str) -> str:
	mapping = {"initial": ["HSC", "Refined.HSC"], 
		"erythroid": ["EryP"],
		"megakaryocyte": ["MKP"], 
		"monocyte": ["Mono"],
		"NK": ["NK"],
		"B": ["B", "Plasma"],
		"dendritic" : ["cDC", "pDC"],
		"T": ["CD4", "CD8"], 
		"intermedate": ["MDP", "GMP", "CMP", "MEP", "MPP", "LMPP", "CLP", "ProB"]}
	for k,v in mapping.items():
		if x in v:
			return k
	return None
 


if __name__=="__main__":
	seed = 42
	np.random.seed(seed)

	working_dir = ...
	donor = ...
	data_path = os.path.join(working_dir, "data", "lineage_tracing", f"{donor}")
	
	data = mu.read_h5mu(os.path.join(data_path, "data.h5mu"))
	data.obs["states"] = data.obs["STD.CellType"].map(lambda x: get_state(x))

	# SELECT INITIAL NEIGHBORHOOD	
	nearest_cells = search_cells(data, embedding_key = "X_umap", grouping_key = "states", n_select=30)
	
	# PLOT
	#data.obs["selected"] = (data.obs_names.isin(nearest_cells["erythroid"])) | (data.obs_names.isin(nearest_cells["megakaryocyte"])) | (data.obs_names.isin(nearest_cells["monocyte"])) | (data.obs_names.isin(nearest_cells["T"])) | (data.obs_names.isin(nearest_cells["B"])) | (data.obs_names.isin(nearest_cells["dendritic"])) | (data.obs_names.isin(nearest_cells["NK"])) | (data.obs_names.isin(nearest_cells["initial"]))
	#mu.pl.embedding(data, basis="X_umap", color=["selected"], save=f"{donor}_selected_palantir.png", show=False)

	cells = {"initial": {"hsc": nearest_cells["initial"]}, "terminal": {"erythroid": nearest_cells["erythroid"], "megakaryocyte": nearest_cells["megakaryocyte"], "monocyte": nearest_cells["monocyte"], "T": nearest_cells["T"], "B": nearest_cells["B"], "NK": nearest_cells["NK"], "dendritic": nearest_cells["dendritic"]}}
	with open(os.path.join(data_path, "selected_cells.json"), "w") as f:
		json.dump(cells,f)

