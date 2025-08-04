import os
import json
import muon as mu
import numpy as np
import pandas as pd
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
	working_dir = ... # set repository 
	donor= ...  #either donor1 or donor2
	data_path = os.path.join(working_dir, "data", "lineage_tracing", f"{donor}", "data.h5mu")
	saving_path = os.path.join(working_dir, "data", "lineage_tracing", f"{donor}", "selected_cells_palantir.json")
	data = mu.read_h5mu(data_path)
	data.obs["states"] = data.obs["STD.CellType"].map(lambda x: get_state(x))

	nearest_cells = search_cells(data, embedding_key = "X_umap", grouping_key = "states", n_select=1)

	cells = {"initial": {"hsc": nearest_cells["initial"][0]}, "terminal": {"erythroid": nearest_cells["erythroid"][0], "megakaryocyte": nearest_cells["megakaryocyte"][0], "monocyte": nearest_cells["monocyte"][0], "T": nearest_cells["T"][0], "B": nearest_cells["B"][0], "NK": nearest_cells["NK"][0], "dendritic": nearest_cells["dendritic"][0]}}

	with open(os.path.join(saving_path), "w") as f:
		json.dump(cells,f)


