import os
import json
import muon as mu
import numpy as np
import pandas as pd

if __name__=="__main__":
	seed = 42
	np.random.seed(seed)
	working_dir = "/home/scvemo" # set repository 
	data_path = os.path.join(working_dir, "data", "e18_mouse", "data.h5mu")
	saving_path = os.path.join(working_dir, "data", "e18_mouse", "selected_cells_palantir.json")
	data = mu.read_h5mu(data_path)

	cells = {}
	
	# select initial cell 
	cells["initial"]= {"rg" : np.random.choice(data.obs[data.obs["celltype"]=="RG, Astro, OPC"].index)}
	with open(saving_path, "w") as f:
		json.dump(cells,f)

