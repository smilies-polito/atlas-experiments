import os
import json
import muon as mu 
import numpy as np

if __name__=="__main__":
	seed = 42
	data_path = ...
	saving_path = ...
	data = mu.read_h5mu(data_path)

	print("Celltype distirbution for cells with pseudotime > 0.9")
	print(data[data.obs["rna:pseudotime"]>0.9].obs.groupby(by="rna:pop").count())
	print("Celltype distribution for cells with pseudotime < 0.1")
	print(data[data.obs["rna:pseudotime"]<0.1].obs.groupby(by="rna:pop").count())

	np.random.seed(seed)
	initial_states = np.random.choice(data[(data.obs["rna:pseudotime"]<0.1) & (data.obs["rna:pop"].isin(["4_1", "4_5"]))].obs_names, 30).tolist()
	np.random.seed(seed)
	terminal41 = np.random.choice(data[(data.obs["rna:pseudotime"]>0.9) & (data.obs["rna:pop"]=="4_1")].obs_names, 30).tolist()
	np.random.seed(seed)
	terminal53 = np.random.choice(data[(data.obs["rna:pseudotime"]>0.9) & (data.obs["rna:pop"]=="5_3")].obs_names, 30).tolist()
	np.random.seed(seed)
	terminal52 = np.random.choice(data[(data.obs["rna:pseudotime"]>0.9) & (data.obs["rna:pop"]=="5_2")].obs_names, 30).tolist()

	cells = {"initial": {"4_1_1" : initial_states}, "terminal":{"4_1_2": terminal41, "5_2":terminal52, "5_3":terminal53}}
	with open(saving_path, "w") as f:
		json.dump(cells, f)
			
	
