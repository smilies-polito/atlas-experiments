import os
import numpy as np
import muon as mu 

if __name__=="__main__":
	working_directory = ... # set to repository 
	data_path = os.path.join(working_directory, "data", "phyla5")
	saving_path = os.path.join(data_path, "branch_assignment.tsv")
	data = mu.read_h5mu(os.path.join(data_path, "30_30_30", "0.5_0.5.h5mu"))

	# Verso stati terminali per cellule appartenenti al cluster stesso 
	data.obs["6-1"] = (data.obs["rna:pop"]=="6_1").astype(int) 
	data.obs["6-7-8-2"] = (data.obs["rna:pop"]=="8_2").astype(int)
	data.obs["6-7-8-3"] = (data.obs["rna:pop"]== "8_3").astype(int) 
	data.obs["6-7-9"] = ((data.obs["rna:pop"] == "9_4") | (data.obs["rna:pop"]=="9_5")).astype(int) 

	# cellule degli stati intermedi: 6_7, 7_8, 8_9
	# cellule di 7_8 possono andare o veso 8_2 o verso 8_3 con probabilità 0.5 
	# cellule di 7_9 possono andare solo verso 6-7-9
	data.obs["6-7-9"] = data.obs["6-7-9"] + (data.obs["rna:pop"]=="7_9").astype(int)
	data.obs["6-7-8-2"] = data.obs["6-7-8-2"] + (data.obs["rna:pop"]=="7_8").astype(int) * 0.5 
	data.obs["6-7-8-3"] = data.obs["6-7-8-3"] + (data.obs["rna:pop"] == "7_8").astype(int)* 0.5 
	
	# cellule di 6_7 possono diventare qualsiasi quindi 
	data.obs["6-7-9"] = data.obs["6-7-9"] + (data.obs["rna:pop"] == "6_7").astype(int) * 1/3
	data.obs["6-7-8-2"] = data.obs["6-7-8-2"] + (data.obs["rna:pop"] == "6_7").astype(int) * 1/3
	data.obs["6-7-8-3"] = data.obs["6-7-8-3"] + (data.obs["rna:pop"] == "6_7").astype(int) * 1/3

	mu.pl.embedding(data, basis="X_umap", color=["6-7-9", "6-7-8-2", "6-7-8-3", "6-1"])
	data.obs[["6-7-9", "6-7-8-2", "6-7-8-3", "6-1"]].to_csv(saving_path, sep="\t", header=True, index=True)

