import os
import numpy as np
import muon as mu 

if __name__=="__main__":
	data_path = ... 
	saving_path = ...
	data = mu.read_h5mu(data_path)

	# Verso stati terminali per cellule appartenenti al cluster stesso 
	data.obs["4-1"] = (data.obs["rna:pop"]=="4_1").astype(int) 
	data.obs["4-5-2"] = (data.obs["rna:pop"]=="5_2").astype(int)
	data.obs["4-5-3"] = (data.obs["rna:pop"]== "5_3").astype(int) 

	# cellule degli stati intermedi: 4_5 
	data.obs["4-5-2"] = data.obs["4-5-2"] + (data.obs["rna:pop"]=="4_5").astype(int) * 0.5 
	data.obs["4-5-3"] = data.obs["4-5-3"] + (data.obs["rna:pop"] == "4_5").astype(int)* 0.5 
	
	mu.pl.embedding(data, basis="X_umap", color=["4-5-2", "4-5-3", "4-1"])
	data.obs[["4-5-2", "4-5-3", "4-1"]].to_csv(saving_path, sep="\t", header=True, index=True)

