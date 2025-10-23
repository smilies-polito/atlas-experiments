import os
import pandas as pd
import json
import muon as mu 
from itertools import product
from src.plots import plot_trend


if __name__=="__main__":
	working_dir = "/scvemo"
	data_dir = os.path.join(working_dir, "data", "female_gonads")
	results_dir = os.path.join(working_dir, "output")
	tfs_path = os.path.join(working_dir, "data", "TF_targets_dictionary.json")
	with open(tfs_path, "r") as f:
		tfs_dict = json.load(f)
		f.close()
	
	datasetType =  ["scvi", "bbknn_filtered", "harmony_filtered", "harmony"]
	model = ["multiomics", "rna"]

	pseudotime_key = "palantir_pseudotime"
	fate_probs_key = "fate_probabilities"
	threshold = 0.4
	grid = product(datasetType, model)

	for data_type, mod in grid:
		data = mu.read_h5mu(os.path.join(data_dir, f"palantir_{data_type}.h5mu"))
		modality = "rna" if mod=="rna" else None
		print(data_type, mod, modality)
		saving_path_root = os.path.join(results_dir, f"female_gonads_{data_type}", f"pseudotime_{mod}")
		print(saving_path_root)

		for tf_name, genes in tfs_dict.items():
			tf_name = tf_name.upper()
			genes = [g.upper() for g in genes]
			print(tf_name)

			if tf_name not in data.var_names:
				print(f"TF not available")
				continue
			genes_of_interest = [g for g in genes if g in data.var_names]
			if len(genes_of_interest) <= 0:
				print(f"No genes associated")
				continue
			for nstates in range(5,20,2):
				saving_path = os.path.join(saving_path_root, f"{nstates}")
				print(saving_path)
				path = os.path.join(data_dir, f"female_{data_type}_{mod}_{nstates}.csv")
				print(path)
				if not os.path.exists(path):
					print(f"{path} not available")
					continue
				fates = pd.read_csv(path, sep=",", header=0, index_col=0)
				print(fates.head(2))
				if model == "rna":
					data["rna"].obsm[fate_probs_key]=fates
				else:
					data.obsm[fate_probs_key]=fates
			# cellrank fates
			for branch in fates.columns:
				plot_trend(data, genes_of_interest = genes_of_interest, tf_name = tf_name, pseudotime_key = pseudotime_key, fate_probs_key=fate_probs_key, branch=branch, threshold=threshold, saving_path = saving_path, modality = modality)
			
		
	
