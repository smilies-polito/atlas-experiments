import os
import pandas as pd
import json
import muon as mu 
from itertools import product
from src.plots import plot_trend


if __name__=="__main__":
	working_dir = "/scvemo"
	data_dir = os.path.join(working_dir, "data", "female_gonads")
	results_dir = os.path.join(working_dir, "output", "female_gonads")
	tfs_path = os.path.join(working_dir, "data", "TF_targets_dictionary.json")
	with open(tfs_path, "r") as f:
		tfs_dict = json.load(f)
		f.close()
	
	datasetType =  ["bbknn", "harmony"]
	model = ["multiomics", "rna"]

	pseudotime_key = "palantir_pseudotime"
	fate_probs_key = "fates_probabilities"
	threshold = 0.4
	n_macrostates = list(range(2,10,2))
	grid = product(datasetType, model, n_macrostates)

	for data_type, mod, n_states in grid:
		data = mu.read_h5mu(os.path.join(data_dir, f"palantir_supporting_{data_type}.h5mu"))
		csv_path = os.path.join(data_dir, f"supporting_{data_type}_{mod}_{n_states}.csv")
		if not os.path.exists(csv_path):
			continue
		fates = pd.read_csv(csv_path, header=0, index_col=0)
		modality = "rna" if mod=="rna" else None
		saving_path = os.path.join(results_dir, f"supporting_{data_type}", f"palantir_{mod}")

		for tf_name, genes in tfs_dict.items():
			tf_name = tf_name.upper()
			genes = [g.upper() for g in genes]

			if tf_name not in data.var_names:
				print(f"TF not available")
				continue
			genes_of_interest = [g for g in genes if g in data.var_names]
			if len(genes_of_interest) <= 0:
				print(f"No genes associated")
				continueù
			if mod=="multiomics":
				data.obsm[fate_probs_key] = fates 
			else:
				data[mod].obsm[fate_probs_key] = fates 
			# cellrank fates
			for branch in fates.columns:
				plot_trend(data, genes_of_interest = genes_of_interest, tf_name = tf_name, pseudotime_key = pseudotime_key, fate_probs_key=fate_probs_key, branch=branch, threshold=threshold, saving_path = saving_path, modality = modality)
			plot_trend(data, genes_of_interest = genes_of_interest, tf_name = tf_name, pseudotime_key = pseudotime_key, fate_probs_key=fate_probs_key, branch=None, threshold=threshold, saving_path = saving_path, modality = modality)
			
		
	
