import os
import argparse 
import pandas as pd
import json
import muon as mu 
from itertools import product
from src.plots import plot_trend, plot_heatmap


if __name__=="__main__":
	working_dir = "/scvemo"

	parser = argparse.ArgumentParser()
	parser.add_argument("--site", type=str)
	parser.add_argument("--lineage", type=str)
	args = parser.parse_args()	
	site = args.site
	lineage = args.lineage
	
	data_dir = os.path.join(working_dir, "data", "skeletalDev")
	results_dir = os.path.join(working_dir, "output", "skeletalDev")
	tfs_path = os.path.join(working_dir, "data", "TF_targets_dictionary.json")
	with open(tfs_path, "r") as f:
		tfs_dict = json.load(f)
		f.close()
	
	model_list = ["multiomics", "rna"]
	macrostates_to_eval = list(range(2,10,2))
	grid = product(model_list, macrostates_to_eval)

	pseudotime_key = "palantir_pseudotime"
	fate_probs_key = "fates_probabilities"
	threshold = 0.4

	for model, n_macrostates in grid:
		data_suffix = f"palantir_harmony_{site}.h5mu" if lineage == "whole" else f"palantir_harmony_{site}_{lineage}.h5mu"
		data = mu.read_h5mu(os.path.join(data_dir, data_suffix))
		modality = "rna" if model=="rna" else None

		saving_path = os.path.join(results_dir, f"harmony_{site}_{lineage}", f"pseudotime_{model}", f"{n_macrostates}")

		for tf_name, genes in tfs_dict.items():
			tf_name = tf_name.upper()
			genes = [g.upper() for g in genes]

			if tf_name not in data.var_names:
				print(f"TF not available")
				continue
			genes_of_interest = [g for g in genes if g in data.var_names]
			if len(genes_of_interest) <= 0:
				print(f"No genes associated")
				continue

			file_name = f"harmony_{site}_{model}_{n_macrostates}.csv" if lineage == "whole" else f"harmony_{site}_{lineage}_{model}_{n_macrostates}.csv" 
			if not os.path.exists(os.path.join(data_dir, file_name)):
				print(f"Execution for {lineage} {site} {model} {n_macrostates} did not end")
				continue
			fates = pd.read_csv(os.path.join(data_dir, file_name), sep=",", header=0, index_col=0)
			if model=="multiomics":	
				data.obsm[fate_probs_key] = fates
			else:
				data[model].obsm[fate_probs_key] = fates

			# cellrank fates
			for branch in fates.columns:
				plot_trend(data, genes_of_interest = genes_of_interest, tf_name = tf_name, pseudotime_key = pseudotime_key, fate_probs_key=fate_probs_key, branch=branch, threshold=threshold, saving_path = saving_path, modality = modality)
			plot_trend(data, genes_of_interest = genes_of_interest, tf_name = tf_name, pseudotime_key = pseudotime_key, fate_probs_key=fate_probs_key, branch=None, threshold=threshold, saving_path = saving_path, modality = modality)
			
		
	
