import os
import json
import muon as mu
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from palantirModel.palantir_wrapper import PalantirWrapper
from palantirModel.palantir_utils import PalantirComparator
from palantirModel.palantir_plots import plot_palantir_results
np.random.seed(42)


if __name__=="__main__":
	data_path = # INSERT PATH 
	data = mu.read_h5mu(os.path.join(data_path, "data.h5mu"))
	saving_path_multiomics = os.path.join(os.getcwd(), "palantir_results_multiomics_terminal")
	saving_path_rna = os.path.join(os.getcwd(), "palantir_results_rna_terminal")
	if not os.path.exists(saving_path_multiomics):
		os.mkdir(saving_path_multiomics)
	if not os.path.exists(saving_path_rna):
		os.mkdir(saving_path_rna)
	pw = PalantirWrapper()	

	with open(os.path.join(data_path, "cells.json"), "r") as f:
		cells = json.load(f)
		early_cell = cells["starting_cell"]
		terminal_states = cells["terminal_cells"]
		f.close()	
	
	#Multiomics run 	
	fix_terminal= True
	
	pw.compute_kernel(data)
	pw.run_diffusion_maps(data)
	pw.determine_multiscale_space(data)
	if fix_terminal:
		pw.run_palantir(data, early_cell=early_cell, num_waypoints=500, terminal_states=terminal_states)
	else:
		pw.run_palantir(data, early_cell=early_cell, num_waypoints=500)
		
	data.obs["kl_divergence"] = pw.compute_priming_degree(data, entropy_type="kl_divergence")
	plot_palantir_results(data, saving_path=saving_path_multiomics, entropy_key = ["palantir_entropy", "kl_divergence"])

	#RNA run 
	pw.compute_kernel(data["rna"], knn_key ="neighbors" , distance_key="distances") 	
	pw.run_diffusion_maps(data["rna"])
	pw.determine_multiscale_space(data["rna"])
	
	if fix_terminal:
		pw.run_palantir(data["rna"], early_cell=early_cell, num_waypoints=500, terminal_states = terminal_states)
	else:
		pw.run_palantir(data["rna"], early_cell=early_cell, num_waypoints=500)

	data["rna"].obs["kl_divergence"] = pw.compute_priming_degree(data["rna"], entropy_type="kl_divergence")
	plot_palantir_results(data, modality_key="rna", saving_path=saving_path_rna, entropy_key = ["palantir_entropy", "kl_divergence"])

	if fix_terminal:	
		pc = PalantirComparator()
		pc.save_palantir_matrix(data, data["rna"], "rna:celltype", saving_path = os.path.join(saving_path_multiomics, "fates.tsv"))
		pc.save_palantir_matrix(data, data["rna"], "rna:celltype", is_fate=False, key1="palantir_entropy", key2="palantir_entropy", saving_path= os.path.join(saving_path_multiomics, "entropy.tsv"))
		pc.save_palantir_matrix(data, data["rna"], "rna:celltype", is_fate=False, key1="kl_divergence", key2="kl_divergence", saving_path = os.path.join(saving_path_multiomics, "kl_divergence.tsv"))
