import os
import json
import muon as mu
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from palantirModel.palantir_wrapper import PalantirWrapper
from palantirModel.palantir_utils import PalantirComparator
from palantirModel.palantir_plots import plot_palantir_results
np.random.seed(52)


if __name__=="__main__":
	data_path = "/Users/lrcq/Documents/devtraj/preprocessing/embryonicMouseBrain10X"
	data = mu.read_h5mu(os.path.join(data_path, "data.h5mu"))
	saving_path = os.path.join(os.getcwd(), "palantir_results_multiomics")
	if not os.path.exists(saving_path):
		os.mkdir(saving_path)
	pw = PalantirWrapper()	

	with open(os.path.join(data_path, "cells.json"), "r") as f:
		cells = json.load(f)
		early_cell = cells["starting_cell"]
		terminal_states = cells["terminal_cells"]
		f.close()	

	#Multiomics run 		
	pw.compute_kernel(data)
	pw.run_diffusion_maps(data)
	pw.determine_multiscale_space(data)
	pw.run_palantir(data, early_cell=early_cell, num_waypoints=500)
	data.obs["kl_divergence"] = pw.compute_priming_degree(data, entropy_type="kl_divergence")
	plot_palantir_results(data, saving_path=saving_path, entropy_key = ["palantir_entropy", "kl_divergence"])

	#RNA run 
	pw.compute_kernel(data["rna"], knn_key ="neighbors" , distance_key="distances") 	
	pw.run_diffusion_maps(data["rna"])
	pw.determine_multiscale_space(data["rna"])
	pw.run_palantir(data["rna"], early_cell=early_cell, num_waypoints=500)
	data["rna"].obs["kl_divergence"] = pw.compute_priming_degree(data["rna"], entropy_type="kl_divergence")
	plot_palantir_results(data, modality_key="rna", saving_path=saving_path, entropy_key = ["palantir_entropy", "kl_divergence"])
	

	# Fix terminal states
	pw.run_palantir(data, early_cell=early_cell, terminal_states = terminal_states, num_waypoints=500)
	pw.run_palantir(data["rna"], early_cell= early_cell, terminal_states=terminal_states, num_waypoints=500)
	data.obs["kl_divergence"] = pw.compute_priming_degree(data, entropy_type="kl_divergence")
	data["rna"].obs["kl_divergence"]=pw.compute_priming_degree(data["rna"], entropy_type= "kl_divergence")	
	plot_palantir_results(data, modality_key="rna", saving_path=saving_path, entropy_key=["palantir_entropy", "kl_divergence"])
	plot_palantir_results(data, saving_path = saving_path, entropy_key=["palantir_entropy", "kl_divergence"])
		

	pc = PalantirComparator()
	pc.save_palantir_matrix(data, data["rna"], "rna:celltype", saving_path = os.path.join(saving_path, "fates.tsv"))
	pc.save_palantir_matrix(data, data["rna"], "rna:celltype", is_fate=False, key1="palantir_entropy", key2="palantir_entropy", saving_path= os.path.join(saving_path, "entropy.tsv"))
