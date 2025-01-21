import os
import json
import muon as mu
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from palantirModel.palantir_wrapper import PalantirWrapper
from palantirModel.palantir_utils import PalantirComparator
np.random.seed(52)


if __name__=="__main__":
	data_path = ... 
	data = mu.read_h5mu(os.path.join(data_path, "data.h5mu"))
	saving_path = os.path.join(os.getcwd(), "palantir_results")
	if not os.path.exists(saving_path):
		os.mkdir(saving_path)
	pw = PalantirWrapper()	

	with open(os.path.join(data_path, "cells.json"), "r") as f:
		cells = json.load(f)
		early_cell = cells["starting_cell"]
		terminal_states = cells["terminal_cells"]
		f.close()	

	#Activity run 		
	pw.compute_kernel(data["activity"], knn_key="neighbors", distance_key="distances")
	pw.run_diffusion_maps(data["activity"])
	pw.determine_multiscale_space(data["activity"])
	pw.run_palantir(data["activity"], early_cell=early_cell, num_waypoints=500)
	pw.plot_palantir_results(data["activity"])
	plt.show()

	#RNA run 
	pw.compute_kernel(data["rna"], knn_key ="neighbors", distance_key="distances") 	
	pw.run_diffusion_maps(data["rna"])
	pw.determine_multiscale_space(data["rna"])
	pw.run_palantir(data["rna"], early_cell=early_cell, num_waypoints=500)
	pw.plot_palantir_results(data, modality_key="rna")
	plt.show()
	

	# Fix terminal states
	pw.run_palantir(data["activity"], early_cell=early_cell, terminal_states = terminal_states, num_waypoints=500)
	pw.run_palantir(data["rna"], early_cell= early_cell, terminal_states=terminal_states, num_waypoints=500)
	pw.plot_palantir_results(data, modality_key="rna")
	plt.show()
	pw.plot_palantir_results(data, modality_key="activity")
	plt.show()

	pc = PalantirComparator()
	saving_folder = os.path.join(os.getcwd(), "palantir_results")
	pc.linear_model(data["activity"], data["rna"], "celltype", save=True, saving_path = os.path.join(saving_folder, "fates"), model1="activity")
	pc.linear_model(data["activity"], data["rna"], "celltype", is_fate=False, save=True, saving_path = os.path.join(saving_folder, "entropy"), key1="palantir_entropy", key2="palantir_entropy", 
model1="activity")
	pc.plot_probability_distribution(data["activity"], data["rna"], "celltype", saving_path = os.path.join(saving_folder, "fates"), model1="activity")	
	pc.plot_entropy_distribution(data["activity"], data["rna"], "celltype", saving_path = os.path.join(saving_folder, "entropy"), model1="activity")
