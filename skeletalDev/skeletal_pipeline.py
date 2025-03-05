import os
import muon as mu
import numpy as np
from palantirModel.palantir_wrapper import PalantirWrapper
from palantirModel.palantir_plots import plot_palantir_results
from palantirModel.palantir_utils import PalantirComparator

if __name__=="__main__":
	seed = 42
	np.random.seed(seed)
	bone = "shoulder"

	data_path = os.path.join(os.getcwd(), "skeletalDev")
	saving_folder_multiomics = os.path.join(os.getcwd(), f"multiomics_{bone}")
	saving_folder_rna = os.path.join(os.getcwd(), F"rna_{bone}")
	if not os.path.exists(saving_folder_multiomics):
		os.mkdir(saving_folder_multiomics)
	if not os.path.exists(saving_folder_rna):
		os.mkdir(saving_folder_rna)

	data = mu.read_h5mu(os.path.join(data_path, f"{bone}_data.h5mu"))
	
	# identificare early cell per pseudotime e terminal states
	# Palantir identifica una singola early cell.	
	early_cell = np.random.choice(data.obs_names[data.obs["rna:pcw"]=="5.7"])
	terminal_states = []
	
	fix_terminal = False
	pw = PalantirWrapper()	
	
	# Multiomics run
	pw.compute_kernel(data)
	pw.run_diffusion_maps(data)
	pw.determine_multiscale_space(data)
	if fix_terminal:
		pw.run_palantir(data, early_cell = early_cell, num_waypoints=2000, terminal_states = terminal_states)
	else:
		pw.run_palantir(data, early_cell = early_cell, num_waypoints=2000) 

	plot_palantir_results(data, saving_path = saving_folder_multiomics, entropy_key=["palantir_entropy"])
	
	# RNA run
	pw.compute_kernel(data["rna"], knn_key="neighbors", distance_key="distances")
	pw.run_diffusion_maps(data["rna"])
	pw.determine_multiscale_space(data["rna"])
	if fix_terminal:
		pw.run_palantir(data["rna"], early_cell=early_cell, num_waypoints=2000, terminal_states=terminal_states)
	else:
		pw.run_palantir(data["rna"], early_cell = early_cell, num_waypoints = 2000)

	plot_palantir_results(data, modality_key = "rna", saving_path = saving_folder_rna, entropy_key=["entropy"])

	if fix_terminal:
		pc=PalantirComparator()
		pc.save_palantir_matrix(data, data["rna"], "rna:Celltype_fig1", saving_path = os.path.join(saving_folder_multiomics, "fates.tsv"))
		pc.save_palantir_matrix(data, data["rna"], "rna:Celltype_fig1", saving_path = os.path.join(saving_folder_multiomics, "entropy.tsv"), is_fate=False, key1="palantir_entropy", key2="palantir_entropy")	
