import os
import muon as mu
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from palantir_wrapper import PalantirWrapper
from palantir_utils import PalantirComparator
np.random.seed(52)


if __name__=="__main__":
	data_path = "/Users/lrcq/Documents/devtraj/preprocessing/embryonicMouseBrain10x"
	data = mu.read_h5mu(os.path.join(data_path, "data.h5mu"))
	pw = PalantirWrapper()	


	starting_cell = np.random.choice(data.obs[data.obs["rna:celltype"]=="RG, Astro, OPC"].index.values, 1)[0]
	data.obs["is_starting"] = data.obs.index == starting_cell
	# sc.pl.embedding(data, basis="X_umap", color = ["is_starting"], palette="PuRd")
	
	#Multiomics run 		
	pw.compute_kernel(data)
	pw.run_diffusion_maps(data)
	pw.determine_multiscale_space(data)
#	pw.run_palantir(data, early_cell=starting_cell, num_waypoints=500)
#	pw.plot_palantir_results(data)
#	plt.show()

	#RNA run 
	pw.compute_kernel(data["rna"], knn_key ="neighbors" , distance_key="distances") 	
	pw.run_diffusion_maps(data["rna"])
	pw.determine_multiscale_space(data["rna"])
#	pw.run_palantir(data["rna"], early_cell=starting_cell, num_waypoints=500)
#	pw.plot_palantir_results(data, modality_key="rna")
#	plt.show()
	
	# TERMINAL STATES: we set terminal states to compare group_wise transition probabilities and entropies. 
	# we set one terminal state in the Upper Layer and one in the Deeper Layer by randomly selecting cells using the rna:celltype 
	# we set the astrocytes and opcs by selecting cells with high expression of marker genes
	ul_cell= np.random.choice(data.obs[data.obs["rna:celltype"]=="Upper Layer"].index.values,1)
	dl_cell = np.random.choice(data.obs[data.obs["rna:celltype"]=="Deeper Layer"].index.values, 1)
	
	astro_genes = [gene for gene in ["Vim", "Gfap", "Aldhl1"] if gene in data.var_names]
	opcs_genes = [ gene for gene in ["Pdgfra", "Olig2", "Sox10"] if gene in data.var_names]
	
	threshold = 0.8
	astro_mask = data["rna"][:, astro_genes].X.A > threshold
	opcs_mask = data["rna"][:, opcs_genes].X.A> threshold
	as_cell= np.random.choice(data[astro_mask.all(axis=1)].obs_names, 1)
	op_cell= np.random.choice(data[opcs_mask.all(axis=1)].obs_names, 1)
	terminal_states = np.concatenate((ul_cell, dl_cell, as_cell, op_cell), axis=0)
	data.obs["is_terminal"] = data.obs.index.isin(terminal_states)
	# sc.pl.embedding(data, basis="X_umap", color = ["is_terminal"], palette = "PuRd")

	pw.run_palantir(data, early_cell=starting_cell, terminal_states = terminal_states, num_waypoints=500)
	pw.run_palantir(data["rna"], early_cell=starting_cell, terminal_states=terminal_states, num_waypoints=500)
#	pw.plot_palantir_results(data, modality_key="rna")
#	plt.show()
#	pw.plot_palantir_results(data)
#	plt.show()

	pc = PalantirComparator()
	saving_folder = os.path.join(os.getcwd(), "palantir_results")
	if not os.path.exists(saving_folder):
		os.mkdir(saving_folder)
	pc.linear_model(multiomics=data, rna=data["rna"], group_key="rna:celltype", saving_path = saving_folder)
	pc.plot_probability_distribution(multiomics=data, rna = data["rna"], group_key="rna:celltype", saving_path = saving_folder)	
		
