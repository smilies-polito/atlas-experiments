import os
import argparse
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt
from atlas import ATLAS
from muon import MuData
from anndata import AnnData
from itertools import product
from scipy.sparse import csr_matrix
from .utils import initial_macrostate, terminal_macrostate, truth_like_fates

if __name__=="__main__":
	seed = 42
	np.random.seed(seed)
	
	# parse arguments from 
	parser = argparse.ArgumentParser() 
	parser.add_argument("--tree", type=str)
	parser.add_argument("--rd", type=float)
	parser.add_argument("--sigma", type=float)
	parser.add_argument("--knn_rna", type=int)
	parser.add_argument("--knn_activity", type=int)
	parser.add_argument("--wnn", type=int)
	args = parser.parse_args()
	
	working_directory = os.getcwd() # set path to repository 
	data_path = os.path.join(working_directory, "data", "simulated_data", args.tree)
	n_pcs_rna = 20
	n_pcs_activity = 10
	
	# Lettura dei dati 
	diff_cif_fraction, cif_sigma = args.rd, args.sigma
	activity = pd.read_csv(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_activity.tsv"), sep="\t", header=0, index_col=0)
	spliced = pd.read_csv(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_spliced.tsv"), sep="\t", header=0, index_col=0)
	unspliced = pd.read_csv(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_unspliced.tsv"), sep="\t", header=0, index_col=0)
	metadata = pd.read_csv(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_metadata.tsv"), sep="\t", header=0, index_col=0)

	# creazione matrice di attività
	activity = AnnData(X=csr_matrix(activity.values), 
			obs = pd.DataFrame(data=None, index=activity.index.values, columns=None), 
			var = pd.DataFrame(data=None, index=activity.columns, columns=None))	
	# creazione matrice di rna	
	total_rna = csr_matrix(spliced.values.T)
	rna = AnnData(X=csr_matrix(total_rna), 
			obs=pd.DataFrame(data=None, index=spliced.columns, columns=None), 
			var=pd.DataFrame(data=None, index=spliced.index.values, columns=None))
	rna.obs = rna.obs.merge(metadata, how="left", left_index=True, right_index=True)
		
	# creazione oggetto MuData 
	data = MuData({"rna":rna, "activity":activity})
	
	# rna preprocessing
	sc.pp.normalize_total(data["rna"])
	sc.pp.log1p(data["rna"])
	sc.pp.pca(data["rna"], random_state=seed, use_highly_variable=False)

	# activity preprocessing
	sc.pp.normalize_total(data["activity"])
	sc.pp.pca(data["activity"], random_state=seed, use_highly_variable=False)

	knn_rna, knn_activity, wnn = args.knn_rna, args.knn_activity, args.wnn
	saving_folder = os.path.join(data_path, f"{knn_rna}_{knn_activity}_{wnn}") 
	if not os.path.exists(saving_folder):
		os.mkdir(saving_folder)
			
	atlas = ATLAS(mudata = data,
					method= "palantir",
					fragment_path = None,
					random_state =  seed)
	assert atlas._impl.rna_key == "rna"
	assert atlas._impl.activity_key == "activity"
	assert atlas._impl.atac_key is None
	assert atlas._impl.use_activity 
	assert atlas._impl.random_state == seed
	assert isinstance(atlas._impl.mudata, MuData)
	assert not hasattr(atlas._impl.mudata["activity"].uns, "files")


	atlas.preprocessing(n_pcs_rna = n_pcs_rna,
						n_pcs_act = n_pcs_activity,
						knn_rna = knn_rna,
						knn_act = knn_activity,
						n_neighbors = wnn) 

	assert "distances" in atlas._impl.mudata["rna"].obsp
	assert "connectivities" in atlas._impl.mudata["rna"].obsp
	assert "distances" in atlas._impl.mudata["activity"].obsp
	assert "connectivities" in atlas._impl.mudata["activity"].obsp
	assert "wnn_distances" in atlas._impl.mudata.obsp	
	assert "wnn_connectivities" in atlas._impl.mudata.obsp	
	assert atlas._impl.mudata.obsp["wnn_distances"].shape == (data.n_obs, data.n_obs)
	assert "X_umap" in atlas._impl.mudata.obsm

	mu.pl.embedding(data, 
					basis="X_umap", 
					color=["rna:pop", "rna:pseudotime"], 
					show=False, 
					save = f"{diff_cif_fraction}_{cif_sigma}_{knn_rna}:{knn_activity}:{wnn}.png" )

	early_cell = initial_macrostate(mudata = data, 
									pseudotime_key = "rna:pseudotime", 
									n_cells=1)
	terminal_cells = []
	for branch in ["4_1", "5_2", "5_3"]:
		cell = terminal_macrostate(mudata = data,
											pseudotime_key = "rna:pseudotime", 
											cluster_key = "rna:pop",
											terminal_state = branch,
											n_cells=1)			
		terminal_cells.extend(cell)

	# test non fixed terminal states	
	atlas.run(early_cell = early_cell[0],
				cluster_key = "rna:pop",
				terminal_states = None,
				knn = 30,
				num_waypoints = 250)
	assert "pseudotime" in data.obs.columns	
	assert "shannon_entropy" in data.obs.columns	
	assert "kl_divergence" in data.obs.columns	
	assert "palantir_waypoints" in data.uns
	assert "fate_probabilities" in data.obsm 
	assert isinstance(data.obsm["fate_probabilities"], pd.DataFrame)
	assert "initial_states" in data.uns
	assert isinstance(data.uns["initial_states"], dict)
	assert "terminal_states" in data.uns
	assert isinstance(data.uns["terminal_states"], dict)
	assert hasattr(atlas._impl, "fate_probability_key")
	assert atlas._impl.fate_probability_key == "fate_probabilities"
	assert "fate_state_colors" in data.uns
	n_states = len(data.uns["initial_states"]) + len(data.uns["terminal_states"])
	assert len(data.uns["fate_state_colors"]) == n_states

	# test fixed terminal states 
	atlas.run(early_cell = early_cell[0], 
				cluster_key = "rna:pop",
				terminal_states = terminal_cells,
				knn = 30,
				num_waypoints = 250)
			
	assert "pseudotime" in data.obs.columns	
	assert "shannon_entropy" in data.obs.columns	
	assert "kl_divergence" in data.obs.columns	
	assert "palantir_waypoints" in data.uns
	assert "fate_probabilities" in data.obsm 
	assert isinstance(data.obsm["fate_probabilities"], pd.DataFrame)
	assert "initial_states" in data.uns
	assert isinstance(data.uns["initial_states"], dict)
	assert "terminal_states" in data.uns
	assert isinstance(data.uns["terminal_states"], dict)
	assert hasattr(atlas._impl, "fate_probability_key")
	assert atlas._impl.fate_probability_key == "fate_probabilities"
	assert "fate_state_colors" in data.uns
	n_states = len(data.uns["initial_states"]) + len(data.uns["terminal_states"])
	assert len(data.uns["fate_state_colors"]) == n_states
		




