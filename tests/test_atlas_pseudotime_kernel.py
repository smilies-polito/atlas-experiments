import os
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
from atlas import ATLAS
from muon import MuData
from anndata import AnnData
from scipy.sparse import csr_matrix
from simulated_data import initial_macrostate, terminal_macrostate, truth_like_fates

if __name__=="__main__":
	seed = 42
	working_directory = os.getcwd() # set path to repository 
	np.random.seed(seed)
	
	#1. TEST CONSTRUCTION WITHOUT FRAGMENT FILES AND ACTIVITY ALREADY PROVIDED
	tree = "three_branches"
	diff_cif_fraction, cif_sigma = 0.3, 0.3
	knn_rna, knn_activity, wnn = 30, 30, 30
	n_pcs_rna, n_pcs_activity = 20, 10
	
	data_path = os.path.join(working_directory, "data", "simulated_data", tree)
	activity = pd.read_csv(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_activity.tsv"), sep="\t", header=0, index_col=0)
	spliced = pd.read_csv(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_spliced.tsv"), sep="\t", header=0, index_col=0)
	metadata = pd.read_csv(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_metadata.tsv"), sep="\t", header=0, index_col=0)

	# create activity matrix
	activity = AnnData(X=csr_matrix(activity.values), 
			obs = pd.DataFrame(data=None, index=activity.index.values, columns=None), 
			var = pd.DataFrame(data=None, index=activity.columns, columns=None))	
	sc.pp.normalize_total(activity)
	sc.pp.pca(activity, random_state=seed, use_highly_variable=False)
	# create rna matrix
	rna = AnnData(X=csr_matrix(spliced.values.T), 
			obs=pd.DataFrame(data=None, index=spliced.columns, columns=None), 
			var=pd.DataFrame(data=None, index=spliced.index.values, columns=None))
	rna.obs = rna.obs.merge(metadata, how="left", left_index=True, right_index=True)
	sc.pp.normalize_total(rna)
	sc.pp.log1p(rna)
	sc.pp.pca(rna, random_state=seed, use_highly_variable=False)

	data = MuData({"rna":rna, "activity":activity})

	# ground truth construction
	early_cell = initial_macrostate(mudata = data, 
									pseudotime_key = "rna:pseudotime", 
									n_cells=30)
	cluster_subset = data.obs["rna:pop"].loc[early_cell].value_counts().idxmax()
	early_cell = {cluster_subset : early_cell}

	n_cells, n_rna, n_activity = data.n_obs, data["rna"].n_vars, data["activity"].n_vars

	atlas = ATLAS(mudata = data,
					method= "pseudotime-kernel",
					fragment_path = None,
					random_state = seed,
					pseudotime_key = "rna:pseudotime",
					connectivity_key = "wnn_connectivities",
					cluster_key = "rna:pop")

	assert atlas._impl.rna_key == "rna"
	assert atlas._impl.activity_key == "activity"
	assert atlas._impl.atac_key is None
	assert atlas._impl.use_activity 
	assert atlas._impl.random_state == seed
	assert isinstance(atlas._impl.mudata, MuData)
	assert not hasattr(atlas._impl.mudata["activity"].uns, "files")
	assert atlas._impl.cluster_key == "rna:pop"
	assert atlas._impl.pseudotime_key == "rna:pseudotime"


	# 2. TEST PREPROCESSING 
	atlas.preprocessing(n_pcs_rna = n_pcs_rna,
						n_pcs_act = n_pcs_activity,
						knn_rna = knn_rna,
						knn_act = knn_activity,
						n_neighbors = wnn) 

	mudata = atlas.get_data()
	assert "distances" in mudata["rna"].obsp
	assert "connectivities" in mudata["rna"].obsp
	assert "distances" in mudata["activity"].obsp
	assert "connectivities" in mudata["activity"].obsp
	assert "wnn_distances" in mudata.obsp	
	assert "wnn_connectivities" in mudata.obsp	
	assert mudata.obsp["wnn_distances"].shape == (n_cells, n_cells)
	assert "X_umap" in mudata.obsm

	# 3. TEST RUN WITH NO FIXED TERMINAL 
	connectivity_key, threshold_scheme = "wnn_connectivities", "hard"
	atlas.run(connectivity_key = connectivity_key,
			threshold_scheme = threshold_scheme, 
			n_states = None, 
			allow_overlap = True)

	
	assert isinstance(atlas._impl._adata, AnnData)
	assert atlas._impl._adata.n_obs == n_cells
	assert atlas._impl._adata.obsp["wnn_connectivities"].shape == (n_cells, n_cells)
	assert atlas._impl.kernel.transition_matrix.shape == (n_cells, n_cells)
	assert isinstance(atlas._impl.mudata.obsm["fate_probabilities"], pd.DataFrame)
	assert atlas._impl.fate_probability_key == "fate_probabilities"
	assert "initial_states" in mudata.uns	
	assert isinstance(mudata.uns["initial_states"], dict)
	assert "terminal_states" in mudata.uns	
	assert isinstance(mudata.uns["terminal_states"], dict)
	assert "intermediate_states" in mudata.uns	
	assert isinstance(mudata.uns["intermediate_states"], dict)
	assert "fate_state_colors" in mudata.uns
	assert "kl_divergence" in mudata.obs.columns
	assert "shannon_entropy" in mudata.obs.columns


	# test fixed terminal states 
	terminal_cells = {}
	for branch in ["4_1", "5_2", "5_3"]:
		cell = terminal_macrostate(mudata = data,
									pseudotime_key = "rna:pseudotime", 
									cluster_key = "rna:pop",
									terminal_state = branch,
									n_cells=30)
		terminal_cells[branch] = cell

	atlas.run(connectivity_key = connectivity_key,
			threshold_scheme = threshold_scheme, 
			initial_states = early_cell, 
			terminal_states = terminal_cells)

	assert "initial_states" in mudata.uns	
	assert isinstance(mudata.uns["initial_states"], dict)
	assert len(mudata.uns["initial_states"]) == 1
	assert "terminal_states" in mudata.uns	
	assert isinstance(mudata.uns["terminal_states"], dict)
	assert len(mudata.uns["terminal_states"]) == len(terminal_cells)
	assert "intermediate_states" in mudata.uns
	assert not mudata.uns["intermediate_states"]
		
