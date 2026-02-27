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
	knn_rna, knn_activity, wnn = 30, 30, 30
	n_pcs_rna, n_pcs_activity = 20, 10
	
	data_path = os.path.join(working_directory, "tests", "test_data")
	activity = pd.read_csv(os.path.join(data_path, f"activity.tsv"), sep="\t", header=0, index_col=0)
	rna = pd.read_csv(os.path.join(data_path, f"rna.tsv"), sep="\t", header=0, index_col=0)
	metadata = pd.read_csv(os.path.join(data_path, f"metadata.tsv"), sep="\t", header=0, index_col=0)

	# create activity matrix
	activity = AnnData(X=csr_matrix(activity.values), 
			obs = pd.DataFrame(data=None, index=activity.index.values, columns=None), 
			var = pd.DataFrame(data=None, index=activity.columns, columns=None))	
	sc.pp.normalize_total(activity)
	sc.pp.pca(activity, random_state=seed, use_highly_variable=False)
	# create rna matrix
	rna = AnnData(X=csr_matrix(rna.values.T), 
			obs=pd.DataFrame(data=None, index=rna.columns, columns=None), 
			var=pd.DataFrame(data=None, index=rna.index.values, columns=None))
	rna.obs = rna.obs.merge(metadata, how="left", left_index=True, right_index=True)
	sc.pp.normalize_total(rna)
	sc.pp.log1p(rna)
	sc.pp.pca(rna, random_state=seed, use_highly_variable=False)

	data = MuData({"rna":rna, "activity":activity})

	# ground truth construction
	early_cell = initial_macrostate(mudata = data, 
									pseudotime_key = "rna:pseudotime", 
									n_cells=1)
	n_cells, n_rna, n_activity = data.n_obs, data["rna"].n_vars, data["activity"].n_vars

	atlas = ATLAS(mudata = data,
					method= "palantir",
					fragment_path = None,
					random_state = seed)

	assert atlas._impl.rna_key == "rna"
	assert atlas._impl.activity_key == "activity"
	assert atlas._impl.atac_key is None
	assert atlas._impl.use_activity 
	assert atlas._impl.random_state == seed
	assert isinstance(atlas._impl.mudata, MuData)
	assert not hasattr(atlas._impl.mudata["activity"].uns, "files")


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
	kernel_key, eigval_key, eigvec_key, sim_key, out_key, waypoint_key  = "kernel", "eigvals", "eigvecs", "similarity", "multiscale", "waypoints"
	n_components, num_waypoints = 5, 250
	atlas.run(early_cell = early_cell[0],
				cluster_key = "rna:pop",
				terminal_states = None,
				n_components= n_components,
				kernel_key = kernel_key,
				eigval_key = eigval_key,
				eigvec_key = eigvec_key,
				sim_key = sim_key,
				eigvec_multi_key = out_key,
				waypoints_key = waypoint_key,
				knn = 30,
				num_waypoints = num_waypoints)
	assert kernel_key in mudata.obsp
	assert isinstance(mudata.obsp[kernel_key], csr_matrix)
	assert mudata.obsp[kernel_key].shape == (n_cells, n_cells)
	assert sim_key in mudata.obsp
	assert mudata.obsp[sim_key].shape == (n_cells, n_cells)
	assert eigvec_key in mudata.obsm 
	assert mudata.obsm[eigvec_key].shape == (n_cells, n_components)
	assert eigval_key in mudata.uns
	assert out_key in mudata.obsm
	assert mudata.obsm[out_key].shape[1] <= n_components
	assert waypoint_key in mudata.uns
	assert len(mudata.uns[waypoint_key]) <= num_waypoints
	assert "pseudotime" in mudata.obs.columns	
	assert "shannon_entropy" in mudata.obs.columns	
	assert "kl_divergence" in mudata.obs.columns	
	assert waypoint_key in mudata.uns
	assert "fate_probabilities" in mudata.obsm 
	assert isinstance(mudata.obsm["fate_probabilities"], pd.DataFrame)
	assert "initial_states" in mudata.uns
	assert isinstance(mudata.uns["initial_states"], dict)
	assert "terminal_states" in mudata.uns
	assert isinstance(mudata.uns["terminal_states"], dict)
	assert hasattr(atlas._impl, "fate_probability_key")
	assert atlas._impl.fate_probability_key == "fate_probabilities"
	assert "fate_state_colors" in mudata.uns
	n_states = len(mudata.uns["initial_states"]) + len(mudata.uns["terminal_states"])
	assert len(mudata.uns["fate_state_colors"]) == n_states

	# test fixed terminal states 
	terminal_cells = []
	for branch in ["4_1", "5_2", "5_3"]:
		cell = terminal_macrostate(mudata = data,
									pseudotime_key = "rna:pseudotime", 
									cluster_key = "rna:pop",
									terminal_state = branch,
									n_cells=1)			
		terminal_cells.extend(cell)
	atlas.run(early_cell = early_cell[0], 
				cluster_key = "rna:pop",
				terminal_states = terminal_cells,
				knn = 30,
				num_waypoints = 250)
			
	assert "pseudotime" in mudata.obs.columns	
	assert "shannon_entropy" in mudata.obs.columns	
	assert "kl_divergence" in mudata.obs.columns	
	assert waypoint_key in mudata.uns
	assert "fate_probabilities" in mudata.obsm 
	assert isinstance(mudata.obsm["fate_probabilities"], pd.DataFrame)
	assert "initial_states" in mudata.uns
	assert isinstance(mudata.uns["initial_states"], dict)
	assert "terminal_states" in mudata.uns
	assert isinstance(mudata.uns["terminal_states"], dict)
	assert len(mudata.uns["terminal_states"]) == len(terminal_cells)
	assert hasattr(atlas._impl, "fate_probability_key")
	assert atlas._impl.fate_probability_key == "fate_probabilities"
	assert "fate_state_colors" in mudata.uns
	n_states = len(mudata.uns["initial_states"]) + len(mudata.uns["terminal_states"])
	assert len(mudata.uns["fate_state_colors"]) == n_states


