###########################################################################
# Palantir Wrapper class to call Palantir routines using muon.MuData object
###########################################################################

import numpy as np 
import pandas as pd
import muon as mu
import scanpy as sc
import palantir
import scipy.stats as st

from muon import MuData
from anndata import AnnData
from src.utils import _check_keys
from palantir.presults import PResults
from scipy.sparse import csr_matrix, find
from typing import Optional, Union, List, Dict



class PalantirWrapper():
	def compute_kernel(self, data: Union[MuData, AnnData], 
						knn_key: str = "wnn",	
						distance_key: str="wnn_distances",  
						knn:Optional[int]=None, 
						alpha:float=0,
						kernel_key : str="DM_Kernel"):
		"""
		Adapted computation of the gaussian kernel that allows for muon.MuData obejct to be used and wnn. 
		Follows Palantir implementation.
		Params:
		--------
		data: muon.MuData or anndata.AnnData 
		knn_key: str
			String in data.uns where WNN parameters are set. By default "wnn".
		distance_key: str
			String in data.obsp where knn is stored. By default "wnn_distances". 
		knn: int
			Number of Nearest Neighbors. Default is None. 
		alpha: float 
			Normalization paramenter for the diffusion operator. By default 0.
		kernel_key: str
			Key to store the kernel in the obsp of the data. Default is "DM_Kernel".  
		"""
		if distance_key not in data.obsp.keys(): 
			raise KeyError(f"{distance_key} not in data.obsp")
		if knn is None:
			print(f"WARNING - knn parameter not specified. Looking for data.uns[{knn_key}][""params""][""n_neighbors""]")
			if knn_key not in data.uns.keys():
				raise KeyError(f"{knn_key} not in data.uns")	
			else:
				knn = int(data.uns[knn_key]["params"]["n_neighbors"])


		N = data.shape[0]
		kNN = data.obsp[distance_key]
		# By default Palantir computes the scaling factor as \sigma_i = distance to l-th neighbor, with l<k 
		# By default Palantir computes KNN neighbor from scratch using PCs the knn:int=30 parameter and n_pcs=0 (sc.pp.neighbors) and then sets l=knn/3
		# Here we expect either knn to be set, or to be searched in data.uns[knn_key]["params"]["n_neighbors"] as per muon.pp.neighbors construction.  
		adaptive_k = int(np.floor(knn/3)) #default by Palantir.  
		adaptive_std = np.zeros(N)
		for i in np.arange(N):
			adaptive_std[i] = np.sort(kNN.data[kNN.indptr[i] : kNN.indptr[i+1]])[adaptive_k - 1]
	
		x,y,dists = find(kNN) # x are row indices, y are column indices and dists the values of non zero entries in csr_matrix
		dists /= adaptive_std[x]
		W = csr_matrix((np.exp(-dists), (x,y)), shape=[N,N])
		kernel = W + W.T
		if alpha > 0:
			D = np.ravel(kernel.sum(axis=1))
			D[D!=0] = D[D!=0]**(-alpha)
			mat = csr_matrix((D, (range(N), range(N))), shape=[N,N])
			kernel = mat.dot(kernel).dot(mat)

		data.obsp[kernel_key] = kernel

	def run_diffusion_maps(self, data: Union[MuData, AnnData],
							n_components:int=10,
							seed: int = 52, 
							kernel_key: str = "DM_Kernel", 
							sim_key: str= "DM_Similarity",
							eigval_key: str = "DM_EigenValues",
							eigvec_key: str = "DM_EigenVectors"):

		"""
		Wrapper for palantir.utils.diffusion_maps_from_kernel. 
		Params:
		------
		data: muon.MuData or anndata.AnnData
		n_components: int 
			Number of Diffusion components to compute. By default 10.
		seed: int
			Numpy random seed. By deafult 0. 
		kernel_key: str
			Key in data.obsp containing the kernel. By default "DM_Kernel". 
		sim_key: str
			Key in data.obsp where the diffusion operator T is stored. By default "DM_Similarity".
		eigval_key: str
			Key in data.uns where to store the eigevalues. By default "DM_EigenValues".
		eigvec_key: str
			Key in data.obsm where to store the eigenvectors. By default "DM_EigenVectors". 
		"""
		if kernel_key not in data.obsp.keys():
			raise KeyError(f"{kernel_key} not in data.obsp")

		kernel= data.obsp[kernel_key]
		print("Calling Palantir diffusion_maps_from_kernel...")
		res = palantir.utils.diffusion_maps_from_kernel(data.obsp[kernel_key], n_components, seed)
		data.obsp[sim_key] = res["T"] 
		data.obsm[eigvec_key] = res["EigenVectors"].values
		data.uns[eigval_key] = res["EigenValues"].values


	def determine_multiscale_space(self, data: Union[MuData, AnnData],
									n_eigs: Optional[int]= None,
									eigval_key: str= "DM_EigenValues",
									eigvec_key: str= "DM_EigenVectors",
									out_key: str = "DM_EigenVectors_multiscaled"):
		"""
		Wrapper for palantir.utils.determine_multiscale_space. 
		Params:
		------
		data: mudata.MuData or anndata.AnnData
		n_eigs: int, optional
			Number of eigen vectors to use. If None the eigen gap heuristic is used. Default is None.
		eigval_key: str
			Key in data.uns where eigenvalues are stored. Default is "DM_EigenValues".
		eigvec_key: str
			Key in data.obsm where eigenvectors are stored. Default is "DM_EigenVectors":
		out_key: str
			Key in data.obsm where the results are stored. Default is "DM_EigenVectors_multiscaled".
		"""
		# Palantir constructs a multi-scale distance to construct a more reliable nearest neighbor graph G_E.
		# G_E is later used for pseudotime computations using waypoints. 
		if eigval_key not in data.uns.keys():
			raise KeyError(f"{eigval_key} not in data.uns")
		if eigvec_key not in data.obsm.keys():
			raise KeyError(f"{eigvec_key} not in data.obsm")
		
		eigenvectors = pd.DataFrame(data.obsm[eigvec_key], index=data.obs_names) if not isinstance(data.obsm[eigvec_key], pd.DataFrame) else data.obsm[eigvec_key] # for compatibility with Palantir framework
		dm_dict = {"EigenValues": data.uns[eigval_key], "EigenVectors": eigenvectors}
		result = palantir.utils.determine_multiscale_space(dm_res = dm_dict, n_eigs=n_eigs, eigval_key = eigval_key, eigvec_key = eigvec_key, out_key=out_key) # eigval_key, eigvec_key and out_key are not used 
		data.obsm[out_key] = result.values


	def run_palantir(self, data: Union[MuData, AnnData],
					early_cell,
					terminal_states: Optional[Union[List, Dict, pd.Series]]= None, 
					knn:int=30,
					num_waypoints:int=1200,
					n_jobs:int = -1,
					scale_components: bool = True,
					use_early_cell_as_start:bool = False,
					max_iterations: int= 25,
					eigvec_key: str = "DM_EigenVectors_multiscaled",
					pseudo_time_key: str = "palantir_pseudotime", 
					entropy_key: str = "palantir_entropy",
					fate_prob_key: str = "palantir_fate_probabilities", 
					save_as_df: bool = True, 
					waypoints_key: str = "palantir_waypoints",
					seed:int = 20):
						
		"""
		Wrapper for the palantir.core.run_palantir function.
		Params:
		-------
		data: muon.MuData or anndata.AnnData
		early_cell: 
			Early Cell specified by the user 
		terminal_states: list, dictionary or pandas.Series, optional
			User-defined terminal states in the form {terminal_name:cell_name}. Default is None.
		knn: int
			Number of nearest neighbors for graph construction. Default is 30.
		num_waypoints: int
			Number of waypoints to sample. Default is 1200.
		n_jobs: int 
			Number of jobs for parallel preprocessing. Default is -1.
		scale_components: bool
			If True components are scaled. Default is True. 
		use_early_cell_as_start: bool
			If True the early cell is used as start. Default is False.
		max_iterations: int
			Maximum number of iterations for pseudotime convergence. Default is 25.
		eigevec_key: str
			Key in data.obsm where the multiscale space diffusion compontents are stored. Default is "DM_EigenVectors_multiscaled".
		pseudo_time_key: str
			Key in data.obs where the pseudotime is stored. Default is "palantir_pseudotime".
		entropy_key: str
			Key in data.obs where the pseudotime is stored. Default is "palantir_entropy".
		fate_prob_key: str
			Key in data.obsm where the fate probabilities are stored. Default is "palantir_fate_probabilities".
		save_as_df: bool
			If True, then the fate probabilities are stored into a dictionary with columns names corresponding to the terminal states.
			Otherwise they are sotred as a numpy array and the terminal states names are stored in uns[fate_probability_key]. Default is True. 
		waypoints_key: str
			Key is data.uns where to store the waypoints. Default is "palantir_waypoints". 
		seed: int 
			Seed for waypoint sampling. Default is 20.
		"""
		if not eigvec_key in data.obsm.keys():
			raise KeyError(f"{eigvec_key} not in data.obsm")

		# Palantir either requires an AnnData object or pandas DataFrame. Using this latter sturcture to avoid data management issues. 
		input_df = pd.DataFrame(data.obsm[eigvec_key], index = data.obs_names)	

		#knn is used to construct a graph G_E connecting cells based on  multiscale distances. G_E used to compute pseudotime.
		res = palantir.core.run_palantir(data=input_df, early_cell=early_cell, terminal_states=terminal_states, knn=knn, num_waypoints=num_waypoints,
				n_jobs=n_jobs, scale_components=scale_components, use_early_cell_as_start=use_early_cell_as_start, max_iterations = max_iterations, 
				eigvec_key = eigvec_key, pseudo_time_key=pseudo_time_key, entropy_key=entropy_key, fate_prob_key=fate_prob_key, save_as_df=save_as_df,
				waypoints_key = waypoints_key, seed=seed)

		data.obs[pseudo_time_key] = res.pseudotime
		data.obs[entropy_key] = res.entropy
		data.uns[waypoints_key] = res.waypoints.values
		if isinstance(terminal_states, pd.Series):
				res.branch_probs.columns = terminal_states[res.branch_probs.columns]
		if save_as_df:
			data.obsm[fate_prob_key] = res.branch_probs
		else:
			data.obsm[fate_prob_key]= res.branch_probs.values
			data.uns[fate_probs_key + "_columns"] = res.branch_probs.columns.values


	def compute_priming_degree(self, data:Union[MuData, AnnData], fate_prob_key:str = "palantir_fate_probabilities", entropy_type: str = "entropy", early_cells: Optional[np.ndarray]=None):
		"""
		Function that computes KL-divergence and entropy as in CellRank.
		Given cell i and fates probabilities towards the k-th terminal state pk, CellRank defines the entropy measure as in Setty et al (palantir paper).
		Shannon entropy defined as H_i = -sum(pk, * log(pk))

		Given cell i and fates probabilities towards the k-th terminal state pk, CellRank KL divergence measures how far the fate distribution is
		from the average fate distribution. 
		
		
		Params:
		--------
		- data: muon.MuData or anndata.AnnData
		- fate_prob_key: str
			Key in data.obsm where fates probabilities towards terminal states are stored. Default is "palantir_fate_probabiltities". 
		- entropy_type: str
			Which type of  entropy measure to use, either "kl_divergence" or "entropy". Default is "entropy".
		- early_cell: np.ndarray, optional
			Contains the cell id or a masking that defines the starting cells. Default is None, meaning all cells are used. 
		"""
		if fate_prob_key not in data.obsm.keys():
			raise KeyError(f"{fate_prob_key} not in data.obsm")
		probabilities = data.obsm[fate_prob_key]
		if entropy_type == "entropy":
			entropy = st.entropy(pk=probabilities, qk=None, axis=1) 
			entropy = np.max(probs) - probs
		elif entropy_type == "kl_divergence":
			early_subset = np.ones((len(probabilities),), dtype=bool) if early_cells is None else early_cells
			early_probabilities = probabilities.iloc[early_subset]
			entropy = np.nan_to_num(
						np.sum(probabilities * np.log2(probabilities/np.mean(early_probabilities, axis=0)), axis=1), 
						nan = 1.0,
						copy = False
					)
		else:
			raise ValueError(f"{entropy_type} must be either entropy or kl_divergence")
		minn, maxx = np.min(entropy), np.max(entropy)
		return (entropy - minn) / (maxx - minn)
	



		 


def results_to_dataframe(data:MuData, entropy_key:str ="entropy", pseudo_time_key:str="palantir_pseudotime", fate_prob_key:str="fates", modality_key:str=None, **kwargs):
	group_key = kwargs["group_key"] if "group_key" in kwargs else None

	_check_keys(data, modality_key = modality_key, entropy_key = entropy_key, pseudo_time_key=pseudo_time_key, obs_key=group_key, fate_prob_key = fate_prob_key)

	true_key = kwargs["true_pseudotime"] if "true_pseudotime" in kwargs else None
	_check_keys(data, modality_key = modality_key, obs_key = true_key)

	data = data if modality_key is None else data[modality_key]

	columns = [entropy_key, pseudo_time_key]
	if group_key is not None:
		columns = columns + [group_key]
	if true_key is not None:
		columns = columns + [true_key]	
	dataframe = data.obs[columns].copy()
	dataframe = pd.concat((dataframe, data.obsm[fate_prob_key]), axis=1)

	return dataframe
