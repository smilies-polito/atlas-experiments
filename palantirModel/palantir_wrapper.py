###########################################################################
# Palantir Wrapper class to call Palantir routines using muon.MuData object
###########################################################################

import warnings
import muon as mu
import scanpy as sc
import palantir
import scipy.stats as sc
from .palantir_environment import *
from palantir.presults import PResults
from scipy.sparse import csr_matrix, find



def _check_keys(data: Union[AnnData, MuData], embedding_basis: Optional[str]= None, pseudo_time_key: Optional[str]= None, entropy_key: Optional[str]= None, fate_prob_key: Optional[str]=None):
	if embedding_basis is not None and embedding_basis not in data.obsm.keys():
		raise KeyError(f"{embedding_basis} not in data.obsm")
	if pseudo_time_key is not None and pseudo_time_key not in data.obs.columns:
		raise KeyError(f"{pseudo_time_key} not in data.obs")
	if entropy_key is not None and entropy_key not in data.obs.columns:
		raise KeyError(f"{entropy_key} not in data.obs")
	if fate_prob_key is not None and fate_prob_key not in data.obsm.keys():
		raise KeyError(f"{fate_prob_key} not in data.obsm")




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
			warnings.warn(f"knn parameter not specified. Looking for data.uns[{knn_key}][""params""][""n_neighbors""]")
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
			mat = csr_matrix((D, (range(N). range(N))), shape=[N,N])
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
									n_eigs: Union[int, None]= None,
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


	def compute_priming_degree(data:Union[MuData, AnnData], fate_prob_key:str = "palantir_fate_probabilities", entropy_type: str = "entropy", early_cells: Optional[np.ndarray]=None):
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
			entropy = sc.entropy(pk=probabilities, qk=None, axis=1) 
		elif entropy_type == "kl_divergence":
			early_subset = np.ones() if early_cell is None else early_cells
			early_probabilities = probabilities[early_subset, :]
			entropy = np.nan_to_num(
						np.sum(probabilities * np.log2(probs/np.mean(early_probabilities, axis=0)), axis=1), 
						nan = 1.0,
						copy = False
					)
		else:
			raise ValueError(f"{entropy_type} must be either entropy or kl_divergence"}
		
		return entropy
	

	def plot_palantir_results(self, data: Union[MuData, AnnData],
								modality_key: Optional[str] = None,  
								embedding_basis: str = "X_umap",
								pseudo_time_key: str = "palantir_pseudotime", 
								entropy_key: str= "palantir_entropy",
								fate_prob_key: str = "palantir_fate_probabilities"):
						
		"""
		Function that wraps palantir.plot.plot_palantir_results. It plots the palantir results over the multiomics embedding for comparison purposes,
		also when modality_key is specified. 
		Params:
		------
		data: muon.MuData or anndata.AnnData	
		modality_key: str, optional
			If specified retrieves palantir run from specific modality and plots on multiomics embedding. Default is None.
		embedding_basis: str
			Key in data.obsm with the embedding to be plot. Default is "X_umap". 
		pseudo_time_key: str
			Key in data.obs where palantir pseudotime is stored. Default is "palantir_pseudotime".
		entropy_key : str
			Key in data.obs where palantir entropy is stored. Default is "palantir_entropy".
		fate_prob_key: str
			Key in data.obsm to access fate probabilities. Default is "palantir_fate_probabilities".
		"""
		if modality_key is None:
			print("No modality has been specified, using MuData Palantir run.")
			_check_keys(data, embedding_basis=embedding_basis, pseudo_time_key=pseudo_time_key, entropy_key=entropy_key, fate_prob_key=fate_prob_key)
			palantir_results = PResults(data.obs[pseudo_time_key], data.obs[entropy_key], data.obsm[fate_prob_key], None)
		else:
			if modality_key not in data.mod.keys():
				raise KeyError(f"Modality {modality_key} not in data")
			print(f"Modality specified. Using {modality_key} Palantir run")
			_check_keys(data, embedding_basis=embedding_basis)
			_check_keys(data.mod[modality_key], pseudo_time_key=pseudo_time_key, entropy_key=entropy_key, fate_prob_key=fate_prob_key)  		
			palantir_results = PResults(data.mod[modality_key].obs[pseudo_time_key], data.mod[modality_key].obs[entropy_key], 
							data.mod[modality_key].obsm[fate_prob_key], None)

		embedding_data = pd.DataFrame(data.obsm[embedding_basis], index=data.obs_names)
		
		return palantir.plot.plot_palantir_results(data=embedding_data, pr_res = palantir_results)

	
		 



				
