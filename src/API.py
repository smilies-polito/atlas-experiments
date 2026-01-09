import inspect
import numpy as np
import pandas as pd
from muon import MuData
from anndata import AnnData
from abc import ABC, abstractmethod
from typing import Optional, Literal, List, Dict, Any, Union, Type



class WrapperBase(ABC):
	def __init__(self, mudata:MuData, **kwargs:Any):
		self._mudata = mudata
		super().__init__(**kwargs)

	@abstractmethod
	def run(self, **kwargs: Any) -> Any: 
		...
		
	@property
	def mudata(self):
		return self._mudata



class PalantirWrapper(WrapperBase):
	def __init__(self, mudata:MuData, **kwargs:Any):
		super().__init__(mudata=mudata, **kwargs)


	def compute_kernel(self, 
						knn_key:str = "wnn",
						distance_key:str = "wnn_distances",
						knn: Optional[int] = None,
						alpha: float = 0,
						kernel_key: str="DM_Kernel"):
		'''
		Adapted computation of the gaussian kernel allowing for muon.MuData objects. 
		Follows palantir implementation.
		Parameters:
			- knn_key: str; in mudata.uns where the WNN parameters are set. Default is "wnn".
			- distance_key: str; key in mudata.obsp where knn graph is stored. Default is "wnn_distances".
			- knn: int; number of nearest neighbors. Default is None (estimated from parameters).
			- alpha: float; Normalization parameter for the diffusion operator. Default is 0.
			- kernel_key: str; Key in obsp where to store the kernel; Default is "DM_Kernel".
		Adds a scipy.sparse.csr_matrix in mudata.obsp slot.
		'''
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
		adaptive_k = int(np.floor(knn/3))
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


	def compute_diffusion_map(self, 
								kernel_key: str="DM_Kernel",
								sim_key: str= "DM_Similarity",
								eigval_key: str = "DM_EigenValues",
								eigvec_key: str = "DM_EigenVectors",
								n_components: int=10,
								seed: Union[int,None] = 42):
		'''
		Wrapper for palantir.utils.diffusion_maps_from_kernel.
		Parameters:
			- kernel_key: str; Key in mudata.obsp containing the kernel. Default is "DM_Kernel".
			- sim_key: str; Key in mudata.obsp where the diffusion operator is stored. Default is "DM_Similarity".
			- eigval_key: str; Key in mudata.uns where the eigenvalues are stored. Default is "DM_EigenValues".
			- eigvec_key: str; Key in mudata.obsm where the eigenvectors are stored. Default if "DM_EigenVectors". 
			- n_components: int; Number of diffusion components to compute. Default is 10. 
			- seed: int; random seed. Default is 0.
		Updates MuData object with the results from diffusion maps.
		'''
		if kernel_key not in data.obsp.keys():
			raise KeyError(f"{kernel_key} not in data.obsp")

		kernel= data.obsp[kernel_key]
		res = palantir.utils.diffusion_maps_from_kernel(data.obsp[kernel_key], n_components, seed)
		data.obsp[sim_key] = res["T"] 
		data.obsm[eigvec_key] = res["EigenVectors"].values
		data.uns[eigval_key] = res["EigenValues"].values

	def compute_multiscale_space(self,
									n_eigs: Optional[int]=None, 
									eigval_key: str = "DM_EigenValues",
									eigvec_key: str = "DM_EigenVectors",
									out_key: str = "DM_EigenVectors_multiscaled"):
		'''
		Wrapper for palantir.utils.determine_multiscale_space.
		Parameters:
			- n_eigs: int, optional; number of eigenvectors to use; if none eigengap euristic is used. Default is None.
			- eigval_key: str; Key in mudata.uns storing eigenvalues. Default is "DM_EigenValues".
			- eigvec_key: str; Key in mudata.obsm storing eigenvectors. Default is "DM_EigenVectors". 
			- out_key: str; Key in mudata.obsm where results are stored. Default is "DM_EigenVectors_multiscaled"
		'''
		if eigval_key not in data.uns.keys():
			raise KeyError(f"{eigval_key} not in data.uns")
		if eigvec_key not in data.obsm.keys():
			raise KeyError(f"{eigvec_key} not in data.obsm")
		
		eigenvectors = pd.DataFrame(data.obsm[eigvec_key], index=data.obs_names) if not isinstance(data.obsm[eigvec_key], pd.DataFrame) else data.obsm[eigvec_key] # for compatibility with Palantir framework
		dm_dict = {"EigenValues": data.uns[eigval_key], "EigenVectors": eigenvectors}
		result = palantir.utils.determine_multiscale_space(dm_res = dm_dict, n_eigs=n_eigs, eigval_key = eigval_key, eigvec_key = eigvec_key, out_key=out_key) # eigval_key, eigvec_key and out_key are not used 
		data.obsm[out_key] = result.values


	def compute_priming_degree(self,
								fate_prob_key: str = "palantir_fate_probabilities",
								entropy_type: Literal["entropy", "kl-divergence"] = "entropy"):
		'''
		Function that computes KL-divergence and entropy as in CellRank. 
		Parameters:
			- fate_prob_key: str; Key in mudata.obsp/uns/obsm where to look for fate probabilities. Default is "palantir_fate_probabilities".
			- entropy_type: str; Key identifying whether to compute Shannon entropy or KL-divergence. Accepts either "entropy" or "kl-divergence". Default is "entropy".
		'''	
		# Guarda se ho dataframe in mudata.obsm[fate_prob_key] o se ho obsp + uns (dipende da self.run save_as_df=
		# checj entropy_type diverso da quelli listati: problema.
		pass		

	
	def run(self, *,
			early_cell: str,
			terminal_states: Optional[Union[List, Dict, pd.Series]] = None,
			knn: int=30,
			num_waypoints: int = 1200,
			n_jobs:int = -1,
			scale_components: bool= True, 
			use_early_cell_as_start: bool= False,
			max_iterations: int=25,
			n_components:int = 10,
			alpha: float = 0,
			n_eigs: Optional[int]=None,
			knn_key: str = "wnn",
			distance_key: str = "wnn_distances",
			sim_key: str = "DM_Similarity", 
			kernel_key: str = "DM_Kernel",
			eigvec_key: str = "DM_EigenVectors",
			eigvec_multi_key: str = "DM_EigenVectors_multiscaled",
			eigval_key: str = "DM_Eigenvalues",
			pseudotime_key: str = "palantir_pseudotime",
			entropy_key: str = "palantir_entropy",
			fate_prob_key: str = "palantir_fate_probabilities",
			waypoints_key: str = "palantir_waypoints",
			save_as_df: bool = True, 
			compute_kernel: bool = True,
			compute_diffusion_maps: bool = True,
			seed:int = 42, **kwargs: Any):
		'''
		Runs Palantir on MuData object. 
		Eventually computes kernel when the compute_kernel key is True, otherwise a predefined kernel must be present in mudata.obsp[kernel_key]. 
		Eventually computes the multiscaled space if compute_diffusion_maps is True. This requires a kernel to be present in mudata.obsp[kernel_key], otherwise provide the precomputed multiscaled space in mudata.obsm[eigenvec_multi_key] to estimate psedutime and trajectories. 

		Parameters:
			- early_cell: str; early_cell specified by the user. 
			- terminal_states: list, dictionary or pandas.Series; User-defined terminal states in the form {"terminal_name: cell_name}. Default is None.
			- knn: int; Number of neighbors in KNN graph construction among waypoints. Default is 30.
			- num_waypoints: int; Number of waypoints to sample. Default is 1200.
			- n_jobs: int; number of jobs. Default is -1.
			- scale_components: bool; If true components are scaled. Default is True.
			- use_early_cell_as_start: bool; If True, then the early cell is used as starting point. Default False. 
			- max_iterations: int; Maximum nuber of iterations for pseudotime convergence. Default is 25. 
			- n_components: int; Number of components to estimate for diffusion maps (see check_diffusion_maps). Defaults is 10. 
			- alpha: float; Normalization parameter for the diffusion operator when diffusion maps need to be computed (see compute_diffusion_maps). Default is 0.  
			- n_eigs: int, optional; Number of eigenvalues to use to determine the multiscale space (see parameter compute_diffusion_map). Only used when diffusion maps need to be computed and if not provided the eigen gap heuristic is used.
			- knn_key: str; Key in mudata.uns where the graph parameters for kernel computation are stored (check parameter compute_kernel). Default is "wnn". 
			- distance_key: str; Key in mudata.obsp where distances are stored and used to compute the gaussian kernel. Check compute_kernel parameter to identify whether the kernel is to be computed. Default is "wnn_distances".
			- sim_key: str; Key in data.obsp where to store the diffusion operator if diffusion maps are estimates (see parameter compute_diffusion_maps). Default is "DM_Similarity".
			- kernel_key: str; Key in mudata.obsp where the precomputed kernel for diffusion maps estimation is located or location where to store the kernel (check parameter precomputed_kernel)
			- eigvec_key: str; Key in mudata.obsm where the eigenvectors are stored. Used only to determine the multiscale space (see parameter compute_diffusion_maps). Default is "DM_EigenVectors".
			- eigvec_multi_key: str; Key in mudata.obsm where the multiscale space is stored. It is either computed (see parameter compute_duffsion_maps) or it needs to be pre-computed and stored in mudata.obsm. Default is "DM_EigenVectors_multiscaled".
			- eigval_key: str; Key in mudata.uns where the eigenvalues are stored. Used only to determine the multiscale space (see parameter compute_diffusion_maps). Default is "DM_EigenValues". 
			- pseudotime_key: str; key in mudata.obs where pseudotime values for each cell are stores. Default is "palantir_pseudotime".
			- entropy_key: str; Key in mudata.obs where entropy for each cell is stored. Default is "palantir_entropy".
			- fate_prob_key: str; Key in mudata.obsm/data.obsp/data.uns where to store fate probabilities (check parameter save_as_df).
			- waypoints_key: str; Key in mudata.uns where to store the waypoints. Default is "palantir_waypoints".
			- save_as_df: bool; If true then fate probabilities towards the terminal states are stored as a pd.DataFrame; otherwise they are stored as a numpy.array and terminal states names are in mudata.uns.
			- compute_kernel: bool; Whether to compute Gaussian Kernel from distance matrix. Default is True.If False, please provide in "kernel_key" the key to access the kernel for diffusion maps computations.
			- compute_diffusion_maps: bool; Whether to compute diffusion maps and multiscaled distances. Defauls is True. If False, please provide in "eigenvec_multi_key" the key to access the multiscales space to correctly run pseudotime and fate probabilities estimation. 
			- seed: int; random state seed. Default is 42.
		Output:
			MuData object with palantir results.
		'''
		pass	
		# Se devo calcolare il kernel (compute_kernel è True) allora chiamare "self.compute_kernel" e passargli knn_key
		# Se devo calcolare i diffusion maps controlla che ci sia un kernel precalcolato o che ci sia compute_kernel = True. Calcolare i diffusion maps significa richiamare sia compute_diffusion_maps che multiscale_space.
		# palantir.run 
		# self.compute_priming_degree
	 

class GPCCAWrapper:
	def __init__(self, forward:bool):
		self.forward = forward


class PseudotimeKernelWrapper(WrapperBase, GPCCAWrapper):
	def __init__(self, 
					mudata:MuData, 
					pseudotime_key:str="pseudotime",
					connectivity_key: str = "wnn_connectivities",
					backward:bool=False):
		'''
		Parameters:
		- mudata: MuData; muon object containing modalities for trajetory inference.
		- pseudotime_key: str; Key in mudata.obs where pseudotime for every cell is computed. Default is "pseudotime". 
		- connectivity_key: str; Key in mudata.obsm where knn connectivites are stored. Default is "wnn_connectivities".
		- backward: bool; Indicating whether forwards or backward direction needs to be identified. Defaults is False.
		'''
		
		super().__init__(mudata=mudata, forward=forward)
		self.pseudotime_key = pseudotime_key
		# creare oggetto AnnData per cellrank kernel
		# che contiene: adata.obsm["connectivities"] le connectivities del wnn cioè mudata.obsm["wnn_connectvitiies"]
		# adata.obs deve contenere adata.obs[pseudotime_key] lo pseudotime per ogni cellula. 
		# creo oggetto cellrank.kernel.PseudotimeKernel 
		# 

	def run(self,
			threshold_scheme: Literal["soft", "hard"] = "hard", 
			frac_to_keep: float = 0.3, 
			b: float = 10.0, nu: float = 0.5, 
			):
		'''
		Computes transition matrix and then runs GPCCA. 
		Parameters:
			- threshold_scheme: str; either "soft" or "hard", defines which method for biasing the graph to use. The "soft" scheme is based on *VIA* and it down weights edges that points against the direction of increasing pseudotime. The "hard" scheme is based on *Palantir* and it removes edges that points against the direction of increasing pseudotime, keeping the ones within a radius in order not to disconnect the graph. 
			- frac_to_keep: float; fraction of nearest neighbors according to local connectivity that is kept independently from pseudotime (radius in threshold_scheme=="hard" see parameter "threshold_scheme"). Default is 0.3.
			- b, nu: float; respectively the ---- and --- used when threshold_scheme is "soft". Default are 10.0 and 0.5. 
		'''
		# to do 
		pass



class WrapperCreator:
	'''
	Inferface that creates the corresponding method instance for pseudotime computation.
	'''
	_MAP: Dict[str, Type[WrapperBase]] = {}
	
	@classmethod
	def create(cls, 
				wrapper_type: str, 
				mudata: MuData,	
				*args:Any, 
				**kwargs:Any) -> WrapperBase:
		'''
		Function creating the correct method instance.
		Parameters:
			- wrapper_type: str; string identifying which interface to use. Either "pseudotime-kernel" or "palantir". 
			- mudata: muon.MuData; Contains multimodal data for trajectory computations.
		'''
		try: 
			wrp = cls._MAP[wrapper_type]
		except KeyError:
			available = ", ".join(sorted(cls._MAP.keys()))
			raise ValueError(f"{wrapper_type} not recognized. Try: {available}")

		provided = dict(kwargs)
		provided["mudata"] = mudata
		try:
			sig = inspect.signature(wrp.__init__)
		except(ValueError, KeyError):
			return wrp(mudata, *args, **kwargs)

		allowed_names = { name for name, param in sig.parameters.items() if name != "sefl" and param.kind in (
								inspect.Parameter.POSITIONAL_OR_KEYWORD,
								inspect.Parameter.KEYWORD_ONLY 
							)}
		filtered_kw = {k: v for k,v in provided.items() if k in allowed_names}
		try: 
			return wrp(**filtered_kw)
		except:
			filtered_no_mudata = {k: v for k, v in filtered_kw.items() if k!='mudata'}
			try: 
				return wrp(mudata, *args, **filtered_no_mudata)
			except TypeError as e:
				raise TypeError(f"Failed to instantiate '{wrapper_type}'")
	
	

WrapperCreator._MAP: Dict[str, Type[WrapperBase]] = {
			"palantir": PalantirWrapper,
			"pseudotime-kernel": PseudotimeKernelWrapper
	}





class Classe:
	def __init__(self, 
				mudata: MuData, 
				fragment_path: Optional[str] = None,
				random_state: int=42
		):	
		'''
		Class initialization
		Parameters
			- mudata: MuData; object containing two modalities "rna" and "atac".
			- fragment_path: optional, str; path to the fragments.gz.tsv and fragments.gz.tsv.tbi files storing fragment information for scATAC-seq data.
			- random_state: int; seed for computation reproducibility. Default is 42.
		'''
		# check on modality existence
		# add fragment path to mudata["atac"] if specified. 
		pass

	def preprocessing(self, 
					min_counts: Optional[int]=None, 
					max_counts: Optional[int]= None, 
					mito_percent: Optional[float]=None,
					tss_score:Optional[float]=None, 
					nucleosome_score:Optional[float]=None, 
					highly_variable_genes: Optional[int] = None,
					target_sum: Optional[float] = None, 
					n_pcs_rna: int= 30,
					n_pcs_act: int = 10,
					knn_rna: int=30,
					knn_act: int=30, 
					wnn: int=30,
					stranded: bool = False,
					features: Optional[pd.DataFrame] = None,
					batch_strategy: Optional[Literal["harmony", "bbknn"]]= None,
					**kwargs
		):
		'''
		scRNA-seq and scATAC-seq standard preprocessing pipeline, including:
		qc metrics, filtering, batch correction, PCA, neighboring graph, gene activity computation and wnn computation.
		Parameters:
			- min_counts, max_counts: optional, int; for scRNA-seq data threshold defining the minimum and maximum number of counts to filter for in quality control. 
			- mito_percent: float, optional; for scRNA-seq data threshold defining the maximum mitochondrial RNA percentage for quality control.  
			- tss_score: float, optional; tss erichment lower bound for scATAC-seq QC metrics filtering. If not specified, then retained above 10-th percentile. 
			- nuclesome_score: float, optional; nucleosome signal upper bound for scATAC-seq QC metrics filterins. If not specified, then retained less than 80-th percentile.
			- highly_variable_genes: int, optional; number of highly variable genes to retain in scRNA-seq data.  
			- target_sum: int, optional; target sum for normalization in scRNA-seq data and activity data.
			- n_pcs_rna: int; number of PCs to retain in scRNA-seq data. Default is 30.
			- n_pcs_act: int; number of PCs to retain in activity data. Deafult is 10. 
			- knn_rna: int; number of nearest neighbors for scRNA-seq KNN graph computation. Default is 30.
			- knn_act: int; number of nearest neighbors for activity KNN graph computation. Default is 30.
			- wnn: int; number of nearest neighbors per modality to consider in wnn computation. Default is 30. 
			- stranded: bool; whether to consider strand in computing gene activity. Default is False. 
			- features: pd.Dataframe, optional; dataframe containing "chromosome", "start", "end", "strand" for genes considered in activity computation. 
			- batch_strategy: string, optional. Key ("harmony" or "bbknn") indicating the strategy for batch effect removal. When specified it requires scanpy parameters to be specified in the kwargs under the same key. i.e., it lood for kwargs["harmony"] or kwargs["bbknn"].
		Returns:
			Updated the muon object with modality "activity". 
			No return 
	
		'''
		# calcolare qc metrics inplace
		# filtrare cellule per max min counts
		# filtrare cellule per mitopercent -> solo se mitopercent soglia specificata
		# check dei controlli sui None ovviamente inutile passare a funzione parametri None. vedi documentazione scanpy/muon
		# RNA qc and filtering
		# ATAC qc and filtering
		# activity computation + normalization + PCA
		# knn normale o bbknn o harmony e knn, dipende dalla strategia di batch correction se specificata
		# wnn multimodale rna e attività
		# umap 
		# al momento fare una pipeline di preprocessing generica, poi in futuro usare una struttura da passare come kwargs per preprocessare il tutto 
		pass


	def get_data(self) -> MuData:
		'''
		Returns the MuData object
		'''		
		pass


	def run(self, modality:Literal["pseudotime-kernel", "palantir"]="palantir"):
		'''
		run pseudotime and fate probabilities
		modality: str, which algorithm to call. either palantir or cellrank pseudotime kernel.
		'''
		# Usare Wrapper Creator per creare l'oggetto corrispondente alla modalità 
		# Check wrapper non sia None
		# Se wrapper è PalantirWrapper: run di palantir
		# Se wrapper è PseudotimeKernelWrapper: run di cellrank pseudotime
		pass	
		
