import os
import scipy
import inspect
import warnings
import palantir
import muon as mu
import numpy as np
import scanpy as sc
import pandas as pd
import scvelo as scv
import scFates as scf
import cellrank as cr
import matplotlib.pyplot as plt
from muon import MuData
from anndata import AnnData
from scipy.sparse import csr_matrix, find
from abc import ABC, abstractmethod
from matplotlib.colors import to_hex
from cellrank.estimators import GPCCA
from cellrank.kernels import PseudotimeKernel
from cellrank._utils._lineage import Lineage
from .utils import _assign_state_colors, MultiBranchGAM
from typing import Optional, Literal, List, Dict, Any, Union, Type, Sequence


class Base(ABC):
	def __init__(self, 
			mudata: MuData, 
			fragment_path: Optional[str] = None,
			random_state: int=42,
			**kwargs
		):	
		'''
		Class initialization
		Parameters
			- mudata: MuData; object containing two modalities "rna" and "atac".
			- fragment_path: optional, str; path to the fragments.gz.tsv and fragments.gz.tsv.tbi files storing fragment information for scATAC-seq data.
			- random_state: int; seed for computation reproducibility. Default is 42.
		'''
		existing_modalities = [mod for mod in mudata.mod.keys()]
		self.rna_key = next((mod for mod in mudata.mod.keys() if mod.lower()=="rna"), None) 
		self.atac_key = next((mod for mod in mudata.mod.keys() if mod.lower()=="atac"), None) 
		self.activity_key = next((mod for mod in mudata.mod.keys() if mod.lower()=="activity"), None) 
		
		if self.rna_key is None:
			raise KeyError(f"rna modality is missing")
		if self.atac_key is None and self.activity_key is None:
			raise KeyError("Both atac and activity modalities are missing. Please provide at least one modality.")
		if self.atac_key is not None and self.activity_key is not None:
			warnings.warn("Both ATAC and ACTIVITY modalities are specified. Only ACTIVITY is used.")			
			self.atac_key = None
		
		self.use_activity = self.activity_key is not None
			
		if fragment_path is not None and self.atac_key is not None:
			if self.atac_key != "atac":
				# compatibility operation for muon.tl.locate_file
				mudata.mod["atac"] = mudata.mod[self.atac_key]
				del mudata.mod[self.atac_key]
				self.atac_key = "atac"

			files = mudata.mod[self.atac_key].uns.get("files", {})
			if "fragments" not in files:
				mu.atac.tl.locate_file(mudata.mod[self.atac_key], file= fragment_path, key="fragments")

		elif fragment_path is not None and self.use_activity: 
				warnings.warn("Fragment path provided not used since activity modality is already present.")

		self.random_state = random_state
		self.mudata = mudata	
		return

	def preprocessing(self, 
			n_pcs_rna: int= 30,
			n_pcs_act: int = 10,
			knn_rna: int=30,
			knn_act: int=30, 
			use_rep: Optional[str] = None,
			n_neighbors: int=30,
			n_bandwidth_neighbors: int=20, 
			n_multineighbors: int = 200,
			metric: Literal = "euclidean",
			stranded: bool = False,
			features: Optional[pd.DataFrame] = None,
		):
		'''
		scRNA-seq and scATAC-seq standard preprocessing pipeline, including:
		qc metrics, filtering, batch correction, PCA, neighboring graph, gene activity computation and wnn computation.
		Parameters:
			- n_pcs_rna: int; number of PCs to retain in scRNA-seq data. Default is 30.
			- n_pcs_act: int; number of PCs to retain in activity data. Deafult is 10. 
			- knn_rna: int; number of nearest neighbors for scRNA-seq KNN graph computation. Default is 30.
			- knn_act: int; number of nearest neighbors for activity KNN graph computation. Default is 30.
			- use_rep: optional, str; data representation to be used in each modality for neighbor computation. check sc.pp.neighbors for further information. 
			- wnn: int; number of nearest neighbors per modality to consider in wnn computation. Default is 30. 
			- stranded: bool; whether to consider strand in computing gene activity. Default is False. 
			- features: pd.Dataframe, optional; dataframe containing "chromosome", "start", "end", "strand" for genes considered in activity computation. 
		Returns:
			Updated the muon object with modality "activity". 
	
		'''
		if not self.use_activity:	
			# compute activity and normalize
			if self.atac_key is not None and "fragments" not in self.mudata.mod[self.atac_key].uns["files"]:
					raise KeyError("Fragment file not available for activity from atac computation")
			if features is None:
					raise ValueError("Feature Dataset is required for gene activity computation")
			features["start"] = features["start"].astype(int) - 1
			features["end"] = features["end"].astype(int)
			if (features["start"]<0).any():
				raise ValueError("Feature start must be >=0 after 0-basedconversion")

			self.activity_key = "activity"
			self.mudata.mod["activity"] = mu.atac.tl.count_fragments_features(data=self.mudata.mod[self.atac_key], 
											features = features, 
											stranded=stranded)	
			sc.pp.normalize_total(self.mudata.mod[self.activity_key])
			sc.pp.pca(self.mudata.mod[self.activity_key], random_state= self.random_state)
	
		# mu.pp.neighbors works on all possible modalities, but we only require rna and activity -> must create a mudata object with only gex and activity
		self.mudata = MuData({ self.rna_key: self.mudata[self.rna_key],
								self.activity_key: self.mudata[self.activity_key] })
	
		if not "distances" in self.mudata[self.rna_key].obsp: 
			sc.pp.neighbors(self.mudata.mod[self.rna_key], n_neighbors=knn_rna, n_pcs=n_pcs_rna, random_state=self.random_state, use_rep=use_rep)
		if not "distances" in self.mudata[self.activity_key].obsp: 
			sc.pp.neighbors(self.mudata.mod[self.activity_key], n_neighbors=knn_act, n_pcs=n_pcs_act, random_state=self.random_state, use_rep=use_rep)
		
		mu.pp.neighbors(self.mudata, 
				key_added="wnn", 
				n_neighbors=n_neighbors, 
				n_bandwidth_neighbors=n_bandwidth_neighbors, 
				n_multineighbors=n_multineighbors, 
				random_state=self.random_state, 
				metric=metric)
		mu.tl.umap(self.mudata, random_state=self.random_state, neighbors_key="wnn")
			
	def get_data(self) -> MuData:
		'''
		Returns the MuData object
		'''		
		return self.mudata

	@staticmethod
	def _minmax(x:np.ndarray) -> np.ndarray:
		if np.max(x) == np.min(x):
			return np.zeros_like(x) 
		return (x-np.min(x))/(np.max(x) - np.min(x))
	

	def compute_entropy(self, fate_prob_key:str="fate_probabilities"):
		probs = self.mudata.obsm.get(fate_prob_key, None)
		if probs is None:
			raise ValueError("Compute fate probabilities before running entropy")

		if not isinstance(probs, pd.DataFrame):
			raise ValueError("Fate probabilities not a DataFrame")

		if (probs.shape[1] == 0 
			or np.any(probs.sum(axis=1) == 0) 
			or np.any(probs.sum(axis=0) == 0)):
			warnings.warn("No terminal states or cells with no developmental probability or state without assignment")
			self.mudata.obs["shannon_entropy"] = np.nan
			self.mudata.obs["kl_divergence"] = np.nan
			return

		shannon_entropy = scipy.stats.entropy(probs, axis=1)
		average_distribution = np.mean(probs, axis=0)
		kl_divergence = np.nan_to_num(scipy.stats.entropy(probs, average_distribution, axis=1, base=2),
								nan=1.0,	
								copy=False)
		shannon_entropy, kl_divergence = self._minmax(shannon_entropy), self._minmax(kl_divergence)
		self.mudata.obs["shannon_entropy"] = pd.Series(shannon_entropy, index=self.mudata.obs.index)
		self.mudata.obs["kl_divergence"] = pd.Series(kl_divergence, index=self.mudata.obs.index)
				

	@abstractmethod
	def run(self, **kwargs: Any) -> None:
		pass	 

	
	def plot_embedding(self,
						embedding_key: str = "X_umap",
						observation: str= "pseudotime",
						save: Optional[Union[bool, str]] = None,
						cmap: str = "Blues",
						**kwargs):

		if observation not in self.mudata.obs.columns:
			warnings.warn(f"WARNING: {observation} not a valid cell metadata")
			return 

		if embedding_key not in self.mudata.obsm:
			raise KeyError(f"{embedding_key} not in mudata.obsm")

		_tmp = AnnData(X = np.zeros((self.mudata.n_obs, 1)),
						obs = self.mudata.obs.copy())
		_tmp.obsm[embedding_key] = self.mudata.obsm[embedding_key]
		title = observation
		kwargs.setdefault("save", save)

		scv.pl.scatter(_tmp, 
						title = title,
						color = observation,
						color_map = cmap,
						**kwargs)


	def plot_fate_probabilities(self,
								embedding_key: str = "X_umap",
								states: Optional[Union[str, Sequence[str]]] = None,
								cmap: str = "viridis",
								title: str =  "Fate Probabilities",
								save: Optional[Union[bool, str]] = None,
								**kwargs):

		if not hasattr(self, "fate_probability_key") or self.fate_probability_key is None: 
			warnings.warn("WARNING: fate probabilities are not available; Try recompute them")
			return 

		if embedding_key not in self.mudata.obsm:
			raise KeyError(f"{embedding_key} not in mudata.obsm")

		fate_probabilities = self.mudata.obsm[self.fate_probability_key]	
		_terminal_states = list(fate_probabilities.columns)
		_all_colors = self.mudata.uns["fate_state_colors"]

		if states is not None and isinstance(states, str):
			states = [states]
		states = [s for s in states if s in _terminal_states] if states is not None else _terminal_states
		if not len(states):
			raise ValueError(f"No lineages have been selected.")
	
		# compatibility with scvelo plot
		_terminal_colors = [_all_colors[state] for state in _terminal_states]
		_data = Lineage(fate_probabilities.to_numpy(),
						names = _terminal_states,
						colors = _terminal_colors)
		
		_singleton = _data.shape[1]==1
		_data = _data[states].copy()
		_X = _data.X
		
		if _X.shape[1] == 1 and np.allclose(_X, 1.0):
			_X = np.ones_like(_X)

		for col in _X.T:
			mask = ~np.isclose(col, 1.0)
			if np.any(mask):
				col[~mask] = np.nanmax(col[mask])
		
		kwargs.setdefault("save", save)
		kwargs.setdefault("legend_loc", "on data")	
		kwargs["color_gradients"] =  _data

		if _singleton and not np.allclose(_X, 1.0):
			kwargs.setdefault("perc", [0,95])
			_ = kwargs.pop("color_gradients", None)

		_tmp = AnnData(X = np.zeros((self.mudata.n_obs, 1)),
						obs = self.mudata.obs.copy())
		_tmp.obsm[embedding_key] = self.mudata.obsm[embedding_key]

		scv.pl.scatter(_tmp, 
						title = title,
						color_map = cmap,
						**kwargs)


	def plot_trends(self, 
					ptf: str,
					gene: str,
					branches: Optional[Union[list, str]] = None,
					deriv_threshold: float = 0.1,
					sharex: bool = True,
					n_splines: int = 8,
					n_points: int = 200,
					order: int = 1,
					save: Optional[str] = None):

		if not hasattr(self, "trends"):
			mbgam = MultiBranchGAM(
							mudata = self.mudata,	
							ptf = ptf, 
							gene = gene, 
							pseudotime_key = self.pseudotime_key,
							fate_prob_key = self.fate_probability_key,
							rna_modality = self.rna_key, 
							activity_modality = self.activity_key,	
							n_splines= n_splines)
			mbgam.fit()	
			mbgam.predict(n_points = n_points)
			mbgam.derivative(n_points= n_points, order=order)
			self.trends = mbgam

		if branches is None:
			branches = list(self.trends.models.keys())
		elif isinstance(branches, str):
			branches = [branches]

		colors = self.mudata.uns.get("fate_state_colors", {})
		default_color = "grey"

		fig, axes = plt.subplots(2, 1, 
							figsize = (7, 9),
							sharex = False)

		for branch in branches:
			pred = self.trends.predictions[branch]
			t = pred["t_grid"]
			c = colors.get(branch, default_color)

			axes[0].plot(t, pred["gex"], color=c, lw=2, label=branch)
			axes[1].plot(t, pred["act"], color=c, lw=2, label=branch)

		axes[0].set_ylabel(f"{ptf} expression (z-score)")
		axes[1].set_ylabel(f"{gene} activity (z-score)")
		axes[1].set_xlabel(self.pseudotime_key)
		axes[0].set_title(f"Dynamics: {ptf} -> {gene}")
		
		handles, labels = axes[0].get_legend_handles_labels()
		fig.legend(handles, labels, loc="center right", frameon=False)

		plt.tight_layout(rect=[0,0,0.85,1])

		if save:
			figure_path = os.path.join(os.getcwd(), "figures")
			if not os.path.exists(figure_path):
				os.mkdir(figure_path)
			path = os.path.join(figure_path, f"trends_{save}.png")
			plt.savefig(path)


	def plot_tree(self,
					embedding_key: str = "umap",
					nodes: int = 300,
					method: Literal["ppt", "epg"] = "ppt",
					ppt_lambda: int = 100,
					auto_root: bool = False,
					root_params: dict = {},
					reassign_pseudotime: bool = False, 
					crowdedness: float = 1, 
					color: Optional[str] = None,
					color_milestones: bool = False,
					n_jobs: int = -1,
					n_map: int = 1,
					save: Optional[str] = None,
					**kwargs):
		fate_prob_key, lineage_key = "term_states_fwd_memberships", "lineages_fwd"

		if not hasattr(self, "fate_probability_key") or self.fate_probability_key is None: 
			raise KeyError("Fate probabilities are not available; Try recompute them")
		if "kl_divergence" not in self.mudata.obs:	
			raise KeyError("entropy as KL-divergence required")
		if f"X_{embedding_key}" not in self.mudata.obsm:
			raise KeyError(f"X_{embedding_key} not in mudata.obsm")
		if color is not None and color not in self.mudata.obs.columns:
				raise KeyError(f"{color} not in mudata.obs")
		
		fate_probabilities = self.mudata.obsm[self.fate_probability_key].loc[self.mudata.obs_names]
		# scFates.cellrank_to_tree does not check n_fates = 1 and cellrank.pl.circular_projection does not work.
		if fate_probabilities is None and fate_probabilities.shape[1] == 0:
			raise KeyError("Fate probabilities are not available; Try recompute them")
		
		if fate_probabilities.shape[1] < 2:
			warnings.warn("WARNING: only one fate has been found, principal tree not available")
			return 

		_terminal_states = list(fate_probabilities.columns)
		_all_colors = self.mudata.uns["fate_state_colors"]
		_terminal_colors = [_all_colors[s] for s in _terminal_states]
		_data = Lineage(fate_probabilities.to_numpy(),
						names = _terminal_states,
						colors = _terminal_colors)


		_tmp = AnnData(X = np.zeros((self.mudata.n_obs, 1)),
						obs = self.mudata.obs.copy())
		_tmp.obsm[fate_prob_key] = self.mudata.obsm[self.fate_probability_key].values 
		_tmp.obsm[lineage_key] = _data
		_tmp.obsm[f"X_{embedding_key}"] = self.mudata.obsm[f"X_{embedding_key}"]

		scf.tl.cellrank_to_tree(adata = _tmp,
						time = self.pseudotime_key,
						Nodes = nodes,		
						method = method, 
						ppt_lambda = ppt_lambda,
						auto_root = auto_root,
						root_params = root_params,
						reassign_pseudotime = reassign_pseudotime,
						key_cellrank = fate_prob_key,
						copy = False,
						**kwargs)
	
		initial_cells = next(iter(self.mudata.uns["initial_states"].values())) 
		initial_idx = _tmp.obs_names.get_indexer(initial_cells)
		R_init = _tmp.obsm["X_R"][initial_idx, :]
		root = int(R_init.mean(axis=0).argmax())

		scf.tl.root(_tmp, root)
		scf.tl.pseudotime(_tmp,
							n_jobs= n_jobs,
							n_map = n_map,
							seed = self.random_state,
							copy=False)
		scf.tl.dendrogram(_tmp, crowdedness=crowdedness)
		plt.close()
		
		scf.pl.graph(_tmp, 
				basis = embedding_key, 
				save = save,
				**kwargs)

		scf.pl.dendrogram(_tmp,
						color_milestones = color_milestones,
						color = color,
						save = save,
						**kwargs)


class ATLAS:
	def __init__(self,
			mudata: MuData,
			method: Literal["palantir", "pseudotime-kernel"],
			fragment_path: Optional[str]=None, 
			random_state:int=42,
			**kwargs:Any):

		if method not in RUN_REGISTRY:
			raise ValueError(f"Unknown method '{method}'. Available: {list(RUN_REGISTRY)}")
		self._method = method
		self._impl = RUN_REGISTRY[method](mudata=mudata,
						fragment_path = fragment_path,
						random_state = random_state, 
						**kwargs)

	def preprocessing(self, **kwargs):
		return self._impl.preprocessing(**kwargs)

	def get_data(self):
		return self._impl.get_data()
	
	def run(self, **kwargs):
		return self._impl.run(**kwargs)
	
	def plot_embedding(self, **kwargs):
		return self._impl.plot_embedding(**kwargs)

	def plot_fate_probabilities(self, **kwargs):
		return self._impl.plot_fate_probabilities(**kwargs)

	def plot_trends(self, **kwargs):
		return self._impl.plot_trends(**kwargs)

	def plot_tree(self, **kwargs):
		return self._impl.plot_tree(**kwargs)

	@property
	def random_state(self):
		return self._impl.random_state


class PalantirWrapper(Base):
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
		if distance_key not in self.mudata.obsp.keys(): 
			raise KeyError(f"{distance_key} not in data.obsp")
		if knn is None:
			print(f"WARNING - knn parameter not specified. Looking for data.uns[{knn_key}][""params""][""n_neighbors""]")
			if knn_key not in self.mudata.uns.keys():
				raise KeyError(f"{knn_key} not in data.uns")	
			else:
				knn = int(self.mudata.uns[knn_key]["params"]["n_neighbors"])

		N = self.mudata.shape[0]
		kNN = self.mudata.obsp[distance_key]
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

		self.mudata.obsp[kernel_key] = kernel


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
		if kernel_key not in self.mudata.obsp.keys():
			raise KeyError(f"{kernel_key} not in data.obsp")

		kernel= self.mudata.obsp[kernel_key]
		res = palantir.utils.diffusion_maps_from_kernel(self.mudata.obsp[kernel_key], n_components, seed)
		self.mudata.obsp[sim_key] = res["T"] 
		self.mudata.obsm[eigvec_key] = res["EigenVectors"].set_index(self.mudata.obs.index)
		self.mudata.uns[eigval_key] = res["EigenValues"].values


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
		if eigval_key not in self.mudata.uns.keys():
			raise KeyError(f"{eigval_key} not in data.uns")
		if eigvec_key not in self.mudata.obsm.keys():
			raise KeyError(f"{eigvec_key} not in data.obsm")
		
		eigenvectors = pd.DataFrame(self.mudata.obsm[eigvec_key], index=self.mudata.obs_names) if not isinstance(self.mudata.obsm[eigvec_key], pd.DataFrame) else self.mudata.obsm[eigvec_key] 
		dm_dict = {"EigenValues": self.mudata.uns[eigval_key], "EigenVectors": eigenvectors}
		result = palantir.utils.determine_multiscale_space(dm_res = dm_dict, n_eigs=n_eigs, eigval_key = eigval_key, eigvec_key = eigvec_key, out_key=out_key) # eigval_key, eigvec_key and out_key are not used 
		self.mudata.obsm[out_key] = result

	
	def run(self, *,
			early_cell: str,
			cluster_key: Optional[str] = None,
			terminal_states: Optional[Union[List, Dict, pd.Series]] = None,
			knn: int=30,
			kernel_knn: Optional[int] = None,
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
			pseudotime_key: str = "pseudotime",
			fate_prob_key: str = "fate_probabilities",
			waypoints_key: str = "palantir_waypoints",
			**kwargs: Any):
	 
		self.compute_kernel(knn_key = knn_key, 
				distance_key = distance_key, 
				knn = kernel_knn,
				alpha = alpha,
				kernel_key = kernel_key)

		self.compute_diffusion_map(kernel_key=kernel_key,
					sim_key= sim_key,
					eigval_key= eigval_key,
					eigvec_key= eigvec_key,
					n_components = n_components,
					seed = self.random_state)

		self.compute_multiscale_space(n_eigs = n_eigs,
					eigval_key = eigval_key,
					eigvec_key = eigvec_key,
					out_key = eigvec_multi_key)

		res = palantir.core.run_palantir(data = self.mudata.obsm[eigvec_multi_key],
						early_cell = early_cell,
						terminal_states = terminal_states,
						knn = knn,
						num_waypoints = num_waypoints,
						n_jobs = n_jobs, 
						scale_components = scale_components,
						use_early_cell_as_start = use_early_cell_as_start,
						max_iterations = max_iterations, 
						eigvec_key = eigvec_key,
						pseudo_time_key = pseudotime_key,
						entropy_key = "palantir_entropy",
						fate_prob_key = fate_prob_key, 
						save_as_df = True,
						waypoints_key = waypoints_key, 
						seed= self.random_state)

		self.mudata.obs[pseudotime_key] = res.pseudotime
		self.mudata.uns[waypoints_key] = res.waypoints.values
		if isinstance(terminal_states, pd.Series):
			res.branch_probs.columns = terminal_states[res.branch_probs.columns]
	
		if cluster_key is not None and cluster_key in self.mudata.obs.columns: 
			cell_to_cluster = self.mudata.obs[cluster_key] 
			terminal_states = {}
			for cell in res.branch_probs.columns: 
				cluster= cell_to_cluster.loc[cell]
				terminal_states.setdefault(cluster, []).append(cell) 
				
			initial_states = {cell_to_cluster.loc[early_cell] : [early_cell]}
			res.branch_probs.columns = cell_to_cluster.loc[res.branch_probs.columns].values
			fate_probs = res.branch_probs.groupby(level=0, axis=1).sum()
				
		else: 
			terminal_states = {cell: [cell] for cell in res.branch_probs}
			initial_states = {early_cell : [early_cell]}
			fate_probs = res.branch_probs

		self.mudata.uns["initial_states"] = initial_states
		self.mudata.uns["terminal_states"] = terminal_states	
		self.mudata.obsm[fate_prob_key] = fate_probs
		self.fate_probability_key = fate_prob_key
		self.pseudotime_key = pseudotime_key
		_assign_state_colors(self.mudata)

		self.compute_entropy(fate_prob_key=fate_prob_key)	
	

class PseudotimeKernelWrapper(Base):
	def __init__(self, 
			mudata:MuData, 
			pseudotime_key:str="pseudotime",
			cluster_key: Optional[str] = None,
			**kwargs):
		'''
		Parameters:
		- mudata: MuData; muon object containing modalities for trajetory inference.
		- pseudotime_key: str; Key in mudata.obs where pseudotime for every cell is computed. Default is "pseudotime". 
		- connectivity_key: str; Key in mudata.obsm where knn connectivites are stored. Default is "wnn_connectivities".
		- backward: bool; Indicating whether forwards or backward direction needs to be identified. Defaults is False.
		'''
		super().__init__(mudata=mudata, **kwargs)

		if pseudotime_key not in self.mudata.obs.columns:
			raise KeyError(f"{pseudotime_key} not in obs")	
		if cluster_key is not None and cluster_key not in self.mudata.obs.columns:
			warnings.warn(f"{cluster_key} not in self.mudata.obs. Setting to None")
			cluster_key = None

		self.cluster_key = cluster_key
		self.pseudotime_key = pseudotime_key


	def run(self, *, 
			connectivity_key: str = "wnn_connectivities", 
			backward: bool = False, 			
			threshold_scheme: Literal["soft", "hard"] = "hard", 
			frac_to_keep: float = 0.3, 
			b: float = 10.0, nu: float = 0.5, 
			n_states: Optional[Union[int, Sequence[int]]] = None,
			n_cells: int= 30, 
			allow_overlap: bool = False,
			states_method: Literal["stability", "top_n", "eigengap", "eigengap_coarse"] = "stability",
			alpha: float = 1, 
			stability_threshold: float = 0.96, 
			terminal_states: Optional[dict[str, Sequence[str]]] = None,
			initial_states: Optional[dict[str, Sequence[str]]] = None,
			solver: Literal["direct", "gmres", "lgmres", "bicgstab", "gcrotmk"] = "gmres", 
			use_petsc: bool = True, 
			n_jobs: int = -1,
 			tol: float = 1e-6,
			n_terminal_states: Optional[int] = None, 
			n_initial_states: int = 1,
			preconditioner: Optional[str] = None,	
			**kwargs):
		'''
		Computes transition matrix and then runs GPCCA. 
		Parameters:
			- threshold_scheme: str; either "soft" or "hard", defines which method for biasing the graph to use. The "soft" scheme is based on *VIA* and it down weights edges that points against the direction of increasing pseudotime. The "hard" scheme is based on *Palantir* and it removes edges that points against the direction of increasing pseudotime, keeping the ones within a radius in order not to disconnect the graph. 
			- frac_to_keep: float; fraction of nearest neighbors according to local connectivity that is kept independently from pseudotime (radius in threshold_scheme=="hard" see parameter "threshold_scheme"). Default is 0.3.
			- b, nu: float; respectively the ---- and --- used when threshold_scheme is "soft". Default are 10.0 and 0.5. 
		'''
		def _invert_assignment(assignment):
			if not isinstance(assignment.dtype, pd.CategoricalDtype):
				assignment = assignment.astype("category")

			inverted_assignment = { state: assignment.index[assignment == state].tolist()
									for state in assignment.cat.categories}
			return inverted_assignment

		if connectivity_key not in self.mudata.obsp.keys():
			raise KeyError(f"{connectivity_key} not in obsp")	
		self.connectivity_key = connectivity_key

		self._adata = AnnData(X = csr_matrix((self.mudata.n_obs, self.mudata["rna"].n_vars)),
												obs = self.mudata.obs, 
												var = pd.DataFrame([], index = self.mudata["rna"].var_names))
		self._adata.obs[self.cluster_key] = self._adata.obs[self.cluster_key].astype("category")

		self._adata.obsp[connectivity_key] = self.mudata.obsp[connectivity_key]
		self.kernel = PseudotimeKernel(adata= self._adata, 
									time_key = self.pseudotime_key, 
									conn_key = connectivity_key, 
									backward = backward).compute_transition_matrix(
														threshold_scheme = threshold_scheme, 
														frac_to_keep = frac_to_keep, 
														b=b, nu = nu, n_jobs = n_jobs) 

		self._G = cr.estimators.GPCCA(self.kernel)										
		self._G.compute_schur()
		if terminal_states is not None and initial_states is not None:
			self._G.set_initial_states(states = initial_states) 
			self._G.set_terminal_states(states = terminal_states)
		else:
			self._G.compute_macrostates(n_states = n_states, cluster_key = self.cluster_key)
			self._G.predict_terminal_states(method = states_method, 
											n_cells = n_cells, 
											alpha = alpha, 
											stability_threshold = stability_threshold,
											n_states = n_terminal_states, 
											allow_overlap = allow_overlap)
			self._G.predict_initial_states(n_states = n_initial_states,
											n_cells = n_cells, 
											allow_overlap = allow_overlap)

		self._G.compute_fate_probabilities(keys = None,
											solver = solver,
											use_petsc = use_petsc, 
											n_jobs = n_jobs,
											tol = tol,
											preconditioner = preconditioner)
	
		self.mudata.obsm["fate_probabilities"] = pd.DataFrame(self._G.fate_probabilities.X, 
																index = self.mudata.obs_names,
																columns = self._G.fate_probabilities.names)		
		self.mudata.uns["initial_states"] = _invert_assignment(self._G.initial_states)
		self.mudata.uns["terminal_states"] = _invert_assignment(self._G.terminal_states)
		if terminal_states is None and initial_states is None:
			intermediate = self._G.macrostates[
											(self._G.initial_states.isna()) & 
											(self._G.terminal_states.isna())
											]
			intermediate = intermediate.cat.remove_unused_categories()
			self.mudata.uns["intermediate_states"] = _invert_assignment(intermediate)
		else:
			self.mudata.uns["intermediate_states"] = {}
			
		self.fate_probability_key = "fate_probabilities"
		_assign_state_colors(self.mudata)

		self.compute_entropy(fate_prob_key="fate_probabilities")	


RUN_REGISTRY = {"palantir": PalantirWrapper,
		"pseudotime-kernel": PseudotimeKernelWrapper}

