import os 
import muon as mu
import scanpy as sc
from muon import MuData
from anndata import AnnData
from scipy.sparse import csr_matrix
from typing import Union, Optional, Literal
from cellrank.kernels import PseudotimeKernel 

class PseudotimeKernelMuon():
	def __init__(self, data: MuData, modality_key:Optional[str] = None, embedding_key : str = "X_umap", connectivity_key: str = "connectivities", pseudotime_key: str="pseudotime", group_key: Optional[Union[str, list]]=None):

		if modality_key is not None:
			if modality_key not in data.mod.keys():
				raise KeyError(f"{modality_key} not available")
			print(f"Considering {modality_key} modality")

		if embedding_key not in data.obsm.keys():
			raise KeyError(f"{embedding_key} not in data.obsm")
	
		adata = data if modality_key is None else data[modality_key]	
		if connectivity_key not in adata.obsp.keys():
			raise KeyError(f"{connectivity_key} not in obsp.keys()")
		if pseudotime_key not in adata.obs.columns:
			raise KeyError(f"{pseudotime_key} not in obs")
		
		if group_key is not None and isinstance(group_key, str):
			group_key = [group_key]
		if group_key is not None:
			for k in group_key:
				if k not in adata.obs.columns:
					raise KeyError(f"{key} not in adata.obs")

		obs = adata.obs[[pseudotime_key] + group_key]
		var = adata.var
		self.data = AnnData(X=csr_matrix((adata.obs.shape[0], adata.var.shape[0])), obs = obs, var = var)
		self.data.obsp[connectivity_key] = adata.obsp[connectivity_key].copy()
		self.data.obsm[embedding_key] = data.obsm[embedding_key].copy()
		self.connectivity_key = connectivity_key
		self.group_key = group_key
		self.pseudotime_key = pseudotime_key
		self.embedding_key = embedding_key


	def compute_transition_matrix(self, threshold_scheme: Literal["soft", "hard"], frac_to_keep: float = 0.3, b:float=10, nu:float = 0.5, n_jobs:int = -1):
		self.kernel = PseudotimeKernel(adata=self.data, time_key = self.pseudotime_key, conn_key=self.connectivity_key)
		self.kernel.compute_transition_matrix(threshold_scheme = threshold_scheme, frac_to_keep= frac_to_keep, b=b, nu=nu, n_jobs = n_jobs)
