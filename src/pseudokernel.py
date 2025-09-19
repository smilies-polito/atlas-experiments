import os 
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
from muon import MuData
from anndata import AnnData
from scipy.stats import entropy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from typing import Union, Optional, Literal
from cellrank.kernels import PseudotimeKernel 
from scipy.spatial.distance import jensenshannon

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


		

def analysis_matrix(P:csr_matrix) -> dict:
	is_stochastic, is_aperiodic, is_connected, n_sinks = topology_analysis(P)
	stationary_distribution = compute_stationary_distribution(P, is_aperiodic, is_connected).tolist()
	eigenvalues = sorted(np.abs(eigs(P.T)[0]), reverse=True)
	gap = 1 - eigenvalues[1] if len(eigenvalues) > 1 else None
	mixing_time = np.log(2 / 0.01) / gap  if gap is not None and gap > 0 else None
	entropy = compute_entropy(P).tolist()
		
	return {"stochastic": is_stochastic, "connected": is_connected, "aperiodic": is_aperiodic, "sinks": n_sinks,
		"eigenGap": gap, "mixing_time":mixing_time, "stationary_distribution":stationary_distribution, 
		"row_entropy": entropy}
		


def compute_stationary_distribution(P: csr_matrix, is_aperiodic: bool=True, is_connected: bool=True) -> np.ndarray:
	if is_connected: 
		vals, vecs = eigs(P.T, k=5, which="LM")
		idx = np.argmin(np.abs(vals - 1))
		v = np.real(vecs[:, idx])
		v = np.maximum(v, 0)
		if v.sum() > 0:
			v = v / v.sum()
		return v
	else:
		G = nx.diGraph(P)
		H = nx.condensation(G)
		sink_nodes = [node for node in H.nodes if H.out_degree(node) == 0]
		sinks = []
		for sink in sink_nodes:
			states = H.nodes[sink]["members"]
			subP = P[states, :][:, states].toarray()
			vals_sub, vecs_sub = np.linalg.eig(P_sub.T)
			idx_sub = np.argmin(np.abs(vals_sub - 1))
			v_sub = np.real(vecs_sub[:, idx_sub])
			v_sub = np.maximum(v_sub, 0)
			if v_sub.sum() > 0:
				v_sub = v_sub / v_sub.sum()
			v_full = np.zeros(P.shape[0])
			v_full[states] = v_sub
			sinks.append(v_full)
			return np.array(sinks)

def compute_entropy(P:csr_matrix) -> np.ndarray:
	return entropy(P.toarray(), axis=1)	
		

def topology_analysis(P:csr_matrix) -> tuple:
	G = nx.DiGraph(P)
	row_sums = P.sum(axis=1)
	all_almost_one = np.all(np.isclose(np.squeeze(np.asarray(row_sums)), 1))
	is_stochastic = all_almost_one and not np.sum(np.any(P<0))
	is_aperiodic = nx.is_aperiodic(G)
	is_connected = nx.is_strongly_connected(G)
	H = nx.condensation(G)
	n_sinks = len([node for node in H.nodes if H.out_degree(node)==0])
	return (is_stochastic, is_aperiodic, is_connected, n_sinks)
	




