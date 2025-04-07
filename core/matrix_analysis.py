import os
import numpy as np
import scipy as sc
import scanpy as sp
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt 

from csv import DictWriter
from anndata import AnnData
from muon import MuData
from typing import Optional, Union

class MatrixAnalyser:
	"""
    Class implementing analysis for the graph inuced by the transition matrix. 

    Attributes
    ----------
    _matrix: numpy.ndarray/scipy.sparse.csr_matrix.
        transition matrix
    _G: networkx.Graph
        graph associated with transition matrix.
    _cluster_key: str
        string indicating the cluster in adata.obs.
    _adata: anndata.AnnData
        AnnData associated to the transition matrix.
    _k: int
        size of local neighborhood.
    _pc: int, optional
        number of principal components.
    _lsi: int,optional
        number of LSI dimensions.
    _seed: int
        seed for computations (default 52).
    _sink: list
        list containing the sink nodes on the condensation graph.
    _is_stochastic: bool
        boolean indicating whether the transition matrix is stochastic.
    _params:
        dictionary containing multiple parameters for graph analysis will be updated. 
	
	"""

	def __init__(self, matrix: Union[np.ndarray, sc.sparse.csr_matrix], adata:Union[ MuData, AnnData], cluster_key: Optional[str]= None, seed:int = 42, **kwargs):
		"""
        Parameters
        ----------
        matrix : numpy.ndarray/scipy.sparse.csr_matrix
            transition matrix
        adata : anndata.AnnData
            AnnData associated to the transition matrix     
        clustr_key: str
            str in adata.obs containing cluster labels. 
        **kwargs: 
            mandatory to specify "k" (int) size of the local neighbohood and at least one between "pc" (int) and "lsi" (int),
            the number of principal compoentns and LSI dimensions.
        

		"""
		self._matrix = matrix
		self._G = nx.DiGraph(matrix)

		if cluster_key is not None and cluster_key not in adata.obs.columns:
			print("Invalid cluster_key")
			cluster_key = None

		self._cluster_key = cluster_key

		for node in self._G.nodes:
			self._G.nodes[node]['barcode'] = adata.obs.index[node]
			self._G.nodes[node][cluster_key] = adata.obs[cluster_key].iloc[node] if cluster_key is not None else None

		self._adata = adata
		
		self._is_stochastic = np.allclose(matrix.sum(axis=1), np.ones(matrix.shape[0])) and not np.sum(np.any(matrix<0))
		self._params = {}
		self._seed =seed

		if not self._is_stochastic:
			raise(ValueError, 'The matrix is not stochastic')


	def _topology_analysis(self):
		""" 

        Function that analyses the graph topology requirements: aperiodicity and connectivity 

        Updates
        -------
        ._params: dict
            adds boolean for "aperiodic" and "strongly_connected"

		"""
		self._params['aperiodic'] = nx.is_aperiodic(self._G)
		self._params['strongly_connected'] = nx.is_strongly_connected(self._G)

		H = nx.condensation(self._G)
		self._H  = H
		self._params["number_of_components"] = len(H.nodes)
		self._params["number_of_sink"] = len([node for node in H.nodes if H.out_degree(node)==0])
		disconnected_nodes = [node for node in H.nodes if H.in_degree(node)==0]
		self._params["disconnected_nodes"] = len([node for node in H.nodes if H.in_degree(node)==0])


	def get_params(self):
		return self._params
    
    


