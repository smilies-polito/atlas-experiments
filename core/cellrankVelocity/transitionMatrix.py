import numpy as np
import scipy as sc
import pandas as pd
from muon import MuData
from anndata import AnnData
from .model import Deterministic, Similarity, SimilarityWrapper, SimilarityComputer, Correlation, Cosine, DotProduct
from scipy.sparse import csr_matrix, issparse, hstack
from typing import Literal, Union, Optional


def _expand_matrix(data:MuData, modality_key1:str, modality_key2:str):
	if modality_key1 not in data.mod.keys() or modality_key2 not in data.mod.keys():
		raise KeyError("Specified modalities not valid")
	return csr_matrix(hstack((data[modality_key1].X, data[modality_key2].X)))

class TransitionMatrixABC:
	"""
	Class that computes transition matrix for the MCMC.
	1. Computes displacement vector between one cell and its neighbors according to the neighborhood graph.
	2. Computes correlation between cell_i velocity and displacement vectors with neighbors. 
	4. Softmax to pass from correlations to probabilities
	Attributes
	-----------
	_data: mudata.MuData
	_X: numpy.ndarray or scipy.sparse.csr_matrix
		peak-barcode matrix over which to compute transition probability matrix.
	_velocities: numpy.ndarray
		velocity vector wose shape should match X.shape 
	_softmax_scale: float, optional
		softmax_scale value for softmax computation (default is None)  
	"""
	def __init__(self, data: Union[MuData, AnnData], velocities: np.ndarray, X: Union[np.ndarray, csr_matrix], softmax_scale: Optional[float]= None):
		"""
		Parameters
		-----------
		data: mudata.MuData
		X: numpy.ndarray or scipy.sparse.csr_matrix
			peak-barcode matrix over which to compute transition probability matrix.
		velocities: numpy.ndarray
			velocity vector wose shape should match X.shape 
		softmax_scale: float, optional
		softmax_scale value for softmax computation (default is None).
		"""
		if not velocities.shape==X.shape:
			raise ValueError("Velocities shape does not match with X shape")

		self._velocities = velocities
		self._data = data
		self._X = X
		self._softmax_scale = softmax_scale

	def estimate_softmax_scale(self, 
								similarity: Union[DotProduct, Cosine, Correlation, SimilarityComputer, Similarity], 
								key: str = 'wnn_connectivities'):
		"""
		Function that estimates the softmax scale. softmax scale is sigma = 1/median{|c_ik|} where c_ik are the logits obtained through the model. 
		Parameters
		----------
		similarity: SimilarityComputer or Similarity or Cosine, Correlation, DotProduct.
			object for the transition matrix computations.
		key: str
			key in _data.obsp where neighborhood matrix is stored. Default is "wnn_connectivities".
		Returns
		--------
		float: softmax scale
		"""
		model = Deterministic(data=self._data, X=self._X, velocities=self._velocities, similarity=similarity, 
                              softmax_scale=1.0, key=key)
		_ , logits = model()
		return 1.0/np.median(np.abs(logits.data))


	def compute_transition_matrix(self, 
                                  similarity: Union[Literal['correlation', 'cosine', 'dot'], Cosine, Correlation, DotProduct, SimilarityComputer, Similarity] = 'correlation', 
                                  key: str = 'wnn_connectivities'):
		"""
		Function that computes transition probabilities and logits.
		Parameters
		----------
		similarity: SimilarityComputer or Similarity or Cosine, Correlation, DotProduct or string among "correlation", "cosine", "dot".
			object for the transition matrix computations. Default is "correlation".
		key: str
			key in _data.obsp where neighborhood matrix is stored. Default is "wnn_connectivities".
		Updates
		--------
		_transition_matrix: scipy.sparse.csr_matrix
		_logits: scipy.sparse.csr_matrix
		"""
		if key not in self._data.obsp:
			raise KeyError(f"{key} not in data.obsp")

		if isinstance(similarity, str):
			similarity = SimilarityWrapper.create(similarity)

		if self._softmax_scale is None:
			self._softmax_scale = self.estimate_softmax_scale(similarity, key)

		model = Deterministic(data=self._data, X=self._X, velocities=self._velocities, similarity=similarity, 
                              softmax_scale=self._softmax_scale, key=key)
		self._transition_matrix, self._logits = model()


	@property
	def transition_matrix(self):
		return self._transition_matrix

            

class TransitionMatrix(TransitionMatrixABC): 
	"""
	Class that computes the transition matrix for multi-omics data.
	Attributes
	-----------
	_data : mudata.MuData
		MuData 
	_X : scipy.sparse.csr_matrix
		displacement vector
	_velocities: np.ndarray
		velocity vector
	"""
	
	def __init__(self, data: MuData, modality_key: str=None, rna_key: str="rna", atac_key: str="activity", velocity_key: str= "velocity", softmax_scale: Optional[float]=None):
	
		if modality_key is not None and modality_key not in data.mod.keys():
			raise KeyError(f"{modality_key} not in data.mod.keys()")	
		if rna_key not in data.mod.keys():
			raise KeyError(f"{rna_key} not in data.mod.keys()")        
		if atac_key not in data.mod.keys():
			raise KeyError(f"{atac_key} not in data.mod.keys()")
		if velocity_key not in data[rna_key].layers:
			raise KeyError(f"{velocity_key} not found recompute velocity")

		if modality_key is None:
			X = _expand_matrix(data, modality_key1 = rna_key, modality_key2 = atac_key)
			velocities = np.hstack((data[rna_key].layers[velocity_key], data[rna_key].layers[velocity_key]))
		else:
			X = data[modality_key].X
			velocities = data[modality_key].layers[velocity_key]
			data = data[modality_key]
		
		super().__init__(data=data, X=X, velocities=velocities, softmax_scale=softmax_scale)
