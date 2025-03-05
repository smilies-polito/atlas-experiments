import numpy as np
import scipy as sc
import pandas as pd

from anndata import AnnData
from muon import MuData
from model import Deterministic, Similarity, SimilarityWrapper, SimilarityComputer, Correlation, Cosine, DotProduct
from scipy.sparse import csr_matrix, issparse, hstack
from typing import Literal, Union, Optional

def _expand_matrix(data:AnnData, annotations: pd.DataFrame, peak_key:str, gene_key:str):
	""" 
	Function that expands the atac matrix according to their promoter role with respect to the genes in the annotation file.

	Params:
	-------
	data: anndata.AnnData
		AnnData with ATAC values
	annotations: pandas.DataFrame
		pandas.DataFrame containing peak-genes association for promoter peaks in the AnnData. 
	peak_key: str 
		string in annotations.columns where promoter peaks are stored.
	gene_key: str
		string in annotations.columns where genes are stored.

	Returns:
	--------
	scipy.sparse.csr_matrix of shape n_cells x (n_promoter_peaks * n_genes_each_promoter) containing peak_i values repeated n_genes_each_promoter,
	i.e., the number of genes peak_i is promoter to. 
 
	"""
	grouped = annotations[[peak_key, gene_key]].groupby(by=[peak_key]).size()
	peaks, howMuch  = grouped.index, grouped.values
	indices = [np.where(data.var_names==peak)[0][0] for peak in peaks]
	indices = np.repeat(indices, howMuch)
	return data.X[:, indices]


def _map_velocities(data: AnnData, annotations:pd.DataFrame, velocity_key:str, peak_key:str, gene_key:str):
	"""
	Function that retrieves gene velocity for each gene associets to a promoter peak. 

	Params:
	------
	data: anndata.AnnData
	annotations: pandas.DataFrame 
		pandas.DataFrame containinf peak-genes association for promoter peaks. 
	velocity_key: str
		string in data.layers indicating where velocities are stored.
	peak_key: str
		string in annotations.columns where promoter peaks are stored.
	gene_key: str
		string in annotations.columns where genes are stored.
	
	Returns:
	--------
	np.ndarray of shape n_cells * (n_promoter_peaks * n_genes_each_promoter) containing gene velocities for every gene a peak is promoter to.

	"""
	velocities = pd.DataFrame(data.layers[velocity_key], index=data.obs_names, columns = data.var_names)
	annotations = annotations.sort_values(by=peak_key)
	return velocities[annotations[gene_key].values].values


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
	def __init__(self, data: MuData, velocities: np.ndarray, X: Union[np.ndarray, csr_matrix], softmax_scale: Optional[float]= None):
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
	
	def __init__(self, data: MuData, annotations: pd.DataFrame, velo_modality: str="rna", velocity_key: str= "velocity",
                 gene_key:str = 'gene', peak_key : str='peak', softmax_scale: Optional[float]=None):
		"""
		Params
		------
		data: mudata.MuData
		velo_modality: str   
			string in data.mod.keys() indicating the modality in which velocities are stored. Default is "rna". 
		annotations: pandas.DataFrame 
			annotations containing promoter peaks and related genes. 
		peak_key: str 
			key in annotations.columns reative to peaks names. Default is "peak".
		gene_key: str
			key annotations.columns relative to gene names. Default is "gene".
		velocity_key: str 
			key in data[velo_modality].layers where velocities are stored. Default is "velocity". 
		softmax_scale: float, optional
			softmax_scale value for softmax computation. Default is None. 	
		Note
		----	
		data should contain "atac" and "rna" modalities. Not checked right now. 
		"""
		if velo_modality not in data.mod.keys():
			raise KeyError(f"{velo_modality} not in data.mod.keys()")        

		if peak_key not in annotations.columns or gene_key not in annotations.columns:
			raise IndexError(f"{peak_key} or {gene_key} not in annotations.columnms")

		if velocity_key not in data[velo_modality].layers:
			raise IndexError(f"{velocity_key} not in data[{velo_modality}].layers") 

		X_atac = _expand_matrix(data=data["atac"], annotations=annotations, peak_key=peak_key, gene_key=gene_key)
		atac_velocities = _map_velocities(annotations=annotations, data=data[velo_modality], velocity_key= velocity_key, peak_key=peak_key, gene_key=gene_key)
		X = csr_matrix(hstack([X_atac, data["rna"].X]))
		velocities = np.hstack((atac_velocities, data[velo_modality].layers[velocity_key]))

		super().__init__(data=data, X=X, velocities=velocities, softmax_scale=softmax_scale)
