import pandas as pd
import numpy as np

from enum import Enum 
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse
from typing import Literal, Union, Optional


class Similarity(Enum):
	"""Enum that imitates CellRank similarity computations"""
	DOT_PRODUCT = "dot"
	COSINE = "cosine"
	CORRELATION = "correlation"

class SimilarityWrapper():
	"""
	Class that allows to create and manage different similarities computations.
	Returns a SimilarityComputer object.
	"""
	def create(similarity: Literal['correlation', 'cosine', 'dot']):
		return Correlation() if similarity=="correlation" else Cosine() if similarity == 'cosine' else DotProduct()
    

class SimilarityComputer():
	"""
	Class that computes the transition matrix given a certain similarity metric.
	Params:
	------
	center_mean: boolean
		Indicates wheter to center velocity and displacement vectors around their mean. Default False.
	scale_by_norm: boolean
		Indicated whether to scale velocity and displacement vectors by their norm. Default False.
	"""
	def __init__(self, center_mean: bool = False, scale_by_norm: bool = False):
		self._center_mean = center_mean
		self._scale_by_norm = scale_by_norm

	def __call__(self, v, X, softmax_scale: float = 1.0):
		"""
		Computes transition matrix: 
		Params:
		------- 
		v: velocity vector for cell_i
		X: displacement vector between cell_i and its neighbors
		sotftmax_scale: softmax scale for softmax computation. Default 1.0
		"""
		if self._center_mean:
			X -= np.expand_dims(np.mean(X, axis=1), axis=1)
			v -= np.mean(v)

		if self._scale_by_norm:
			denom = np.linalg.norm(v) * np.linalg.norm(X, axis = 1)
			mask = denom == 0
			denom[mask] = 1
			return self.softmax_masked(X.dot(v)/denom, mask, softmax_scale)

		return self.softmax(X.dot(v), softmax_scale)
    

	def softmax_masked(self, x: np.array, mask: np.array, softmax_scale: float = 1.0):
		"""
		Computes softmax and returns probabilities and logits.
		Params:
		------
		x: vectors of logits from which compute the probabilities
		mask: mask indicating which part of x result from a divsion by 0
		softmax_scale: softmax scale for softmax computation. Default is 1.0
		"""
		numerator = x*softmax_scale
		numerator = np.exp(numerator - np.nanmax(numerator))
		numerator = np.where(mask, 0, numerator)
		return numerator/np.nansum(numerator), x

	def softmax(self, x: np.array, softmax_scale: float = 1.0):
		"""
		Computes softmax and returns probabilities and logits.
		Params:
		-------
		x: vectors of logits from which compute the proebabilities
		softmax_scale: softmax scale for softmax computation. Default is 1.0
		"""
		numerator = x * softmax_scale
		numerator = np.exp(numerator - np.max(numerator))
		return numerator / np.sum(numerator), x


class Cosine(SimilarityComputer):
	def __init__(self):
		super().__init__(center_mean=False, scale_by_norm=True)

class Correlation(SimilarityComputer):
	def __init__(self):
		super().__init__(center_mean= True, scale_by_norm=True)

class DotProduct(SimilarityComputer):
	def __init__(self):
		super().__init__(center_mean = False, scale_by_norm = False)


class Deterministic():
    """Class that simulates the deterministic model of CellRank for transition matrix computations."""

    def __init__(self, 
                 adata: AnnData,
                 velocitites: np.ndarray,
                 similarity: Union[Cosine, Correlation, DotProduct, SimilarityComputer, Similarity], 
                 X: Optional[Union[np.ndarray, csr_matrix]]=None, 
                 softmax_scale: float = 1.0, 
                 key: str = 'connectivities'
    ):
        """
            params:
                - adata: annotated data. 
                - X: np.ndarray peak-barcode matrix n_cellsxn_peaks. 
                    Default is None, in this case then adata.X is used for computations.
                - velocities: np.ndarray mathicng the shape of adata or the shape of X.
                - similarity: similarity class that implements the transition matrix computation strategy. 
                - softmax_scale: float for the softmax computation. Default 1.0
                - key: key in adata.obsp where to search for neighbor connectivities. Deafult 'connectivities'    
        """
        self._X = X if X is not None else adata.X
        self._adata = adata


        if isinstance(similarity, Similarity):
            similarity = SimilarityWrapper.create(similarity)

        self._similarity= similarity
        self._key = key
        self._velocities = velocitites
        self._softmax_scale = softmax_scale
        self._probabilities = None
        self._logits = None
        self._indices = None
        self._indptr = None


    def uniform(self, n_neighbors: int):
        """Function that returns uniform distribution over the neighbors.
            prarams: 
                n_neighbors: number of neighbors for cell_i
        """
        return np.ones(n_neighbors)/n_neighbors, np.ones(n_neighbors)*1e-6
    

    def compute_displacement_vector(self, idx: int, key: str ='connectivities'):
        """
            Function that compute the displacement vector for the specific cell
            params:
                adata: annotated data
                idx: cell index integer
                key: where to look for the neighborhood
        """
        indptr, indices = self._adata.obsp[key].indptr, self._adata.obsp[key].indices
        start, end = indptr[idx], indptr[idx+1]
        neighbors_idx = indices[start:end]
        if issparse(self._X):
            displacement = self._X[neighbors_idx].A - self._X[idx].A
        else:
            displacement =  self._X[neighbors_idx]- self._X[idx]

        return neighbors_idx, displacement
    

    def update_transitions(self, probabilities: np.ndarray, logits: np.ndarray, neighbors: np.ndarray):
        """
        Function that updates the object's fields to construct csr_matrix. 
            params:
                probabilities: (n_neighbors,) array containing the cell_i probabilities of transitioning
                logits: (n_neighbors, ) array containing the cell_i logits
                neighbors: (n_neighbors, ) array containing the cell_i neighbor indices
        """
        if self._probabilities is None: 
            self._probabilities = probabilities
            self._logits = logits 
            self._indices = neighbors
            self._indptr = np.array([0, len(neighbors)])
        
        else:
            self._indices = np.hstack((self._indices, neighbors))
            self._indptr = np.hstack((self._indptr, self._indptr[-1]+len(neighbors)))
            self._probabilities = np.hstack((self._probabilities, probabilities))
            self._logits = np.hstack((self._logits, logits))

    
    @property
    def result(self):
        "Function that returns the csr_matrix for probabilities"
        return (csr_matrix((self._probabilities, 
                            self._indices, self._indptr), 
                            shape = (self._adata.shape[0], self._adata.shape[0])), 
                csr_matrix((self._logits,
                            self._indices, self._indptr), 
                            shape = (self._adata.shape[0], self._adata.shape[0]))
        )


    def __call__(self):
        "Function that computes the probability transition matrix"
        for idx, barcode in enumerate(self._adata.obs_names):
            neigh_idx, X = self.compute_displacement_vector(idx, key=self._key)
            v_i = self._velocities[idx]

            if(np.all(v_i)==0):
                probabilities, logits =  self.uniform(len(neigh_idx))
            else: 
                probabilities, logits = self._similarity(v_i, X, softmax_scale = self._softmax_scale)

            self.update_transitions(probabilities=probabilities, logits=logits, neighbors=neigh_idx)
        return self.result
        
