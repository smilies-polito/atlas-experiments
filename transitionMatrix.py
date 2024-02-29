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
    """Class that computes the transition matrix given a certain similarity metric.
        params:
            center_mean: boolean if to center velocity and displacement vectors around their mean. Default False
            scale_by_norm: boolean if to scale velocity and displacement vectors by their norm. Default False
    """
    def __init__(self, center_mean: bool = False, scale_by_norm: bool = False):
        self._center_mean = center_mean
        self._scale_by_norm = scale_by_norm 

    def __call__(self, v, X, softmax_scale: float = 1.0):
        """
        Computes transition matrix: 
            params: 
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
            params:
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
            params:
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
                 softmax_scale: float = 1.0, 
                 key: str = 'connectivities'
    ):
        """
            params:
                adata: AnnData of shape n_cells x n_peaks.
                velocities: np.ndarray of shape n_cells x n_peaks
                similarity: similarity class that implements the transition matrix computation strategy. 
                softmax_scale: float for the softmax computation. Default 1.0
                key: key in adata.obsp where to search for neighbor connectivities. Deafult 'connectivities'    
        """
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
        """Function that compute the displacement vector for the specific cell
        params:
            adata: annotated data
            idx: cell index integer
            key: where to look for the neighborhood
        """
        indptr, indices = self._adata.obsp[key].indptr, self._adata.obsp[key].indices
        start, end = indptr[idx], indptr[idx+1]
        neighbors_idx = indices[start:end]
        if issparse(self._adata.X):
            displacement = self._adata.X[neighbors_idx].A- self._adata.X[idx].A
        else:
            displacement =  self._adata.X[neighbors_idx]- self._adata.X[idx]

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
                probabilities, logits = self._similarity(v_i, X, softmax_scale = 1.0)

            self.update_transitions(probabilities=probabilities, logits=logits, neighbors=neigh_idx)

        return self.result
        

        


class TransitionMatrix:
    """
        Class that given an AnnData for scATAC-seq data computes transition matrix for MCMC.
        1. Computes displacement vector between one cell and its neighbors according to the neighborhood graph.
        2. Computes correlation between cell_i velocity and displacement vectors with neighbors. 
        4. Softmax to pass from correlations to probabilities

        There are some requirements:
            1. AnnData object contains scATAC-seq data in .X: the field is used to compute the displacement vector.
            2. AnnData object contains neighborhood graph in sparse csr_format: use of ._indptr and ._ indices to retrieve neighbors.
            3. AnnData object contains a key in adata.var which specifies for each peak the gene it is promoter to. 
    """


    def __init__(self, 
                 adata: AnnData, 
                 velocities: Union[np.ndarray, pd.DataFrame], 
                 promoter_key : Optional[str] = None, 
                 softmax_scale: Optional[float]= None
        ):
        """
            param:
                adata : AnnData
                promoter_key: string identifyinf a fiel in adata.var specifying the gene the peak is promoter of. It must be specifies if adata._var is used to retrieve 
                    the peak-gene relationship. Default None
                velocities: pandas.DataFrame containing velocities for the genes of shape n_cells x n_genes. If it is a pd.DataFrame it must contain cell bacordes
                    as row indices and gene names as columns names.
                softmax_scale: float indicating the softmax scale for probabilities computations. if None it is estimated. 
        """


        if isinstance(velocities, np.ndarray):
            if velocities.shape[0]!=adata.shape[0] or velocities.shape[1] != adata.shape[1]:
                raise ValueError(f"Shapes of velocity array {velocities.shape} and adata {adata.shape} do not match.")
            mask = np.isnan(velocities)
            velocities[mask] = 0
            self._velocities = velocities

        
        elif isinstance(velocities, pd.DataFrame):
            if velocities.shape[0]!= adata.shape[0] or not set(velocities.index) == set(adata.obs_names):
                raise ValueError(f"Barcordes no not match.")
            if promoter_key is None: 
                raise ValueError("With pd.DataFrame promoter_key field is mandatory")
            elif promoter_key not in adata.var.columns:
                raise ValueError(f"{promoter_key} not in adata.var")
            if not set(adata.var[promoter_key]).issubset(set(velocities.columns)):
                raise ValueError(f"Not all genes in adata.var[{promoter_key}] are in velocities")
            ref_genes = adata.var[promoter_key] 
            velocities = velocities.fillna(0) 
            self._velocities = velocities.get(ref_genes).to_numpy(dtype=np.float64) #shape is (n_cells, n_peaks) stores velocities for every gene whose peak is promoter per every cell
        

        self._adata = adata
        self._promoter_key = promoter_key
        self._softmax_scale = softmax_scale


    @property
    def key(self):
        return self._promoter_key
    
    
    def estimate_softmax_scale(self, similarity: Union[DotProduct, Cosine, Correlation, SimilarityComputer, Similarity], key: str = 'connectivities'):
        """
        Estimation of softmax scale. By cellrank definition, softmax scale is sigma = 1/median{|c_ik|} where c_ik are the logits obtained through the model 
        params:
            similarity: SimilarityComputer object for probabilities and logits computations. 
            key: where to search for neighbors for each cell. Default "connectivities" to search in adata.obsp['connectivities']
        """
        model = Deterministic(adata=self._adata, velocitites=self._velocities, similarity=similarity, softmax_scale=1.0, key=key)
        _ , logits = model()
        return 1.0/np.median(np.abs(logits.data))



    def compute_transition_matrix(self, 
                                  similarity: Union[Literal['correlation', 'cosine', 'dot'], Cosine, Correlation, DotProduct, SimilarityComputer, Similarity] = 'correlation', 
                                  key: str = 'connectivities'
    ):
        """
        Computes the transition probabilities and logits. 
        params: 
            similarity: similarity metric for probability computations, either a string or a SimilarityComputer object. Default: correlation. 
            key: where to search for neighbors for each cell. Default "connectivities" to search in adata.obsp['connectivities']
        """
        if isinstance(similarity, str):
            similarity = SimilarityWrapper.create(similarity)

        if self._softmax_scale is None:
            self._softmax_scale = self.estimate_softmax_scale(similarity, key)

        model = Deterministic(adata=self._adata, velocitites=self._velocities, similarity=similarity, softmax_scale=self._softmax_scale, key=key)
        self._transition_matrix, self._logits = model()


    @property
    def transition_matrix(self):
        return self._transition_matrix

            


