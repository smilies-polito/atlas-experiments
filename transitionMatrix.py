import pandas as pd
import numpy as np

from enum import Enum
from scipy.sparse import csr_matrix, csr_array

class Similarity(Enum):
    """Enum that imitates CellRank similarity computations"""
    DOT_PRODUCT = "dot"
    COSINE = "cosine"
    CORRELATION = "correlation"


class SimilarityWrapper():
    """Class that allows to create and manage different similarities computations"""
    def create(similarity):
        """Returns a SimilarityComputer object"""
        return Correlation() if similarity=="correlation" else Cosine() if similarity == 'cosine' else DotProduct()
    

class SimilarityComputer():
    """Class that computes the transition matrix given a certain similarity metric."""
    def __init__(self, center_mean, scale_by_norm):
        self._center_mean = center_mean
        self._scale_by_norm = scale_by_norm 

    def __call__(self, v, X, softmax_scale):
        """
        Computes transition matrix: 
            params: 
                v: velocity vector for cell_i
                X: displacement vector between cell_i and its neighbors
                sotftmax_scale: softmax scale for softmax computation
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
    

    def softmax_masked(self, x, mask, softmax_scale):
        """
        Computes softmax and returns probabilities.
            params:
                x: vectors of logits from which compute the proebabilities
                mask: mask indicating which part of x result from a divsion by 0
                softmax_scale: softmax scale for softmax computation
        """
        numerator = x*softmax_scale
        numerator = np.exp(numerator - np.nanmax(numerator))
        numerator = np.where(mask, 0, numerator)
        return numerator/np.nansum(numerator), x
    

    def softmax(self, x, softmax_scale):
        """
        Computes softmax and returns probabilities.
            params:
                x: vectors of logits from which compute the proebabilities
                softmax_scale: softmax scale for softmax computation
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
    
    def __init__(self, adata, velocitites, similarity, softmax_scale, key='connectivities'):
        self._adata = adata
        self._similarity = similarity
        self._key = key
        self._velocities = velocitites
        self._softmax_scale = softmax_scale
        self._probabilities = None
        self._logits = None
        self._indices = None
        self._indptr = None


    def uniform(self, n_neighbors):
        """Function that returns uniform distribution over the neighbors.
            prarams: 
                n_neighbors: number of neighbors for cell_i"""
        return np.ones(n_neighbors)/n_neighbors, np.zeros(n_neighbors)
    

    def compute_displacement_vector(self, idx, key='connectivities'):
        """Function that compute the displacement vector for the specific cell
        params:
            adata: annotated data
            idx: cell index integer
            key: where to look for the neighborhood
        """
        indptr, indices = self._adata.obsp[key].indptr, self._adata.obsp[key].indices
        start, end = indptr[idx], indptr[idx+1]
        neighbors_idx = indices[start:end]
        displacement = self._adata.X[neighbors_idx]- self._adata.X[idx]
        displacement = self._adata.X[neighbors_idx].toarray() - self._adata.X[idx].toarray()
        return displacement
    

    def update_transitions(self, probabilities, logits, neighbors):
        """
        Function that updates the object's fields to construct csr_matrix. 
            params:
                probabilities: (n_neighbors,) array containing the cell_i probabilities of transitioning
                logits: (n_neighbors, ) array containing the cell_i logits
                neighbors: (n_neighbors, ) array containing the cell_i neighbor indices
        """
        if self._data is None: 
            self._probabilities = probabilities
            self._logits = logits 
            self._indices = neighbors
            self._indptr = np.array([len(neighbors)])
        
        else:
            self._indices = np.hstack((self._indices, neighbors))
            self._indptr = np.hstack(self._indptr, len(neighbors))
            self._probabilities = np.hstack(self._probabilities, probabilities)
            self._logits = np.hstack(self._logits, logits)
    
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
        for idx, barcode in enumerate(self._adata.var_names):
            neigh_idx, X = self.compute_displacement_vector(idx, key=self._key)
            v_i = self._velocities.iloc[idx]
            
            if(np.all(v_i)==0):
                return self.uniform(len(neigh_idx))
            
            probabilities, logits = self._similarity(v_i, X, softmax_scale = 1.0)
            self.update_transitions(probabilities=probabilities, logits=logits, neighbors=neigh_idx)

        return self.result()
        

        


class TransitionMatrix_ATAC:
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


    def __init__(self, adata, velocities, promoter_key = None, softmax_scale=None):
        """
            param:
                adata : AnnData
                promoter_key: string identifyinf a fiel in adata.var specifying the gene the peak is promoter of. It must be specifies if adata._var is used to retrieve the peak-gene relationship.
                    Defaul None
                velocities: pandas.DataFrame containing velocities for the genes of shape n_cells x n_genes. If it is a pd.DataFrame it must contain cell bacordes
                    as row indices and gene names as columns names.
                softmax_scale: float indicating the softmax scale for probabilities computations. if None it is estimated. 
        """
        velocities = velocities.fillna(0) #if Nans are present
        ref_genes = adata.var[promoter_key] 
        self._velocities = velocities.get(ref_genes) #shape is (n_cells, n_peaks) stores velocities for every gene whose peak is promoter per every cell
        self._adata = adata
        self._promoter_key = promoter_key
        self._softmax_scale = softmax_scale
        

    @property
    def key(self):
        return self._promoter_key
    
    
    def estimate_softmax_scale(self, similarity, key):
        """
        Estimation of softmax scale. By cellrank definition, softmax scale is sigma = 1/median{|c_ik|} where c_ik are the logits obtained through the model 
        params:
            similarity: SimilarityComputer object for probabilities and logits computations. 
            key: where to search for neighbors for each cell. Default "connectivities" to search in adata.obsm['connectivities']
        """
        model = Deterministic(adata=self._adata, velocitites=self._velocities, similarity=similarity, softmax_scale=1.0, key=key)
        _ , logits = model()
        return 1.0/np.median(np.abs(logits)) #DA VEDERE PERCHE' LOGITS.DATA IN CELLRANK !!!!



    def compute_transition_matrix(self, similarity = 'correlation', key = 'connectivities'):
        
        if isinstance(similarity, str):
            similarity = SimilarityWrapper.create(similarity)

        if self._softmax_scale is None:
            self._softmax_scale = self.estimate_softmax_scale(similarity, key)

        model = Deterministic(adata=self._adata, velocitites=self._velocities, similarity=similarity, softmax_scale=self._softmax_scale, key=key)
        probabilities, logits = model()

            


