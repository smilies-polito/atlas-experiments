import numpy as np
import scipy as sc
import pandas as pd

from anndata import AnnData
from model import Deterministic, Similarity, SimilarityWrapper, SimilarityComputer, Correlation, Cosine, DotProduct
from scipy.sparse import csr_matrix, issparse, hstack
from typing import Literal, Union, Optional

def _expand_matrix(adata:AnnData, annotations: pd.DataFrame, peak_key:str, gene_key:str):
    grouped = annotations[[peak_key, gene_key]].groupby(by=[peak_key]).size()
    peaks, howMuch  = grouped.index, grouped.values
    indices = [np.where(adata.var_names==peak)[0][0] for peak in peaks]
    indices = np.repeat(indices, howMuch)
    return adata.X[:, indices]


def _map_velocities(adata: AnnData, annotations:pd.DataFrame, velocity_key:str, peak_key:str, gene_key:str):
    velocities = pd.DataFrame(adata.layers[velocity_key], index=adata.obs_names, columns = adata.var_names)
    annotations = annotations.sort_values(by=peak_key)
    return velocities[annotations[gene_key].values].values


class TransitionMatrix:
    """

    Class that computes transition matrix for the MCMC.
    1. Computes displacement vector between one cell and its neighbors according to the neighborhood graph.
    2. Computes correlation between cell_i velocity and displacement vectors with neighbors. 
    4. Softmax to pass from correlations to probabilities

    Attributes
    -----------
    _adata: anndata.AnnData
    _X: numpy.ndarray or scipy.sparse.csr_matrix
        peak-barcode matrix over which to compute transition probability matrix.
    _velocities: numpy.ndarray
        velocity vector wose shape should match X.shape 
    _softmax_scale: float, optional
        softmax_scale value for softmax computation (default is None)  

    """

    def __init__(self, 
                 adata: AnnData,
                 velocities: np.ndarray,  
                 X: Union[np.ndarray, csr_matrix], 
                 softmax_scale: Optional[float]= None, 
                 
        ):
        """

        Parameters
        -----------
        adata: anndata.AnnData
        X: numpy.ndarray or scipy.sparse.csr_matrix
            peak-barcode matrix over which to compute transition probability matrix.
        velocities: numpy.ndarray
            velocity vector wose shape should match X.shape 
        softmax_scale: float, optional
            softmax_scale value for softmax computation (default is None)

        """
        
        if not velocities.shape==X.shape:
            raise ValueError("Velocities shape does not match with X shape")

        self._velocities = velocities
        self._adata = adata
        self._X = X
        self._softmax_scale = softmax_scale
    
    
    def estimate_softmax_scale(self, similarity: Union[DotProduct, Cosine, Correlation, SimilarityComputer, Similarity], key: str = 'connectivities'):
        """

        Function that estimates the softmax scale. softmax scale is sigma = 1/median{|c_ik|} where c_ik are the logits obtained through the model 
        
        Parameters
        ----------
        similarity: SimilarityComputer or Similarity or Cosine, Correlation, DotProduct
            object for the transition matrix computations.
        key: str
            key in _adata.obsp where neighborhood matrix is stored (default is "connectivities")

        Returns
        --------
        float: softmax scale

        """
        model = Deterministic(adata=self._adata, X=self._X, velocitites=self._velocities, similarity=similarity, 
                              softmax_scale=1.0, key=key)
        _ , logits = model()
        return 1.0/np.median(np.abs(logits.data))


    def compute_transition_matrix(self, 
                                  similarity: Union[Literal['correlation', 'cosine', 'dot'], Cosine, Correlation, DotProduct, SimilarityComputer, Similarity] = 'correlation', 
                                  key: str = 'connectivities'
    ):
        """

        Function that computes transition probabilities and logits.
        
        Parameters
        ----------
        similarity: SimilarityComputer or Similarity or Cosine, Correlation, DotProduct or string among "correlation", "cosine", "dot" (default is "correlation")
            object for the transition matrix computations.
        key: str
            key in _adata.obsp where neighborhood matrix is stored (default is "connectivities")

        Updates
        --------
        _transition_matrix: scipy.sparse.csr_matrix
        _logits: scipy.sparse.csr_matrix

        """
        if isinstance(similarity, str):
            similarity = SimilarityWrapper.create(similarity)

        if self._softmax_scale is None:
            self._softmax_scale = self.estimate_softmax_scale(similarity, key)

        model = Deterministic(adata=self._adata, X=self._X, velocitites=self._velocities, similarity=similarity, 
                              softmax_scale=self._softmax_scale, key=key)
        self._transition_matrix, self._logits = model()


    @property
    def transition_matrix(self):
        return self._transition_matrix

            

class MultiVeloTM(TransitionMatrix):
    """

    Class that computes the transtion matrix for multi omics data and Multivelo velocities.

    Attributes
    -----------
    _adata : anndata.AnnData
        AnnData 
    _X : scipy.sparse.csr_matrix
        displacement vector
    _velocities: np.ndarray
        velocity vector

    """
    def __init__(self, 
                 adata: AnnData,
                 atac_key: str = "ATAC",
                 rna_key: Optional[str] = None,
                 atac_velo_key : str = "velo_chrom",
                 rna_velo_key: str = "velo_s",
                 softmax_scale: Optional[float]= None, 
                 
    ):
        """
        
        Params
        ------
        adata: anndata.AnnData 
            scRNA-seq + scATAC-seq AnnData resulting from Multivelo pipeline.
        rna_key: str, optional 
            key in adata.layers containing scRNA-seq data (default is None). If None, adata.X is used for scRNA-seq values.
        atac_key: str
            key in adata.layers containing scATAC-seq data (default is Multivelo "ATAC")
        atac_velo_key: str
            key in adata.layers containing chromatin velocities (default is Multivelo "velo_chrom")
        rna_velo_key: str
            key in adata.layers containing RNA velocities (default is Multivelo "velo_s")
        softmax_scale: float, optional
            softmax_scale value for softmax computation (default is None)
            
        """

        X = sc.hstack([adata.X, adata.layers[atac_key]]) if rna_key is None else sc.hstack([adata.layers[rna_key], adata.layers[atac_key]]) 
        velocities = np.hstack([adata.layers[rna_velo_key], adata.layers[atac_velo_key]])
        super().__init__(adata=adata, X=X, velocities=velocities, softmax_scale=softmax_scale)


class ATAC_TM(TransitionMatrix):
    """

    Class that computes the transtion matrix for scATAC-seq data.

    Attributes
    -----------
    _adata : anndata.AnnData
        AnnData 
    _X : scipy.sparse.csr_matrix
        displacement vector
    _velocities: np.ndarray
        velocity vector

    """
    def __init__(self, atac:AnnData, rna: AnnData, annotations: pd.DataFrame, peak_key:str='peak', gene_key:str='gene', 
                 velocity_key:str='velocity', softmax_scale: Optional[float]=None):
        """
        
        Params
        ------
        atac: anndata.AnnData 
            scATAC-seq AnnData
        rna: anndata.AnnData
            scRNA-seq AnnData
        annotations: pandas.DataFrame 
            annotations containing promoter peaks and genes.
        peak_key: str 
            key in annotations.columns reative to peaks names  (default is "peak")
        gene_key: str 
            key annotations.columns relative to gene names (default is "gene")
        velocity_key: str 
            key in rna.layers where RNA-velocities are stored
        softmax_scale: float, optional
            softmax_scale value for softmax computation (default is None)

        """
        
        if peak_key not in annotations.columns or gene_key not in annotations.columns:
            raise IndexError(f"{peak_key} or {gene_key} not in annotations columns.")
        
        if velocity_key not in rna.layers:
            raise IndexError(f"{velocity_key} not in scRNA-seq AnnData layers.")            

        # Creation of ATAC expanded matrix
        X = _expand_matrix(adata=atac, annotations= annotations, peak_key=peak_key, gene_key=gene_key)
        # Creation of expanded velocity matrix
        velocities = _map_velocities(adata=rna, annotations=annotations, velocity_key=velocity_key, peak_key=peak_key, gene_key=gene_key)

        super().__init__(adata=atac, velocities=velocities, X=X, softmax_scale=softmax_scale)




class MultiOmics_TM(TransitionMatrix): 
    """

    Class that computes the transtion matrix for multi omics data.

    Attributes
    -----------
    _adata : anndata.AnnData
        AnnData 
    _X : scipy.sparse.csr_matrix
        displacement vector
    _velocities: np.ndarray
        velocity vector

    """

    def __init__(self, atac: AnnData, rna: AnnData, annotations: pd.DataFrame, velocity_key: str= "velocity",
                 gene_key:str = 'gene', peak_key : str='peak', softmax_scale: Optional[float]=None):
        """
        
        Params
        ------
        atac: anndata.AnnData 
            scATAC-seq AnnData
        rna: anndata.AnnData
            scRNA-seq AnnData
        annotations: pandas.DataFrame 
            annotations containing promoter peaks and genes.
        peak_key: str 
            key in annotations.columns reative to peaks names  (default is "peak")
        gene_key: str 
            key annotations.columns relative to gene names (default is "gene")
        velocity_key: str 
            key in rna.layers where RNA-velocities are stored
        softmax_scale: float, optional
            softmax_scale value for softmax computation (default is None)

        """
            
        if peak_key not in annotations.columns or gene_key not in annotations.columns:
            raise IndexError(f"{peak_key} or {gene_key} not in annotations.columnms")
        
        # if(atac.shape[0]!=rna.shape[0] or len(np.intersect1d(atac.obs_names, rna.obs_names)) < len(atac.obs_names)):
        #     raise ValueError("Barcodes do not match between atac and rna")

        if velocity_key not in rna.layers:
             raise IndexError(f"{velocity_key} not in rna.layers")

        X_atac = _expand_matrix(adata=atac, annotations=annotations, peak_key=peak_key, gene_key=gene_key)
        atac_velocities = _map_velocities(annotations=annotations, adata=rna, velocity_key= velocity_key, peak_key=peak_key, gene_key=gene_key)
        X = csr_matrix(hstack([X_atac, rna.X]))
        velocities = np.hstack((atac_velocities, rna.layers[velocity_key]))

        super().__init__(adata=rna, X=X, velocities=velocities, softmax_scale=softmax_scale)
