import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix


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

    def __init__(self, adata, velocities, promoter_key = None):
        """
            param:
                adata : AnnData
                promoter_key: string identifyinf a fiel in adata.var specifying the gene the peak is promoter of. It must be specifies if adata._var is used to retrieve the peak-gene relationship.
                    Defaul None
                velocities: pandas.DataFrame containing velocities for the genes of shape n_cells x n_genes. If it is a pd.DataFrame it must contain cell bacordes
                    as row indices and gene names as columns names.
        """
        velocities = velocities.fillna(0) #if Nans are present
        ref_genes = adata.var[promoter_key] 
        self._velocities = velocities.get(ref_genes) #shape is (n_cells, n_peaks) stores velocities for every gene whose peak is promoter per every cell
        self._adata = adata
        self._promoter_key = promoter_key


    @property
    def key(self):
        return self._promoter_key
    

    def compute_displacement_vector(self, idx, where='connectivities'):
        """Function that compute the displacement vector for the specific cell
        params:
            adata: annotated data
            idx: cell index integer
            where: where to look for the
        """
        indptr, indices = self._adata.obsp[where].indptr, self._adata.obsp[where].indices
        start, end = indptr[idx], indptr[idx+1]
        neighbors_idx = indices[start:end]
        W = self._adata.X[neighbors_idx]- self._adata.X[idx]
        W = self._adata.X[neighbors_idx].toarray() - self._adata.X[idx].toarray()
        return W
    

    def compute_transition_matrix(self, where = 'connectivities'):
        for idx, barcode in enumerate(self._adata.var_names):
            neigh, X = self.compute_displacement_vector(idx, where=where)
            v_i = self._velocities.iloc[idx]
            dot_prod = np.dot(X, v_i) 


# MANCA ESPANDERE E FARE SOFTMAX + salvarmela da qualche parte :)



