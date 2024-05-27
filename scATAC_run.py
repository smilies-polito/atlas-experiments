import os
import pandas as pd
import scanpy as sc
import cellrank as cr

from utils import _check_conjugate, _plot_quality_gpcca,_add_cell_type, _save_qualities_gpcca, _compute_macrostates
from transitionMatrix import ATAC_TM
from matrix_analysis import MatrixAnalyser
from typing import Sequence, Literal, Optional

class Simulator():
    """
    Class implementing the pipeline for scATAC-seq data. 

    Attributes
    ----------
    _adata_path : str
        path where h5ad files are stored and saved 
    _k : int
        dimensionality of the scATAC-seq neighborhood
    _lsi : int
        number of dimensions for SVD
    _save: bool
        whether to save the AnnData objects in h5ad format
    _quality_dict: dict
        dict storing GPCCA macrostate quality results
    _atac : anndata.AnnData
        scATAC-seq AnnData
    _rna : anndata.AnnData
        scRNA-seq AnnData
    _cell_type_key: str
        str in _atac where cell types are stored

    """
    def __init__(self, path: str, k:int, lsi:int, save: bool = False):
        """
        Parameters
        ----------
        path : str
            path where h5ad files are stored and saved 
        k : int
            dimensionality of the scATAC-seq neighborhood
        lsi : int
            number of dimensions for SVD
        save: bool
            whether to save the AnnData objects in h5ad format

        """
        self._adata_path = path
        self._seed = 52
        self._k = k
        self._lsi = lsi
        self._save = save
        self._quality_dict = {}
        self._atac = sc.read_h5ad(os.path.join(self._adata_path, f'scATAC_{k}K{lsi}LSI.h5ad'))
        self._rna = None
        self._cell_type_key = None


    def preprocessing(self, annotations: pd.DataFrame, k:int=30, pc:int=20, peak_key:str="peak", gene_key:str="gene", 
                            cell_types: Optional[pd.DataFrame]= None, cell_type_key: str='celltype'):
        """
        Preprocessing: gets RNA velocities from an scRNA-seq specified embedding and subset both scRNA-seq and scATAC-seq
        according to promoter genes and peaks stored in the annotations. 

        Parameters
        -----------
        annotations: pandas.DataFrame 
            annotations containing promoter peaks and genes.
        peak_key: str 
            key in annotations.columns reative to peaks names  (default is "peak")
        gene_key: str 
            key annotations.columns relative to gene names (default is "gene")
        k: int
            size of the neighborhood of the scRNA-seq AnnData (default is 30)
        pc: int
            number of principal components of the scRNA-seq AnnData (default is 20) 
        cell_types: pandas.DataFrame, optional
            containing cell clusters annotaitons to be added to scATAC-seq AnnData (default is None)   
        cell_type_key: str 
            key in _atac.obs.columns where celltype annotations are stored (default is "cell_type").  

        Updates 
        -------
        _cell_type_key : str
            key in _atac.obs.columns where celltype annotations are stored (default is "cell_type").  
        _rna : anndata.AnnData
            scRNA-seq embedding for RNA-velocities.    

        Returns
        -------
        annotations: pandas.DataFrame
            annotations containing promoter peaks and genes 
            
        """
        if peak_key not in annotations.columns or gene_key not in annotations.columns:
            raise KeyError(f"{peak_key} or {gene_key} not in annotations.columns")
        
        if cell_types is not None: 
            _add_cell_type(self._atac, annotations=cell_types)

        if cell_type_key not in self._atac.obs.columns:
            raise KeyError(f"{cell_type_key} not in atac.obs.columns.")
        
        self._cell_type_key=cell_type_key
    
        try:
            self._rna = sc.read_h5ad(os.path.join(self._adata_path, f'scRNA_{k}K{pc}PC.h5ad'))

            # Subset for common promoter and peak and genes.
            annotations = annotations[(annotations[gene_key].isin(self._rna.var_names)) & (annotations[peak_key].isin(self._atac.var_names))]
            intersection = set(annotations[peak_key]).intersection(set(self._atac.var_names))
            annotations = annotations[annotations[peak_key].isin(intersection)]
            self._atac = self._atac[:, self._atac.var_names.isin(intersection)]

            intersection = set(annotations[gene_key]).intersection(set(self._rna.var_names))
            annotations = annotations[annotations[gene_key].isin(intersection)]
            self._rna = self._rna[:, self._rna.var_names.isin(intersection)]   
        
            if self._save:
                path = os.path.join(self._adata_path, f"scRNA_{k}K{pc}PC.h5ad")
                self._rna.write_h5ad(path)
                path = os.path.join(self._adata_path, f"scATAC_{self._k}K{self._pc}LSI.h5ad")
                self._rna.write_h5ad(path)

            return annotations
        
        except:
            raise FileNotFoundError("Impossible to load RNA")
        
            

    def compute_transition_matrix(self, annotations: pd.DataFrame, peak_key:str, gene_key:str, 
                                   velocity_key:str, perform_analysis: bool = False, save_composition: bool=False,
                                   similarity: Literal["cosine", "correlation", "dot"] = 'correlation', 
                                   key: str = "connectivities", softmax_scale:Optional[float]=None):
        """
        Computes transition matrix for CellRank.
        
        Parameters
        ----------
        annotations: pandas.DataFrame 
            annotations containing promoter peaks and genes.
        peak_key: str 
            key in annotations.columns reative to peaks names
        gene_key: str 
            key annotations.columns relative to gene names
        velocity_key: str 
            key in _rna.layers where RNA-velocities are stored
        perform_analysis: bool
            whether to perform analysis of the graph associated with the transition matrix (default is False)
        save_composition: bool
            whether to save nodes composition in condensation graph (default is False)
        similarity: str
            indicates similarity metric for the transition matrix computations. Valid options are "cosine", "correlation", "dot" (default is "correlation")
        key: str
            key in _atac.obsp where neighborhood matrix is stored (default is "connectivities")
        softmax_scale: float, optional
            softmax_scale value for softmax computation (default is None)

        Updates
        --------
        _atac.obsp['matrix']: scipy.sparse.csr_matrix
            transition matrix.

        """
        transitionMatrix = ATAC_TM(atac=self._atac, rna= self._rna, annotations=annotations, peak_key=peak_key, gene_key=gene_key, velocity_key=velocity_key, softmax_scale=softmax_scale)
        transitionMatrix.compute_transition_matrix(similarity=similarity, key=key)
        self._atac.obsp['matrix']= transitionMatrix.transition_matrix

        if perform_analysis:
            analyser = MatrixAnalyser(matrix = transitionMatrix.transition_matrix, adata=self._atac, cluster_key=self._cell_type_key, k=self._k, lsi=self._lsi)
            analyser._topology_analysis()
            analyser._condensation_graph(save_composition=save_composition)
            analyser._find_invariant()
            save_path = os.path.join(os.getcwd(), "scATAC_params.csv")
            analyser._save_params(path=save_path)


    def cell_rank(self, start_ixs: Sequence, plot: bool = False, save_fate: bool = False):
        """ 
        
        Function that performs CellRank simulation.

        Parameters
        -----------
        start_ixs: Sequence
            barcodes as initial distribution for the random walk.
        plot: bool
            if to plot and save schur decomposition, random walk and results for GPCCA (default is False)
        save_fate: bool
            if to save fate probabilities as .csv (default is False) 

        Updates
        -------
        ._gpcca : cellrank.estimators.GPCCA
            GPCCA object
            
        """

        kernel= cr.kernels.PrecomputedKernel(object=self._atac, obsp_key='matrix')
    
        if plot:
            title = f'Random Walk K={self._k} LSI={self._lsi}'
            path = f'random_walk_{self._k}K{self._lsi}LSI.png'
            kernel.plot_random_walks(start_ixs = start_ixs , n_sims=200, seed=self._seed, title=title, save=path)

        g = cr.estimators.GPCCA(kernel)
        g.compute_schur()
        self._gpcca = g

        if plot:
            title = f'Schur decomposition K={self._k} LSI={self._lsi}' 
            path = f'schurDecomposition_{self._k}K{self._lsi}LSI.png'
            g.plot_spectrum(title=title, save=path, real_only=True)

        eigenvalues = g.eigendecomposition['D']
        idx = 1

        while idx<10:
            eig = eigenvalues[idx]
            idx = _check_conjugate(idx, eig)
            try:
                self._quality_dict[idx] = _compute_macrostates(n_states=idx, gpcca=self._gpcca, plot=plot, save_fate=save_fate, barcodes=self._atac.obs_names,
                                                                cell_type_key=self._cell_type_key, k=self._k, lsi=self._lsi)
            except ValueError as e:
                print(e)

        df = pd.DataFrame(self._quality_dict, index = ["spectral_gap", 'minChi', 'crispness']).T
        
        if plot: 
            _plot_quality_gpcca(df, k=self._k, lsi=self._lsi)

        _save_qualities_gpcca(df, k=self._k, lsi=self._lsi)


if __name__ == "__main__": 
    from itertools import product

    k = [10]
    lsi = [20]
    grid = product(k,lsi)

    annotations = pd.read_csv(os.path.join(os.getcwd(), 'peak_annotation.tsv'), sep='\t', header = 0)
    annotations = annotations[annotations.peak_type == 'promoter']
    annotations['peak'] = annotations[['chrom', 'start', 'end']].apply(lambda row: '-'.join(row.values.astype(str)), axis=1)

    start_ixs = pd.read_csv(os.path.join(os.getcwd(), 'rw_starting_barcodes.csv'), header=0, index_col=0, sep=",").values.flatten()
    adata_path = os.path.join(os.getcwd(), 'data_folder', 'scATAC')

    cell_type = pd.read_csv(os.path.join(os.getcwd(), 'cell_annotations.tsv'), sep='\t', header = 0, index_col=0)
    cell_type_key = cell_type.columns[0]

    for k, lsi in grid:
        sim = Simulator(k=k, lsi=lsi, path=adata_path, save=False)
        annotations = sim.preprocessing(annotations=annotations, cell_types=cell_type)
        sim.compute_transition_matrix(annotations=annotations, peak_key="peak", gene_key="gene", velocity_key="velocity",
                                    perform_analysis=True, save_composition=False, key="distances")
        sim.cell_rank(start_ixs = start_ixs, plot=True, save_fate=True)
    