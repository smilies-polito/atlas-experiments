import os
import numpy as np
import pandas as pd
import scanpy as sc
import cellrank as cr
import scvelo as scv

from transitionMatrix import MultiOmics_TM
from utils import _check_conjugate, _compute_macrostates, _plot_quality_gpcca, _add_cell_type, _save_qualities_gpcca
from matrix_analysis import MatrixAnalyser
from typing import Sequence, Optional, Literal

class Simulator():
    """
    Class implementing the pipeline for multi omics data. 

    Attributes
    ----------
    _adata_path : str
        path where h5ad files are stored and saved 
    _k : int
        dimensionality of the scATAC-seq neighborhood
    _lsi : int
        number of dimensions for SVD
    _pc : int
        number of principal components
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
    def __init__(self, path: str, k:int, pc:int, lsi:int, save: bool = False):
        """

        Parameters
        ----------
        path : str
            path where h5ad files are stored and saved 
        k : int
            dimensionality of the scATAC-seq neighborhood
        lsi : int
            number of dimensions for SVD
        pc : int
            number of principal components
        save: bool
            whether to save the AnnData objects in h5ad format

        """
        self._adata_path = path
        self._seed = 52
        self._k = k
        self._pc = pc
        self._lsi = lsi
        self._save = save
        self._quality_dict = {}
        
        self._atac = sc.read_h5ad(os.path.join(self._adata_path, f'ATAC_{k}K{pc}PC{lsi}LSI.h5ad'))
        self._rna = sc.read_h5ad(os.path.join(self._adata_path, f'RNA_{k}K{pc}PC{lsi}LSI.h5ad'))
        #add column highly variable which is omitted by Signac but used in scVelo
        self._rna.var['highly_variable'] = np.ones(len(self._rna.var_names), dtype=bool) 

        self._cell_type_key = None


    def preprocessing(self, 
                       annotations: pd.DataFrame, 
                       cell_types: Optional[pd.DataFrame] = None, 
                       cell_type_key: str = "celltype", 
                       wsnn_key: str = 'wsnn',
                       gene_key: str = "gene",
                       peak_key: str = "peak"):
        
        """

        Preprocessing: computes scRNA-seq neighbors + RNA velocities using scVelo + subset both scRNA-seq and scATAC-seq
        according to promoter genes and peaks stored in the annotations. 

        Parameters
        -----------
        annotations: pandas.DataFrame 
            annotations containing promoter peaks and genes.
        peak_key: str 
            key in annotations.columns reative to peaks names  (default is "peak")
        gene_key: str 
            key annotations.columns relative to gene names (default is "gene")
        wsnn_key: str
            key where to move _rna.obsp['distances'] still in obs to preserve multi modal neighbors
        cell_type: pandas.DataFrame, optional
            containing cell clusters annotaitons to be added to scATAC-seq AnnData (default is None)   
        cell_type_key: str 
            key in _atac.obs.columns where celltype annotations are stored (default is "cell_type").  

        Updates 
        -------
        _cell_type_key : str
            key in _atac.obs.columns where celltype annotations are stored (default is "cell_type").  
        _rna : anndata.AnnData
            computes velocities using scVelo

        Returns
        -------
        annotations: pandas.DataFrame
            annotations containing promoter peaks and genes 

        """
        if cell_types is not None:
            _add_cell_type(self._atac, annotations=cell_types)
            _add_cell_type(self._rna, annotations=cell_types)

        if cell_type_key not in self._atac.obs.columns:
            raise IndexError(f"{cell_type_key} not in atac.obs")
        
        if peak_key not in annotations.columns or gene_key not in annotations.columns:
            raise IndexError(f"{peak_key} or {gene_key} not in annotations.columns")
        
        self._cell_type_key = cell_type_key

        #Reset rna.uns to be able to compute neighbors again based on scRNA only 
        self._rna.obsp[wsnn_key] = self._rna.obsp['distances']
        self._rna.uns.pop('neighbors', None)
 
        # Compute RNA-velocities 
        sc.pp.neighbors(self._rna, n_neighbors=self._k, n_pcs=self._pc, random_state = self._seed)
        scv.pp.moments(self._rna, n_neighbors=self._k, n_pcs=self._pc)
        scv.tl.recover_dynamics(self._rna, n_jobs=8)
        scv.tl.velocity(self._rna, mode='dynamical')
        scv.tl.velocity_graph(self._rna, n_jobs=-1)

        title = f"Dynamical model K={self._k}  LSI={self._lsi} PC={self._pc}"
        path= f"dynamicalModel_{self._k}K{self._lsi}LSI{self._pc}PC.png"
        scv.pl.velocity_embedding_stream(self._rna, basis='umap', color=[self._cell_type_key], title=title, save=path, show=False)

        #Compute latent time (cell's internal clock)
        title = f"Cell specific latent time K={self._k} LSI={self._lsi} PC={self._pc}"
        path= f"latentTime_{self._k}K{self._lsi}LSI{self._pc}PC.png"
        scv.tl.latent_time(self._rna)
        scv.pl.scatter(self._rna, color='latent_time', color_map='gnuplot', size=40, save=path, title=title, show=False)

        # Subset for promoter peanks and genes
        annotations = annotations[(annotations[gene_key].isin(self._rna.var_names)) & (annotations[peak_key].isin(self._atac.var_names))]
        intersection = set(annotations[peak_key]).intersection(set(self._atac.var_names))
        annotations = annotations[annotations[peak_key].isin(intersection)]
        self._atac = self._atac[:, self._atac.var_names.isin(intersection)]

        intersection = set(annotations[gene_key]).intersection(set(self._rna.var_names))
        annotations = annotations[annotations[gene_key].isin(intersection)]
        self._rna = self._rna[:, self._rna.var_names.isin(intersection)]   

        if self._save:
            path = os.path.join(self._adata_path, f'RNA_{self._k}K{self._pc}PC{self._lsi}LSI.h5ad')
            self._rna.write_h5ad(path)
            path = os.path.join(self._adata_path, f'ATAC_{self._k}K{self._pc}PC{self._lsi}LSI.h5ad')
            self._atac.write_h5ad(path)

        return annotations

    
    def compute_transition_matrix(self, annotations: pd.DataFrame, perform_analysis:bool=True, velocity_key:str="velocity", 
                                   gene_key:str="gene", peak_key:str="peak", softmax_scale:Optional[float]=None, 
                                   similarity:Literal["cosine", "correlation", "dot"]="correlation", 
                                   key:str="wsnn", save_composition:bool=False):
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
        similarity: str
            indicates similarity metric for the transition matrix computations. Valid options are "cosine", "correlation", "dot" (default is "correlation")
        key: str
            key in _rna.obsp where neighborhood matrix is stored (default is "wsnn")
        softmax_scale: float, optional
            softmax_scale value for softmax computation (default is None)
        save_composition: bool
            whether to save nodes composition in condensation graph (default is False)

        Updates
        --------
        _atac.obsp['matrix']: scipy.sparse.csr_matrix
            transition matrix.

        """
        transitionMatrix = MultiOmics_TM(atac=self._atac, rna=self._rna, annotations=annotations, velocity_key=velocity_key,
                                         gene_key=gene_key, peak_key=peak_key, softmax_scale=softmax_scale)
        transitionMatrix.compute_transition_matrix(similarity=similarity, key=key)
        self._atac.obsp['matrix']= transitionMatrix.transition_matrix

        if perform_analysis:
            analyser = MatrixAnalyser(matrix = transitionMatrix.transition_matrix, adata=self._atac, cluster_key=self._cell_type_key, k=self._k, pc=self._pc, lsi=self._lsi)
            analyser._topology_analysis()
            analyser._condensation_graph(save_composition=save_composition)
            analyser._find_invariant()
            save_path = os.path.join(os.getcwd(), "multiomics_params.csv")
            analyser._save_params(path=save_path)


    def cell_rank(self, start_ixs: Sequence, plot: bool = False, save_fate:bool = True):
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
            title = f'Random Walk K={self._k} LSI={self._lsi} PC={self._pc}'
            path = f'random_walk_{self._k}K{self._lsi}LSI{self._pc}PC.png'
            kernel.plot_random_walks(start_ixs = start_ixs , n_sims=200, seed=self._seed, title=title, save=path)

        g = cr.estimators.GPCCA(kernel)
        g.compute_schur()
        self._gpcca = g

        if plot:
            title = f'Schur decomposition K={self._k} LSI={self._lsi} PC={self._pc}' 
            path = f'schurDecomposition_{self._k}KLSI={self._lsi}{self._pc}PC.png'
            g.plot_spectrum(title=title, save=path, real_only=True)

        eigenvalues = g.eigendecomposition['D']
        idx = 1

        while idx<10:
            eig = eigenvalues[idx]
            idx = _check_conjugate(idx, eig)
            try:
                self._quality_dict[idx] = _compute_macrostates(n_states=idx, gpcca=self._gpcca,  plot=plot, save_fate=save_fate, barcodes=self._rna.obs_names,
                                                               cell_type_key=self._cell_type_key, k=self._k, pc=self._pc, lsi=self._lsi)
            except ValueError as e:
                print(e)

        df = pd.DataFrame(self._quality_dict, index = ["spectral_gap", 'minChi', 'crispness']).T
        
        if plot: 
            _plot_quality_gpcca(df, k=self._k, pc=self._pc, lsi=self._lsi)
        
        _save_qualities_gpcca(df, k=self._k, pc=self._pc, lsi=self._lsi)
        


if __name__ == "__main__": 
    from itertools import product

    k = [10,20,30,50,60,80]
    lsi = [10,15,20,25,30]
    pc =  [10,15,20,25,30]
    grid = product(k, lsi, pc)

    annotations = pd.read_csv(os.path.join(os.getcwd(), 'peak_annotation.tsv'), sep='\t', header = 0)
    annotations = annotations[annotations.peak_type == 'promoter']
    annotations['peak'] = annotations[['chrom', 'start', 'end']].apply(lambda row: '-'.join(row.values.astype(str)), axis=1)

    start_ixs = pd.read_csv(os.path.join(os.getcwd(), 'rw_starting_barcodes.csv'), header=0, index_col=0, sep=",").values.flatten()
    adata_path = os.path.join(os.getcwd(), 'data_folder', 'multiomics')

    cell_type = pd.read_csv(os.path.join(os.getcwd(), 'cell_annotations.tsv'), sep='\t', header = 0, index_col=0)
    cell_type_key = cell_type.columns[0]    

    for k, pc, lsi in grid:
        sim = Simulator(path = adata_path, k=k, pc = pc, lsi=lsi, save=False)
        annotations = sim.preprocessing(annotations=annotations, cell_types=cell_type, cell_type_key=cell_type_key,
                        wsnn_key="wsnn", gene_key="gene", peak_key="peak")
        print(annotations.shape, sim._atac.shape, sim._rna.shape)
        sim.compute_transition_matrix(annotations=annotations, perform_analysis=True, velocity_key="velocity", save_composition=False,
                                    gene_key="gene", peak_key="peak", softmax_scale="None", key="wsnn", similarity="correlation")
        sim.cell_rank(start_ixs = start_ixs, plot=True, save_fate=True)
        del sim