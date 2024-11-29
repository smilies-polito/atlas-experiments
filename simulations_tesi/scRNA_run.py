import os
import csv
import numpy as np
import pandas as pd
import scanpy as sc
import cellrank as cr
import scvelo as scv
import matplotlib.pyplot as plt

from utils import _check_conjugate, _compute_macrostates, _plot_quality_gpcca, _add_cell_type, _save_qualities_gpcca
from matrix_analysis import MatrixAnalyser
from typing import Sequence, Optional, Literal

class Simulator():
    """
    Class implementing the pipeline for scRNA-seq data. 

    Attributes
    ----------
    _adata_path : str
        path where h5ad files are stored and saved 
    _k : int
        dimensionality of the scATAC-seq neighborhood
    _pc : int
        number of principal components
    _save: bool
        whether to save the AnnData objects in h5ad format
    _quality_dict: dict
        dict storing GPCCA macrostate quality results
    _adata : anndata.AnnData
        scRNA-seq AnnData
    _cell_type_key: str
        str in _atac where cell types are stored

    """
    def __init__(self, path: str, k:int, pc:int, save: bool = False ):
        """
        Parameters
        ----------
        path : str
            path where h5ad files are stored and saved 
        k : int
            dimensionality of the scATAC-seq neighborhood
        pc : int
            number of principal components
        save: bool
            whether to save the AnnData objects in h5ad format

        """
        self._adata_path = path
        self._seed = 52
        self._k = k
        self._pc = pc
        self._save = save
        self._quality_dict = {}
        self._cell_type_key = None

        self._adata = sc.read_h5ad(os.path.join(self._adata_path, f'scRNA_{k}K{pc}PC.h5ad'))
        #add column highly variable which is omitted by Signac but used in scVelo
        self._adata.var['highly_variable'] = np.ones(len(self._adata.var_names), dtype=bool) 


    def preprocessing(self, cell_types: Optional[pd.DataFrame] = None , cell_type_key: Optional[str] = None):
        """
        Preprocessing: neighbors, louvain, umap + moments and ScVelo RNA-velocities. 

        Parameters
        -----------
        cell_type: pandas.DataFrame, optional
            containing cell clusters annotaitons to be added to scRNA-seq AnnData (default is None)   
        cell_type_key: str, optinal
            key in _atac.obs.columns where celltype annotations are stored. if not specified. then "louvain"

        Updates 
        -------
        _cell_type_key : str
            key in _atac.obs.columns where celltype annotations are stored (default is "cell_type").  
        _adata: AnnData  

        """
        sc.pp.neighbors(self._adata, n_neighbors=self._k, n_pcs=self._pc, random_state = self._seed)
        sc.tl.louvain(self._adata, resolution=1.0, random_state=self._seed)
        sc.tl.umap(self._adata, random_state=self._seed)
 
        title = f'K={self._k} PC={self._pc}'
        path =  f'_scRNA_{self._k}K{self._pc}PC.png'
        sc.pl.umap(self._adata, color=['louvain'], title=title, show=False, save=path)

        if cell_types is not None:
            _add_cell_type(self._adata, cell_types)

        self._cell_type_key = cell_type_key if (cell_type_key is not None and cell_type_key not in self._adata.obs.columns) else "louvain"

        scv.pp.moments(self._adata, n_neighbors=self._k, n_pcs=self._pc)
        scv.tl.recover_dynamics(self._adata, n_jobs=4)
        scv.tl.velocity(self._adata, mode='dynamical')
        scv.tl.velocity_graph(self._adata, n_jobs=-1)

        title = f"Dynamical model K={self._k} PC={self._pc}"
        path= f"dynamicalModel_{self._k}K{self._pc}PC.png"
        scv.pl.velocity_embedding_stream(self._adata, basis='umap', color=[self._cell_type_key], title=title, save=path, show=False)

        #Compute latent time (cell's internal clock)
        title = f"Cell specific latent time K={self._k} PC={self._pc}"
        path= f"latentTime_{self._k}K{self._pc}PC.png"
        scv.tl.latent_time(self._adata)
        scv.pl.scatter(self._adata, color='latent_time', color_map='gnuplot', size=40, save=path, title=title, show=False)

        if self._save:
            path = os.path.join(self._adata_path, f"scRNA_{self._k}K{self._pc}PC.h5ad")
            self._adata.write_h5ad(path)


    def compute_transition_matrix(self, velocity_key:str, perform_analysis: bool = False, save_composition:bool = False,
                                   similarity: Literal["cosine", "correlation", "dot"] = 'correlation', 
                                   key: str = "connectivities", softmax_scale:Optional[float]=None):
        """
        Computes transition matrix for CellRank.
        
        Parameters
        ----------
        velocity_key: str 
            key in _adata.layers where RNA-velocities are stored
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
        self._vk: cellrank.kernels.VelocityKernel

        """
        vk = cr.kernels.VelocityKernel(self._adata)
        vk.compute_transition_matrix(softmax_scale=softmax_scale, similarity=similarity)

        if perform_analysis:
            analyser = MatrixAnalyser(matrix = vk.transition_matrix, adata=self._adata, cluster_key=self._cell_type_key, k=self._k, pc=self._pc)
            analyser._topology_analysis()
            analyser._condensation_graph(save_composition=save_composition)
            analyser._find_invariant()
            save_path = os.path.join(os.getcwd(), "scRNA_params.csv")
            analyser._save_params(path=save_path)
        
        self._vk = vk



    def cell_rank(self, start_ixs: Sequence, plot: bool = False, save_fate:bool=True):
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

        if plot:
            title = f'Random Walk K={self._k} PC={self._pc}'
            path = f'random_walk_{self._k}K{self._pc}PC.png'
            self._vk.plot_random_walks(start_ixs = start_ixs , n_sims=200, seed=self._seed, title=title, save=path)

        g = cr.estimators.GPCCA(self._vk)
        g.compute_schur()
        self._gpcca = g

        if plot:
            title = f'Schur decomposition K={self._k} PC={self._pc}' 
            path = f'schurDecomposition_{self._k}K{self._pc}PC.png'
            g.plot_spectrum(title=title, save=path)

        eigenvalues = g.eigendecomposition['D']
        idx = 1

        while idx<10:
            eig = eigenvalues[idx]
            idx = _check_conjugate(idx, eig)
            try:
                self._quality_dict[idx] = _compute_macrostates(n_states=idx, gpcca=self._gpcca, plot=plot, save_fate=save_fate, barcodes=self._adata.obs_names,
                                                               cell_type_key=self._cell_type_key, k=k, pc=self._pc)
            except ValueError as e:
                print(e)

        df = pd.DataFrame(self._quality_dict, index = ["spectral_gap", 'minChi', 'crispness']).T
        
        if plot: 
            _plot_quality_gpcca(df, k=self._k, pc=self._pc)
        
        _save_qualities_gpcca(df, minChi_threshold=0.1, crispness_threshold=0.6, k=self._k, pc=self._pc)


if __name__ == "__main__": 
    from itertools import product

    k = [10,20,30,50,60,80]
    pc = [10,15,20,25,30]
    grid = product(k, pc)

    start_ixs = pd.read_csv(os.path.join(os.getcwd(), 'rw_starting_barcodes.csv'), header=0, index_col=0, sep=",").values.flatten()
    adata_path = os.path.join(os.getcwd(), 'data_folder', 'scRNA')

    cell_type = pd.read_csv(os.path.join(os.getcwd(), 'cell_annotations.tsv'), sep='\t', header = 0, index_col=0)
    cell_type_key = cell_type.columns[0]

    for k, pc in grid:
        sim = Simulator(k=k, pc=pc, path=adata_path, save = False)
        sim.preprocessing(cell_types=cell_type, cell_type_key=cell_type_key)
        sim.compute_transition_matrix(velocity_key="velocity", perform_analysis=True, similarity="correlation",
                                      key="connectivities", save_composition=False, softmax_scale=None)
        sim.cell_rank(start_ixs=start_ixs, plot=True, save_fate=True)
        del sim

