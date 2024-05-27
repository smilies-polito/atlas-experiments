import os
import numpy as np
import pandas as pd
import scanpy as sc 
import scvelo as scv
import cellrank as cr
import multivelo as mv

from anndata import AnnData
from matrix_analysis import MatrixAnalyser
from utils import _add_cell_type, _check_conjugate, _compute_macrostates, _plot_quality_gpcca, _save_qualities_gpcca, _save_probabilities, _check_macrostate_quality
from transitionMatrix import MultiVeloTM
from typing import Sequence, Optional, Literal

class MultiVelo:
    """
    Class implementing the pipeline for multi omics data and computes veloicities using Multivelo. 

    Attributes
    ----------
    _adata_path : str
        path where h5ad files are stored and saved 
    _k : int
        dimensionality of the scATAC-seq neighborhood
    _lsi : int
        number of dimensions for SVD
    _pc: int
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
    def __init__(self, path: str, lsi:int=10, pc:int=20, k:int=30, save:bool=False, atac: Optional[AnnData]= None):
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
        atac : anndata.AnnData, optional
            scATAC-seq dataset (used to save reading time)

        """
        self._adata_path = path
        self._k = k
        self._pc = pc
        self._lsi = lsi

        if atac is None:
            # reading 
            self._atac = sc.read_10x_mtx(os.path.join(self._adata_path, "filtered_feature_bc_matrix"), var_names="gene_symbols", gex_only=False)
            self._atac = self._atac[:, self._atac.var['feature_type']=="Peaks"]
        else:
            self._atac = atac

        self._rna = sc.read_h5ad(os.path.join(self._adata_path, f"RNA_{self._k}K{self._pc}PC{self._lsi}LSI.h5ad"))
        self._seed = 52
        self._save= save
        self._quality_dict={}



    def preprocessing(self, cell_types:Optional[pd.DataFrame], cell_type_key:str, peak_annotation_path:str, feature_linkage_path:str,
                       nn_idx:np.ndarray, nn_dist:np.ndarray):
        """

        Preprocessing: rna preprocessing using scanpy + smoothing genes aggregated by peaks by neighbors + aggregate peaks 
        + comphte velocities using MultiVelo.

        Parameters
        -----------
        cell_types: pandas.DataFrame, optional
            containing cell clusters annotaitons to be added to scATAC-seq AnnData (default is None)   
        cell_type_key: str 
            key in _atac.obs.columns where celltype annotations are stored (default is "cell_type"). 
        peak_annotation_path : str
            path to the peak annotation file from CellRanger ARC 2.0.0 used by Multivelo to aggregate peaks.  
        peak_annotation_path : str
            path to the feature linkage file from CellRanger ARC 2.0.0 used by Multivelo to aggregate peaks. 
        nn_idx : np.ndarray
            array containing indices from neighbors search in Seurat and used to impute scATAC-seq values.
        nn_dist : np.ndarray
            array containing distances from neighbors search in Seurat. and used to impute scATAC-seq values. 

        Updates 
        -------
        _cell_type_key : str
            key in _atac.obs.columns where celltype annotations are stored (default is "cell_type").  
        _rna : anndata.AnnData
            scRNA-seq embedding for RNA-velocities.    

        Updates
        --------
        self._adata : anndata.AnnData 
            stores the results of MultiVelo computations 
                    
        """

        if cell_types is not None:
            _add_cell_type(self._rna, cell_types)

        if cell_type_key not in self._rna.obs.columns: 
            raise ValueError(f"{cell_type_key} not in rna.obs")
        
        self._cell_type_key = cell_type_key

        self._rna.uns.pop('neighbors', None)

        # Aggiungo serve per velocity computations
        self._rna.var['highly_variable'] = np.ones(len(self._rna.var_names), dtype=bool)

        sc.pp.neighbors(self._rna, n_neighbors=self._k, n_pcs=self._pc, random_state = self._seed)
        scv.pp.moments(self._rna, n_pcs=self._pc, n_neighbors=self._k)

        mv.tfidf_norm(self._atac)
        self._atac = mv.aggregate_peaks_10x(self._atac, peak_annotation_path, feature_linkage_path)

        # Subset for shared genes and ordering :)
        shared_barcodes = pd.Index(np.intersect1d(self._atac.obs_names, self._rna.obs_names))
        shared_genes = pd.Index(np.intersect1d(self._atac.var_names, self._rna.var_names))
        self._rna = self._rna[shared_barcodes, shared_genes]
        self._atac= self._atac[shared_barcodes, shared_genes]
 
        mv.knn_smooth_chrom(self._atac, nn_idx, nn_dist)

        self._adata = mv.recover_dynamics_chrom(self._rna, self._atac, parallel=True, n_jobs=10)
        mv.velocity_graph(self._adata)
        mv.latent_time(self._adata)

        title = f"Multivelo model K={self._k}  LSI={self._lsi} PC={self._pc}"
        path= f"multiveloModel_{self._k}K{self._lsi}LSI{self._pc}PC.png"
        mv.velocity_embedding_stream(self._adata, basis='umap', color=[self._cell_type_key], title=title, save=path, show=False)

        title = f"Multivelo latent time K={self._k}  LSI={self._lsi} PC={self._pc}"
        path= f"multiveloLatentTime_{self._k}K{self._lsi}LSI{self._pc}PC.png"
        scv.pl.scatter(self._adata, color='latent_time', color_map='gnuplot', size=80, title=title, save=path)

        if self._save:
            path = os.path.join(self._adata_path, f'RNA_{self._k}K{self._pc}PC{self._lsi}LSI.h5ad')
            self._rna.write_h5ad(path)
            path = os.path.join(self._adata_path, f'ATAC_{self._k}K{self._pc}PC{self._lsi}LSI.h5ad')
            self._atac.write_h5ad(path)
            path = os.path.join(self._adata_path, f'ADATA_{self._k}K{self._pc}PC{self._lsi}LSI.h5ad')
            self._adata.write_h5ad(path)


    def compute_transition_matrix(self, perform_analysis:bool=True, 
                                   atac_key: str="ATAC", rna_key: Optional[str] = None, 
                                   rna_velo_key: str = "velo_s", atac_velo_key: str="velo_chrom", 
                                   similarity: Literal['cosine', "correlation", "dot"] = "correlation", 
                                   key: str = 'connectivities', softmax_scale: Optional[float] = None
            ):
        """
        Computes transition matrix for CellRank.
        
        Parameters
        ----------
        rna_key: str, optional 
            key in adata.layers containing scRNA-seq data (default is None). If None, adata.X is used for scRNA-seq values.
        atac_key: str
            key in adata.layers containing scATAC-seq data (default is Multivelo "ATAC")
        atac_velo_key: str
            key in adata.layers containing chromatin velocities (default is Multivelo "velo_chrom")
        rna_velo_key: str
            key in adata.layers containing RNA velocities (default is Multivelo "velo_s")
        perform_analysis: bool
            whether to perform analysis of the graph associated with the transition matrix (default is False)
        similarity: str
            indicates similarity metric for the transition matrix computations. Valid options are "cosine", "correlation", "dot" (default is "correlation")
        key: str
            key in _atac.obsp where neighborhood matrix is stored (default is "connectivities")
        softmax_scale: float, optional
            softmax_scale value for softmax computation (default is None)

        Updates
        --------
        _adata.obsp['matrix']: scipy.sparse.csr_matrix
            transition matrix.

        """
        transitionMatrix = MultiVeloTM(adata=self._adata, atac_key = atac_key, rna_key=rna_key,
                                        atac_velo_key = atac_velo_key, rna_velo_key=rna_velo_key, softmax_scale=softmax_scale)
        transitionMatrix.compute_transition_matrix(similarity=similarity, key=key)
        self._adata.obsp['matrix']= transitionMatrix.transition_matrix

        if perform_analysis:
            analyser = MatrixAnalyser(matrix = transitionMatrix.transition_matrix, adata=self._adata, cluster_key=self._cell_type_key, k=self._k, pc=self._pc, lsi=self._lsi)
            analyser._topology_analysis()
            analyser._condensation_graph()
            analyser._find_invariant()
            save_path = os.path.join(os.getcwd(), "multivelo_params.csv")
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

        kernel= cr.kernels.PrecomputedKernel(object=self._adata, obsp_key='matrix')    

        if plot:
            title = f'Random Walk K={self._k} LSI={self._lsi} PC={self._pc}'
            path = f'random_walk_{self._k}K{self._lsi}LSI{self._pc}PC.png'
            kernel.plot_random_walks(start_ixs = start_ixs , n_sims=200, seed=self._seed, title=title, save=path)

        g = cr.estimators.GPCCA(kernel)
        g.compute_schur()
        self._gpcca = g

        if plot:
            title = f'Schur decomposition K={self._k} LSI={self._lsi} PC={self._pc}' 
            path = f'schurDecomposition_{self._k}K{self._lsi}LSI{self._pc}PC.png'
            g.plot_spectrum(title=title, save=path, real_only=True)

        eigenvalues = g.eigendecomposition['D']
        idx = 1

        while idx<10:
            eig = eigenvalues[idx]
            idx = _check_conjugate(idx, eig)
            try:
                self._quality_dict[idx] = _compute_macrostates(n_states=idx, gpcca=self._gpcca, plot=plot, save_fate=save_fate, barcodes=self._adata.obs_names, k=self._k,
                                                               lsi=self._lsi, pc=self._pc, cell_type_key=self._cell_type_key)
            except ValueError as e:
                print(e)   

        df = pd.DataFrame(self._quality_dict, index = ["spectral_gap", 'minChi', 'crispness']).T
        
        if plot: 
            _plot_quality_gpcca(df, k=self._k, pc=self._pc, lsi=self._lsi)
        
        _save_qualities_gpcca(df, k=self._k, pc=self._pc, lsi=self._lsi)
                     


if __name__ == "__main__": 
    from itertools import product

    k = [80]
    pc = [10,15,20,25,30]
    lsi = [30]
    grid = product(k, pc, lsi)

    adata_path = os.path.join(os.getcwd(), 'data_folder', 'multiomics')
    peak_annotation_path = os.path.join(os.getcwd(), 'peak_annotation.tsv')
    feature_linkage_path = os.path.join(os.getcwd(), 'feature_linkage.bedpe')
    cell_type = pd.read_csv(os.path.join(os.getcwd(), 'cell_annotations.tsv'), sep='\t', header = 0, index_col=0)
    cell_type_key = cell_type.columns[0]

    start_ixs = pd.read_csv(os.path.join(os.getcwd(), 'rw_starting_barcodes.csv'), header=0, index_col=0, sep=",").values.flatten()

    atac= sc.read_10x_mtx(os.path.join(adata_path, 'filtered_feature_bc_matrix'), var_names="gene_symbols", cache=True, gex_only=False)
    atac = atac[:,atac.var['feature_types'] == "Peaks"]

    for k, pc, lsi in grid:
        nn_idx = np.loadtxt(os.path.join(adata_path, f"idx_{k}K{lsi}LSI{pc}PC.txt"), delimiter=',')
        nn_dist = np.loadtxt(os.path.join(adata_path, f"dist_{k}K{lsi}LSI{pc}PC.txt"), delimiter=',')
        sim = MultiVelo(path=adata_path, lsi=lsi, pc=pc, k=k, save=False, atac=atac)
        sim.preprocessing(cell_types=cell_type, cell_type_key=cell_type_key, peak_annotation_path=peak_annotation_path,
                          feature_linkage_path=feature_linkage_path, nn_dist=nn_dist, nn_idx=nn_idx)
        sim.compute_transition_matrix(perform_analysis=True, atac_key = "ATAC", rna_velo_key="velo_s", atac_velo_key="velo_chrom",
                                      similarity="correlation", key="connectivities", softmax_scale=None)
        sim.cell_rank(start_ixs = start_ixs, plot=True, save_fate=True)
        del sim
