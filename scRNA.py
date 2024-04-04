import os
import numpy as np
import pandas as pd
import scanpy as sc
import cellrank as cr
import scvelo as scv
import matplotlib.pyplot as plt

from utils import _check_conjugate, _check_macrostate_quality, _plot_quality_gpcca
from matrix_analysis import MatrixAnalyser
from typing import Sequence

class scRNA():
    def __init__(self, k:int = 10, pc:int = 10, res: float = 1.0, path: str = 'adata_folder', save: bool = False ):
        """
        params: 
            - path: where results are stored.
            - k: dimensionality oh neighborhood.
            - pc: number of components in PCA.
            - res: resolution of Louvain clustering algorithm. 
            - save: boolean if to save the adata.
        """
        self._adata_path = os.path.join(os.getcwd(), path)
        self._seed = 52
        self._k = k
        self._pc = pc
        self._res = res
        self._save = save
        self._quality_dict = {}

        if not os.path.exists(self._adata_path):
            os.makedirs(self._adata_path)


        self._starting_cells = pd.read_csv(os.path.join(os.getcwd(), 'rw_starting_barcodes.csv'), 
                                           header=0, index_col = 0).to_numpy().flatten()
        
        path = os.path.join(os.getcwd(), '10X_multiome_mouse_brain.loom')
        adata = sc.read_loom(path)
        adata.var_names_make_unique()
        adata.obs_names = [obs.split(':')[1][:-1] + '-1' for obs in adata.obs_names]
        self._adata = adata


                                
    def _basic_preprocessing(self, plot: bool = False):
        """
            Basic preprocessing: loading from loom + qc + qc plots + filtering
            params:
                - plot: if to save figures for preprocessing
        """

        self._adata.var['mt'] = self._adata.var_names.str.startswith('mt-')
        sc.pp.calculate_qc_metrics(self._adata, qc_vars=['mt'],log1p=False, percent_top=None, inplace=True)

        if plot:
            sc.pl.violin(self._adata, ['n_genes_by_counts', 'total_counts',  'pct_counts_mt'], jitter = 0.04, 
                        multi_panel= True, save = "_scRNA.png", show=False)
            sc.pl.scatter(self._adata, x='n_genes_by_counts', y='total_counts', save = "_scRNA_totalCounts.png", show=False)
            sc.pl.scatter(self._adata, x='n_genes_by_counts', y='pct_counts_mt', save = "_scRNA_mitoPercent.png", show=False)

        lower, upper = np.percentile(self._adata.obs.n_genes_by_counts, [2,98]).astype(int)
        print(f'Retaining cells with {lower} < n_genes_by_counts < {upper}')
        sc.pp.filter_cells(self._adata, min_genes=lower)
        sc.pp.filter_cells(self._adata, max_genes=upper)
        scv.pp.filter_and_normalize(self._adata, min_shared_counts=10, n_top_genes=2000)
        sc.pp.highly_variable_genes(self._adata, n_top_genes=2000)

        preprocessing_path = os.path.join(self._adata_path, '10xMouse_loom_preprocessed.h5ad')
        self._adata.write_h5ad(preprocessing_path)   


    def _save_pca_figures(self, adata):
        """
            Save elbow plot and cumulative variance of PCA
            params:
                - adata: annotated data.
        """
        sc.pl.pca_variance_ratio(adata, n_pcs=50, save='_scRNA.png', show=False)

        # Explained variance plot
        cumulative_variance_ratio = np.cumsum(adata.uns['pca']['variance_ratio'])
        fig, ax = plt.subplots(figsize=(7, 6))
        plt.scatter(range(1,51), cumulative_variance_ratio,)
        # Plotting variance explained by first 10,15,20,25,30 components according to elbow plot 
        plt.axhline(cumulative_variance_ratio[9], color='silver', linestyle='dashed')  
        plt.axhline(cumulative_variance_ratio[14], color='silver', linestyle='dashed') 
        plt.axhline(cumulative_variance_ratio[19], color='silver', linestyle='dashed') 
        plt.axhline(cumulative_variance_ratio[24], color='silver', linestyle='dashed') 
        plt.axhline(cumulative_variance_ratio[29], color='silver', linestyle='dashed') 
        ax.set_xticks(range(0, 100, 10))
        plt.savefig(os.path.join(os.getcwd(), 'figures', 'scRNA_explainedVar.png'))


    def _load_preprocessed_adata(self, path: str):
        """
            Function loads h5ad file for annotated dataset.
            params:
                - path: path of the h5ad file to be loaded.
        """
        self._adata = sc.read_h5ad(path)


    def _subset_cells(self, barcodes: Sequence):
        """
            Function that subset adata according to barcodes.
            params:
                -barcodes: set of cells to keep 
        """
        self._adata = self._adata[self._adata.obs_names.isin(barcodes)]


    def _rna_preprocessing(self, plot: bool= False):
        """
            Terminates preprocessing pca + neighborhood + umap + louvain
            params:
                - plot: if to plot PCA elbow plot and explained variance. 
        """
                # PCA
        scv.pp.pca(self._adata, n_comps=50, use_highly_variable=True)
        if plot:
            self._save_pca_figures(self._adata)

        sc.pp.neighbors(self._adata, n_neighbors=self._k, n_pcs=self._pc, random_state = self._seed)
        sc.tl.louvain(self._adata, resolution=self._res, random_state=self._seed)
        title = f'K={self._k} PC={self._pc} res={self._res}'
        path =  f'_scRNA_{self._k}K{self._pc}PC{self._res}res.png'
        sc.tl.umap(self._adata, random_state=self._seed)
        sc.pl.umap(self._adata, color=['louvain'], title=title, show=False, save=path)

        if self._save:
            path = os.path.join(self._adata_path, f"10xMouse_{self._k}K{self._pc}PC{self._res}res.h5ad")
            self._adata.write_h5ad(path)


    def _rna_velocities(self):
        """
            Computes velocities using dynamical model scVELO.
        """
        scv.pp.moments(self._adata, n_neighbors=self._k, n_pcs=self._pc)
        scv.tl.recover_dynamics(self._adata, n_jobs=-1)
        scv.tl.velocity(self._adata, mode='dynamical')
        scv.tl.velocity_graph(self._adata, n_jobs=-1)

        title = f"Dynamical model K={self._k} PC={self._pc} res={self._res}"
        path= f"dynamicalModel_{self._k}K{self._pc}PC{self._res}res.png"
        scv.pl.velocity_embedding_stream(self._adata, basis='umap', color=['louvain'], title=title, save=path, show=False)

        #Compute latent time (cell's internal clock)
        title = f"Cell specific latent time K={self._k} PC={self._pc} res={self._res}"
        path= f"latentTime_{self._k}K{self._pc}PC{self._res}res.png"
        scv.tl.latent_time(self._adata)
        scv.pl.scatter(self._adata, color='latent_time', color_map='gnuplot', size=40, save=path, title=title, show=False)

        if self._save:
            path = os.path.join(self._adata_path, f"10xMouse_{self._k}K{self._pc}PC{self._res}res.h5ad")
            self._adata.write_h5ad(path)


    def _cell_rank(self, perform_analysis : bool = True):
        """
            CellRank simulation.
            params: 
                - perform_analysis: boolean indicating is to analyse the transition matrix. 
        """
        vk= cr.kernels.VelocityKernel(self._adata)
        vk.compute_transition_matrix()

        if perform_analysis:
            analyser = MatrixAnalyser(matrix = vk.transition_matrix, adata=self._adata, cluster_key='louvain', k=self._k, pc=self._pc, res=self._res)
            analyser._topology_analysis()
            print(f"Graph aperiodic {analyser._aperiodic} and strogly_connected {analyser._strongly_connected}")
            analyser._condensation_graph()
            analyser._find_invariant()

        
        title = f'Random Walk K={self._k} PC={self._pc} res={self._res}'
        path = f'random_walk_{self._k}K{self._pc}PC{self._res}res.png'
        vk.plot_random_walks(start_ixs = self._starting_cells , n_sims=200, seed=self._seed, title=title, save=path)

        g = cr.estimators.GPCCA(vk)
        g.compute_schur()
        self._gpcca = g
        title = f'Schur decomposition K={self._k} PC={self._pc} res={self._res}' 
        path = f'schurDecomposition_{self._k}K{self._pc}PC{self._res}res.png'
        g.plot_spectrum(title=title, save=path)

        eigenvalues = g.eigendecomposition['D']
        idx = 1

        while idx<12:
            eig = eigenvalues[idx]
            idx = _check_conjugate(idx, eig)
            try:
                self._quality_dict[idx] = self._compute_macrostates(idx)
            except ValueError as e:
                print(e)

        df = pd.DataFrame(self._quality_dict, index = ["spectral_gap", 'minChi', 'crispness']).T
        _plot_quality_gpcca(df, k=self._k, pc=self._pc, res=self._res)


    def _compute_macrostates(self, idx: int):
            self._gpcca.compute_macrostates(n_states=idx, cluster_key='louvain')

            title = f'Macrostate ({idx}) Composition K={self._k} PC={self._pc} res={self._res}'
            path = f'macrostateComposition{idx}_{self._k}K{self._pc}PC{self._res}res.png'
            self._gpcca.plot_macrostate_composition(key='louvain', show=False, save = path, title=title)

            title = f'Coarse Transition Matrix ({idx}) K={self._k} PC={self._pc} res={self._res}'
            path = f'coarseT{idx}_{self._k}K{self._pc}PC{self._res}res.png'
            self._gpcca.plot_coarse_T(annotate=True, save=path, title=title)

            self._gpcca.predict_initial_states()
            self._gpcca.predict_terminal_states(allow_overlap=True)
            title = f'Initial states ({idx}) K={self._k} PC={self._pc} res={self._res}'
            path = f'initial{idx}_{self._k}K{self._pc}PC{self._res}res.png'
            self._gpcca.plot_macrostates(which="initial", legend_loc="right", s=100, show=False, save=path, title=title)

            path = f'terminal{idx}_{self._k}K{self._pc}PC{self._res}res.png'
            title = f'Terminal states ({idx}) K={self._k} PC={self._pc} res={self._res}'        
            self._gpcca.plot_macrostates(which="terminal", legend_loc="right", s=100, show=False, save=path, title=title)
            self._gpcca.compute_fate_probabilities(tol=1e-10, use_petsc=True, preconditioner="ilu")

            path = f'fateProb{idx}_{self._k}K{self._pc}PC{self._res}res.png'
            title = f'Fate Probabilities ({idx}) K={self._k} PC={self._pc} res={self._res}'
            self._gpcca.plot_fate_probabilities(same_plot=True, save = path, title=title)
            
            return _check_macrostate_quality(self._gpcca, idx)



if __name__ == "__main__": 
    from itertools import product

    k = [10, 20, 30, 50, 60, 80]
    pc = [10, 15, 20, 25, 30]
    res = [0.7, 1.0, 1.4]
    grid = product(k, pc, res)

    multivelo_barcodes = pd.read_csv(os.path.join(os.getcwd(), 'cell_annotations.tsv'), sep='\t',
                                    index_col = 0, header=0)
    multivelo_barcodes = multivelo_barcodes[~multivelo_barcodes.celltype.isin(['Interneurons1', 'Interneurons2', 'Interneurons3'])]

    sim = scRNA(k=30, pc=20, res=1.0, save = True)
    sim._subset_cells(multivelo_barcodes.index)
    sim._basic_preprocessing(plot=True)
    sim._rna_preprocessing(plot=True)


    for k, pc, res in grid:
        sim = scRNA(k=k, pc=pc, res = res, save = False)
        sim._load_preprocessed_adata(os.path.join(os.getcwd(),'adata_folder','10xMouse_loom_preprocessed.h5ad'))
        sim._rna_preprocessing(plot=False)
        sim._rna_velocities()
        sim._cell_rank()  
