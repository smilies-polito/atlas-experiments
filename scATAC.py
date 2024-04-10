import os
import numpy as np
import pandas as pd
import scanpy as sc
import cellrank as cr
import scvelo as scv
import matplotlib.pyplot as plt

from utils import _check_conjugate, _check_macrostate_quality, _plot_quality_gpcca
from matrix_analysis import MatrixAnalyser
from typing import Sequence, Literal

class scATAC():
    def __init__(self, k:int = 10, lsi:int = 10, res: float = 1.0, save: bool = False ):
        """
        params: 
            - path: where results are stored.
            - k: dimensionality oh neighborhood.
            - lsi: number of components in LSI.
            - res: resolution of Smart Local Moving clustering algorithm. 
            - save: boolean if to save the adata.
        """
        self._adata_path = os.path.join(os.getcwd(), path)
        self._seed = 52
        self._k = k
        self._lsi = lsi
        self._res = res
        self._save = save
        self._quality_dict = {}

        if not os.path.exists(self._adata_path):
            os.makedirs(self._adata_path)


        self._starting_cells = pd.read_csv(os.path.join(os.getcwd(), 'rw_starting_barcodes.csv'), 
                                           header=0, index_col = 0).to_numpy().flatten()
        
        #Load atac and rna adata
        path = os.path.join(os.getcwd(), 'signac_folder', f'scATAC_{self.k}K{self.lsi}LSI.h5ad')
        self._atac = sc.read_h5ad(path)
        self._rna = sc.read_h5ad(os.path.join(os.getcwd(), 'adata_folder', '10XMouse_30K20PC1.0res.h5ad'))

        # Load annotations from 10XGenomics
        # self._annotations = pd.read_csv(os.path.join(os.getcwd(), 'peak_annotation.tsv'), sep='\t', header = 0)
        # self._annotations = self._annotations[self._annotations.peak_type == 'promoter']
        # self._annotations['peak'] = self._annotations[['chrom', 'start', 'end']].apply(lambda row: '-'.join(row.values.astype(str)), axis=1)


    def _merge_rna(self):
        """
            Function that performs merge between atac and rna:
                - Subset for common barcodes
                - Add Louvain clusters to atac
        """
        # Subset for common barcodes
        intersection = set(self._atac.obs_names) - set(self._rna.obs_names)
        self._subset_cells(adata='atac', barcodes=intersection)
        self._subset_cells(adata='rna', barcodes=intersection)

        # Add Louvain clusters 
        self._atac.obs = self._atac.obs.merge(self._rna.obs.louvain, how='left', left_index = True, right_index=True)
        self._atac.obs.seurat_clusters = self._atac.obs.seurat_clusters.astype("category")


    def _add_annotations(self, annotations: pd.DataFrame, gene:str, peak: str, type: str, promoter_key: str = 'promoter'):
        """
            Function that adds peak annotations to the atac annotated data.
            params:
                - annotations: pandas dataframe with peaks, genes linking each peak to a gene and peak_type (promoter, distal, etc.).
                - gene: column name in annotations identifying the gene column. 
                - type: column name in annotations identifying the peak_type. 
                - peak: column name in annotations identifying the peak. 
                - promoter_key: string identifying the way promoters are identified in column 'type'. Default is "promoter".
        """
        # Filter for promoter peaks + filter for values present in atac and rna and merge to a dataset
        annotations = annotations[annotations[type]==promoter_key]
        annotations = annotations[(annotations[gene].isin(self._rna.var_names)) & (annotations[peak].isin(self._atac.var_names))]
        merging = self._atac.var.merge(annotations, how='left', left_index =True, right_on='peak')

        # Check if there are peaks which are not promoter to any gene and delete then from adata
        peaks_to_keep = merging[~merging[gene].isna()][peak].values
        self._subset_peaks(peaks_to_keep)

        # Check if there are peaks which are promoter to multiple genes
        peaks_multiple = merging.groupby(by=peak).size()
        peaks_multiple = peaks_multiple[peaks_multiple>1] #series con index il peak e value il count
        if peaks_multiple.shape[0] > 0:
            self._adjust_for_multiple_genes(peaks_multiple) 
        


    def _adjust_for_multiple_genes(self, peaks: Sequence):
        """
            Function that adjusts the ATAC matrix repeating the peak data for the number of genes it is promoter to.
            params:
                - peaks: list of peaks that are promoter to more than one gene
        """
        # TO DO 
        pass 

    def _subset_peaks(self, peaks: Sequence):
        """
            Function that subset adata according to peaks.
            params:
                - peaks: set of peaks to keep 
        """
        self._atac = self._atac[:, self._atac.var_names.isin(peaks)]


    def _subset_cells(self, adata: Literal['atac', 'rna'], barcodes: Sequence):
        """
            Function that subset adata according to barcodes.
            params:
                -adata: name of the adata to subset, either "atac" or "rna"
                -barcodes: set of cells to keep 
        """
        if adata == 'atac':
            self._atac = self._atac[self._atac.obs_names.isin(barcodes)]
        else:
            self._rna = self._rna[self._rna.obs_names.isin(barcodes)]
    



    def _cell_rank(self, perform_analysis : bool = True, plot: bool = False):
        """
            CellRank simulation.
            params: 
                - perform_analysis: boolean indicating is to analyse the transition matrix. 
                - plot: if to plot and save schur decomposition, random walk and results for GPCCA
        """
#         vk= cr.kernels.VelocityKernel(self._adata)
#         vk.compute_transition_matrix()

#         if perform_analysis:
#             analyser = MatrixAnalyser(matrix = vk.transition_matrix, adata=self._adata, cluster_key='louvain', k=self._k, pc=self._pc, res=self._res)
#             analyser._topology_analysis()
#             print(f"Graph aperiodic {analyser._aperiodic} and strogly_connected {analyser._strongly_connected}")
#             analyser._condensation_graph()
#             analyser._find_invariant()

#         if plot:
#             title = f'Random Walk K={self._k} PC={self._pc} res={self._res}'
#             path = f'random_walk_{self._k}K{self._pc}PC{self._res}res.png'
#             vk.plot_random_walks(start_ixs = self._starting_cells , n_sims=200, seed=self._seed, title=title, save=path)

#         g = cr.estimators.GPCCA(vk)
#         g.compute_schur()
#         self._gpcca = g

#         if plot:
#             title = f'Schur decomposition K={self._k} PC={self._pc} res={self._res}' 
#             path = f'schurDecomposition_{self._k}K{self._pc}PC{self._res}res.png'
#             g.plot_spectrum(title=title, save=path)

#         eigenvalues = g.eigendecomposition['D']
#         idx = 1

#         while idx<12:
#             eig = eigenvalues[idx]
#             idx = _check_conjugate(idx, eig)
#             try:
#                 self._quality_dict[idx] = self._compute_macrostates(idx, plot)
#             except ValueError as e:
#                 print(e)

#         df = pd.DataFrame(self._quality_dict, index = ["spectral_gap", 'minChi', 'crispness']).T
        
#         if plot: 
#             _plot_quality_gpcca(df, k=self._k, pc=self._pc, res=self._res)


#     def _compute_macrostates(self, idx: int, plot:bool = False):
#             self._gpcca.compute_macrostates(n_states=idx, cluster_key='louvain')
#             if plot:
#                 title = f'Macrostate ({idx}) Composition K={self._k} PC={self._pc} res={self._res}'
#                 path = f'macrostateComposition{idx}_{self._k}K{self._pc}PC{self._res}res.png'
#                 self._gpcca.plot_macrostate_composition(key='louvain', show=False, save = path, title=title)

#                 title = f'Coarse Transition Matrix ({idx}) K={self._k} PC={self._pc} res={self._res}'
#                 path = f'coarseT{idx}_{self._k}K{self._pc}PC{self._res}res.png'
#                 self._gpcca.plot_coarse_T(annotate=True, save=path, title=title)

#                 self._gpcca.predict_initial_states()
#                 self._gpcca.predict_terminal_states(allow_overlap=True)
#                 title = f'Initial states ({idx}) K={self._k} PC={self._pc} res={self._res}'
#                 path = f'initial{idx}_{self._k}K{self._pc}PC{self._res}res.png'
#                 self._gpcca.plot_macrostates(which="initial", legend_loc="right", s=100, show=False, save=path, title=title)

#                 path = f'terminal{idx}_{self._k}K{self._pc}PC{self._res}res.png'
#                 title = f'Terminal states ({idx}) K={self._k} PC={self._pc} res={self._res}'        
#                 self._gpcca.plot_macrostates(which="terminal", legend_loc="right", s=100, show=False, save=path, title=title)
            
#             self._gpcca.compute_fate_probabilities(tol=1e-10, use_petsc=True, preconditioner="ilu")

#             if plot: 
#                 path = f'fateProb{idx}_{self._k}K{self._pc}PC{self._res}res.png'
#                 title = f'Fate Probabilities ({idx}) K={self._k} PC={self._pc} res={self._res}'
#                 self._gpcca.plot_fate_probabilities(same_plot=True, save = path, title=title)
            
#             return _check_macrostate_quality(self._gpcca, idx)






# if __name__ == "__main__": 
#     from itertools import product

#     k = [80]
#     pc = [10, 15, 20, 25, 30]
#     res = [0.7, 1.0, 1.4]
#     grid = product(k, pc, res)

#     multivelo_barcodes = pd.read_csv(os.path.join(os.getcwd(), 'cell_annotations.tsv'), sep='\t',
#                                     index_col = 0, header=0)
#     multivelo_barcodes = multivelo_barcodes[~multivelo_barcodes.celltype.isin(['Interneurons1', 'Interneurons2', 'Interneurons3'])]

#     #sim = scRNA(k=30, pc=20, res=1.0, save = True)
#     #sim._subset_cells(multivelo_barcodes.index)
#     #sim._basic_preprocessing(plot=True)
#     #sim._rna_preprocessing(plot=True)


#     for k, pc, res in grid:
#         sim = scRNA(k=k, pc=pc, res = res, save = False)
#         sim._load_preprocessed_adata(os.path.join(os.getcwd(),'adata_folder','10xMouse_loom_preprocessed.h5ad'))
#         sim._rna_preprocessing(plot=False)
#         sim._rna_velocities()
#         sim._cell_rank()  
