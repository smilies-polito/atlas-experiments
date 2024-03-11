import os
import argparse
import pandas as pd
import numpy as np
import scanpy as sc
import scipy as sp
import cellrank as cr

from utils import _check_conjugate
from cellrank.kernels import PrecomputedKernel
from transitionMatrix import TransitionMatrix

seed = 52

CWD = os.getcwd()
ADATA_FOLDER = os.path.join(CWD, 'adata_folder')

parser = argparse.ArgumentParser()
parser.add_argument('-k_atac', type = int, help = 'Integer indicating the size of the scATAC-seq data neighborhood')
parser.add_argument('-k_rna', type = int, help = 'Integer indicating the size of the scRNA-seq data neighborhood')
parser.add_argument('-pc', type = int, help='Integer indicating the number of principal components for scRNA-seq data')
parser.add_argument('-lsi', type = int, help='Integer indicating the number of dimensions for scATAC-seq data')
parser.add_argument('-res_rna', type = float, help='Float indicating the number resolution for Louvain community detection algorithm for scRNA-seq data')
args = parser.parse_args()

starting_cells = pd.read_csv(os.path.join(CWD, 'starting_cells.csv'), sep=',', index_col=0, header=0).values.flatten()

atac_path = os.path.join(ADATA_FOLDER, f'10xMouse_{args.k_atac}K{args.lsi}LSI.h5ad')
atac = sc.read_h5ad(atac_path)

rna_path = os.path.join(ADATA_FOLDER, f'10xMouse_{args.k_rna}K{args.pc}PC{args.res_rna}res.h5ad')
rna = sc.read_h5ad(rna_path)

# Gets velocity Dataframe
velocities = pd.DataFrame(rna.layers['velocity'], index=rna.obs_names, columns =rna.var_names)

# Compute transition matrix
transitionMatrix = TransitionMatrix(adata=atac, velocities=velocities, promoter_key='gene')
transitionMatrix.compute_transition_matrix(similarity='cosine', key='distances')
atac.obsp['matrix'] = transitionMatrix.transition_matrix

# Compute random walks
kernel = PrecomputedKernel(object = atac, obsp_key = 'matrix')

title = f'Random Walk K={args.k_atac} LSI={args.lsi}'
path = os.path.join(os.getcwd(), 'figures', f'random_walk_{args.k_atac}K{args.lsi}.png')
kernel.plot_random_walks(n_sims=200, seed=seed, start_ixs = starting_cells, title=title, save = path)


# Schur decomposition
g = cr.estimators.GPCCA(kernel)
g.compute_schur()
title = f'Schur decomposition K={args.k_atac} LSI={args.lsi}' 
path = f'schurDecomposition_{args.k_atac}K{args.lsi}LSI.png'
g.plot_spectrum(title=title, save=path, real_only=True)

print(g.eigendecomposition)

# Macrostates
eigenvalues = g.eigendecomposition['D']
idx = 1

while idx < 10:
    # Mi aspetto 1 iniziale 4 terminali (almeno 5 stati) 
    eig = eigenvalues[idx]
    idx = _check_conjugate(idx, eig)
    g.compute_macrostates(n_states=idx, cluster_key='seurat_clusters')
    title = f'Macrostate ({idx}) Composition K={args.k_atac} LSI={args.lsi}'
    path = f'macrostateComposition{idx}_{args.k_atac}K{args.lsi}LSI.png'
    g.plot_macrostate_composition(key='seurat_clusters', show=False, save = path, title=title)
    title = f'Coarse Transition Matrix ({idx}) K={args.k_atac} LSI={args.lsi}'
    path = f'coarseT{idx}_{args.k_atac}K{args.lsi}LSI.png'
    g.plot_coarse_T(annotate=True, save=path, title=title)
    g.predict_initial_states()
    g.predict_terminal_states(allow_overlap=True)
    title = f'Initial states ({idx}) K={args.k_atac} LSI={args.lsi}'
    path = f'initial{idx}_{args.k_atac}K{args.lsi}LSI.png'
    g.plot_macrostates(which="initial", legend_loc="right", s=100, show=False, save=path, title=title)
    path = f'terminal{idx}_{args.k_atac}K{args.lsi}LSI.png'
    title = f'Terminal states ({idx}) K={args.k_atac} LSI={args.lsi}'        
    g.plot_macrostates(which="terminal", legend_loc="right", s=100, show=False, save=path, title=title)
    g.compute_fate_probabilities(tol = 1e-10, use_petsc=True, preconditioner='ilu')
    path = f'fateProb{idx}_{args.k_atac}K{args.lsi}LSI.png'
    title = f'Fate Probabilities ({idx}) K={args.k_atac} LSI={args.lsi}'
    g.plot_fate_probabilities(same_plot=True, save = path, title=title)

