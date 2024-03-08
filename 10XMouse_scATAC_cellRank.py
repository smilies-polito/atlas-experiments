import os
import argparse
import pandas as pd
import numpy as np
import scanpy as sc
import cellrank as cr

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

res_atac = 1.0 #Kept constant

starting_cells = pd.read_csv(os.path.join(CWD, 'starting_cells.csv'), sep=',', index_col=0, header=0).values.flatten()

atac_path = os.path.join(ADATA_FOLDER, f'10xMouse_{args.k_atac}K{args.lsi}LSI{res_atac}res.h5ad')
atac = sc.read_h5ad(atac_path)

rna_path = os.path.join(ADATA_FOLDER, f'10xMouse_{args.k_rna}K{args.pc}PC{args.res_rna}res.h5ad')
rna = sc.read_h5ad(rna_path)

# Gets velocity Dataframe
velocities = pd.DataFrame(rna.layers['velocity'], index=rna.obs_names, columns =rna.var_names)

# Compute transition matrix
transitionMatrix = TransitionMatrix(adata=atac, velocities=velocities, promoter_key='gene')
transitionMatrix.compute_transition_matrix(similarity='correlation', key='distances')
atac.obsp['matrix'] = transitionMatrix.transition_matrix

# Compute random walks
kernel = PrecomputedKernel(object = atac, obsp_key = 'matrix')

title = f'Random Walk K={args.k_atac} LSI={args.lsi} res={res_atac}'
path = os.path.join(os.getcwd(), 'figures', f'random_walk_{args.k_atac}K{args.lsi}LSI{res_atac}res.png')
kernel.plot_random_walks(n_sims=200, seed=seed, start_ixs = starting_cells, title=title, save = path)

# Schur decomposition
g = cr.estimators.GPCCA(kernel)
g.compute_schur()
title = f'Schur decomposition K={args.k_atac} LSI={args.lsi} res={res_atac}' 
path = f'schurDecomposition_{args.k_atac}K{args.lsi}LSI{res_atac}res.png'
g.plot_spectrum(title=title, save=path, real_only=True)

# Macrostates
eigenvalues = g.eigendecomposition['D']
for idx, eig in enumerate(eigenvalues):
    #At least two macrostates otherwise no sense, at most 12 macrostates to limit computations (should be enough) and considering only real values
    if(eig.imag == 0) and idx > 0 and idx < 12: 
        ns = idx+1
        g.compute_macrostates(n_states=ns, cluster_key='louvain')
        tile = f'Macrostate ({ns}) Composition K={args.k_atac} LSI={args.lsi} res={res_atac}'
        path = f'macrostateComposition{ns}_{args.k_atac}K{args.lsi}LSI{res_atac}res.png'
        g.plot_macrostate_composition(key='louvain', show=False, save = path, title=title)
        title = f'Coarse Transition Matrix ({ns}) K={args.k_atac} LSI={args.lsi} res={res_atac}'
        path = f'coarseT{ns}_{args.k_atac}K{args.lsi}LSI{res_atac}res.png'
        g.plot_coarse_T(annotate=True, save=path, title=title)
        g.predict_initial_states()
        g.predict_terminal_states(allow_overlap=True)
        title = f'Initial states ({ns}) K={args.k_atac} LSI={args.lsi} res={res_atac}'
        path = f'initial{ns}_{args.k_atac}K{args.lsi}LSI{res_atac}res.png'
        g.plot_macrostates(which="initial", legend_loc="right", s=100, show=False, save=path, title=title)
        path = f'terminal{ns}_{args.k_atac}K{args.lsi}LSI{res_atac}res.png'
        title = f'Terminal states ({ns}) K={args.k_atac} LSI={args.lsi} res={res_atac}'        
        g.plot_macrostates(which="terminal", legend_loc="right", s=100, show=False, save=path, title=title)
        g.compute_fate_probabilities()
        path = f'fateProb{ns}_{args.k_atac}K{args.lsi}LSI{res_atac}res.png'
        title = f'Fate Probabilities ({ns}) K={args.k_atac} LSI={args.lsi} res={res_atac}'
        g.plot_fate_probabilities(same_plot=True, save = path, title=title)

