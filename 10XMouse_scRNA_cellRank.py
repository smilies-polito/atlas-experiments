import os
import argparse
import pandas as pd
import scanpy as sc 
import cellrank as cr

from utils import _check_conjugate, _check_macrostate_quality

seed = 52

parser = argparse.ArgumentParser()
parser.add_argument('-k', type = int, help='Integer describing the size of the local neighborhood')
parser.add_argument('-pc', type = int, help = 'Integer describing the number of principal components')
parser.add_argument('-res', type=float, help='Float describing the resolution of the Louvain community detection algorithm')
args = parser.parse_args()

CWD = os.getcwd()
FIGURE_FOLDER = os.path.join(CWD, 'figures')
ADATA_FOLDER = os.path.join(CWD, 'adata_folder')
DATA_PATH = os.path.join(ADATA_FOLDER, f'10xMouse_{args.k}K{args.pc}PC{args.res}res.h5ad')

starting_cells = pd.read_csv('rw_starting_barcodes.csv', header=None, index_col = 0).to_numpy().flatten()

adata = sc.read_h5ad(DATA_PATH)

vk= cr.kernels.VelocityKernel(adata)
vk.compute_transition_matrix()

# Random walk
title = f'Random Walk K={args.k} PC={args.pc} res={args.res}'
path = f'random_walk_{args.k}K{args.pc}PC{args.res}res.png'
vk.plot_random_walks(start_ixs = starting_cells , n_sims=200, seed=seed, title=title, save=path)


#Compute macrostates
g = cr.estimators.GPCCA(vk)
g.compute_schur()
title = f'Schur decomposition K={args.k} PC={args.pc} res={args.res}' 
path = f'schurDecomposition_{args.k}K{args.pc}PC{args.res}res.png'
g.plot_spectrum(title=title, save=path)
 

eigenvalues = g.eigendecomposition['D']
idx = 1

quality_dict={}

while idx<len(eigenvalues):
    # Mi aspetto 1 iniziale 4 terminali (almeno 5 stati) 
    eig = eigenvalues[idx]
    idx = _check_conjugate(idx,eig) 
    g.compute_macrostates(n_states=idx, cluster_key='louvain')
    title = f'Macrostate ({idx}) Composition K={args.k} PC={args.pc} res={args.res}'
    path = f'macrostateComposition{idx}_{args.k}K{args.pc}PC{args.res}res.png'
    g.plot_macrostate_composition(key='louvain', show=False, save = path, title=title)
    title = f'Coarse Transition Matrix ({idx}) K={args.k} PC={args.pc} res={args.res}'
    path = f'coarseT{idx}_{args.k}K{args.pc}PC{args.res}res.png'
    g.plot_coarse_T(annotate=True, save=path, title=title)
    g.predict_initial_states()
    g.predict_terminal_states(allow_overlap=True)
    title = f'Initial states ({idx}) K={args.k} PC={args.pc} res={args.res}'
    path = f'initial{idx}_{args.k}K{args.pc}PC{args.res}res.png'
    g.plot_macrostates(which="initial", legend_loc="right", s=100, show=False, save=path, title=title)
    path = f'terminal{idx}_{args.k}K{args.pc}PC{args.res}res.png'
    title = f'Terminal states ({idx}) K={args.k} PC={args.pc} res={args.res}'        
    g.plot_macrostates(which="terminal", legend_loc="right", s=100, show=False, save=path, title=title)
    g.compute_fate_probabilities(tol=1e-10, use_petsc=True, preconditioner='ilu')
    path = f'fateProb{idx}_{args.k}K{args.pc}PC{args.res}res.png'
    title = f'Fate Probabilities ({idx}) K={args.k} PC={args.pc} res={args.res}'
    g.plot_fate_probabilities(same_plot=True, save = path, title=title)
    quality_dict[idx] = _check_macrostate_quality(g, idx)

# Save results for GPCCA
path = os.path.join(os.getcwd(), f"quality_{args.k}K{args.pc}PC{args.res}res.csv")
pd.DataFrame(quality_dict, index=['spectralGap', 'minChi', 'crispness'] ).to_csv(path, index = True, header = True, sep=',')
