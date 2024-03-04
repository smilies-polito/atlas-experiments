import os
import argparse
import pandas as pd
import scanpy as sc 
import cellrank as cr

seed = 52

parser = argparse.ArgumentParser()
parser.add_argument('-k', type = int, help='Integer describing the size of the local neighborhood')
parser.add_argument('-pc', type = int, help = 'Integer describing the number of principal components')
parser.add_argument('-res', type=float, help='Float describing the resolution of the Louvain community detection algorithm')
args = parser.parse_args()

CWD = os.getcwd()
FIGURE_FOLDER = os.path.join(CWD, 'figures')
ADATA_FOLDER = os.path.join(CWD, 'scRNA_adata')
DATA_PATH = os.path.join(ADATA_FOLDER, f'10xMouse_{args.k}K{args.pc}PC{args.res}res.h5ad')

starting_cells = pd.read_csv('starting_cells.csv', header=None, index_col = 0).to_numpy().flatten()

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
for idx, eig in enumerate(eigenvalues):
    #At least two macrostates otherwise no sense, at most 12 macrostates to limit computations (should be enough) and considering only real values
    if(eig.imag == 0) and idx > 0 and idx < 12: 
        ns = idx+1
        g.compute_macrostates(n_states=ns, cluster_key='louvain')
        tile = f'Macrostate ({ns}) Composition K={args.k} PC={args.pc} res={args.res}'
        path = f'macrostateComposition{ns}_{args.k}K{args.pc}PC{args.res}res.png'
        g.plot_macrostate_composition(key='louvain', show=False, save = path, title=title)
        title = f'Coarse Transition Matrix ({ns}) K={args.k} PC={args.pc} res={args.res}'
        path = f'coarseT{ns}_{args.k}K{args.pc}PC{args.res}res.png'
        g.plot_coarse_T(annotate=True, save=path, title=title)
        g.predict_initial_states()
        g.predict_terminal_states(allow_overlap=True)
        title = f'Initial states ({ns}) K={args.k} PC={args.pc} res={args.res}'
        path = f'initial{ns}_{args.k}K{args.pc}PC{args.res}res.png'
        g.plot_macrostates(which="initial", legend_loc="right", s=100, show=False, save=path, title=title)
        path = f'terminal{ns}_{args.k}K{args.pc}PC{args.res}res.png'
        title = f'Terminal states ({ns}) K={args.k} PC={args.pc} res={args.res}'        
        g.plot_macrostates(which="terminal", legend_loc="right", s=100, show=False, save=path, title=title)
        g.compute_fate_probabilities()
        path = f'fateProb{ns}_{args.k}K{args.pc}PC{args.res}res.png'
        title = f'Fate Probabilities ({ns}) K={args.k} PC={args.pc} res={args.res}'
        g.plot_fate_probabilities(same_plot=True, save = path, title=title)
