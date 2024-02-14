import os
import argparse
import scanpy as sc
import scvelo as scv

seed = 52
parser= argparse.ArgumentParser()
parser.add_argument('-k', type=int, help='Integer describing the size of the local neighborhood')
parser.add_argument('-pc', type=int, help='Integer describing the number of principal components')
parser.add_argument('-res', type=float, help='Float indicating the resolution of the Louvain community detection algorithm')
args = parser.parse_args()

CWD = os.getcwd()
DATA_FOLDER = os.path.join(CWD, "scRNA_adata")
DATA_PATH = os.path.join(DATA_FOLDER, '10xMouse_loom_preprocessed.h5ad')

adata = sc.read_h5ad(DATA_PATH)

scv.pp.moments(adata, n_neighbors=args.k, n_pcs=args.pc)
sc.pp.neighbors(adata, n_neighbors=args.k, n_pcs=args.pc, random_state = seed)
sc.tl.louvain(adata, resolution=args.res, random_state=seed)
sc.tl.umap(adata, random_state=seed)

# Recovers full splicing kinetics 
scv.tl.recover_dynamics(adata, n_jobs=-1)
scv.tl.velocity(adata, mode='dynamical')
scv.tl.velocity_graph(adata, n_jobs=-1)

title = f"Dynamical model K={args.k} PC={args.pc} resolution={args.res}"
path= f"dynamicalModel_{args.k}K{args.pc}PC{args.res}resolution.png"
scv.pl.velocity_embedding_stream(adata, basis='umap', color=['louvain'], title=title, save= path, show=False)

# Compute latent time (cell's internal clock)
title = f"Cell specific latent time K={args.k} PC={args.pc} resolution={args.res}"
path= f"latentTime_{args.k}K{args.pc}PC{args.res}resolution.png"
scv.tl.latent_time(adata)
scv.pl.scatter(adata, color='latent_time', color_map='gnuplot', size=40, save=path, title=title, show=False)

# Saving file
path = os.path.join(DATA_FOLDER, f"scVeloDynamical_{args.k}K{args.pc}PC{args.res}res.h5ad")
adata.write_h5ad(path)

# Top-likelihood genes
scv.tl.rank_dynamical_genes(adata, groupby='louvain')
df = scv.get_df(adata, 'rank_dynamical_genes/names')
path = os.path.join(DATA_FOLDER, f'likelihoodGenes_{args.k}K{args.pc}PC{args.res}res.csv')
df.to_csv(path, header=True, sep=',')

for cluster in set(adata.obs['louvain']):
    title = f'Top genes cluster {cluster}: K={args.k} PC={args.pc} res={args.res}'
    path = f'topGenesscVELO{cluster}_{args.k}K{args.pc}PC{args.res}res.png'
    scv.pl.scatter(adata, df[cluster][:5], ylabel=cluster, title=title, save = path)