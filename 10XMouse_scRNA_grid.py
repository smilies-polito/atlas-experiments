import os
import argparse
import scanpy as sc
import scvelo as scv

seed = 52

parser = argparse.ArgumentParser()
parser.add_argument('-k', type=int, help='Integer describing the size of the local neighborhood')
parser.add_argument('-pc', type=int, help='Integer describing the number of principal components')
parser.add_argument('-res', type=float, help='Float describing the resolution of Louvain community detection algorithm')
args = parser.parse_args()

CWD = os.getcwd()
ADATA_PATH = os.path.join(CWD,'scRNA_adata')
DATA_PATH = os.path.join(ADATA_PATH, '10xMouse_loom_preprocessed.h5ad')

adata =sc.read_h5ad(DATA_PATH)

sc.pp.neighbors(adata, n_neighbors=args.k, n_pcs=args.pc, random_state = seed)
sc.tl.louvain(adata, resolution=args.res, random_state=seed)
scv.pp.moments(adata, n_neighbors=args.k, n_pcs=args.pc)
title = f'K={args.k} PC={args.pc} res={args.res}'
path =  f'_scRNA_{args.k}K{args.pc}PC{args.res}res.png'
sc.tl.umap(adata, random_state=seed)
sc.pl.umap(adata, color=['louvain'], title=title, show=False, save=path)

path = os.path.join(ADATA_PATH, f'10xMouse_{args.k}K{args.pc}PC{args.res}res.h5ad')
adata.write_h5ad(path)
