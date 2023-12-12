import os 
import argparse
import scanpy as sc

seed= 52
CWD = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument('h5adName')
parser.add_argument('K',  type=int,  help='Integer indicating number of neighbors to compute the KNN graph')
parser.add_argument('PC', type=int, help='Integer indicating the number of PCs to consider')
args = parser.parse_args()

print("Loading h5ad file")
DATA_PATH = os.path.join(CWD, args.h5adName)
adata = sc.read_h5ad(DATA_PATH)


if 'loom' in args.h5adName:
    save_figure_path =  f'_loom_{args.K}K_{args.PC}PC.png'
else:
    save_figure_path =  f'_cellRanger_{args.K}K_{args.PC}PC.png'

sc.pp.neighbors(adata, n_neighbors=args.K, n_pcs=args.PC, random_state = seed)
sc.tl.umap(adata, random_state=seed)
sc.tl.louvain(adata, random_state=seed)
sc.pl.umap(adata, color='louvain', save = save_figure_path)
