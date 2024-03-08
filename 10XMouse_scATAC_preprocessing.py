import os 
import argparse
import numpy as np
import pandas as pd
import scanpy as sc 

from csvLoader import CSVLoader


CWD = os.getcwd()
SIGNAC_FOLDER = os.path.join(CWD, 'signac_folder')
RESULT_FOLDER = os.path.join(CWD, 'adata_folder')

parser = argparse.ArgumentParser()
parser.add_argument('-k_atac', type = int, help = 'Integer indicating the size of the scATAC-seq data neighborhood')
parser.add_argument('-k_rna', type = int, help = 'Integer indicating the size of the scRNA-seq data neighborhood')
parser.add_argument('-pc', type = int, help='Integer indicating the number of principal components for scRNA-seq data')
parser.add_argument('-lsi', type = int, help='Integer indicating the number of dimensions for scATAC-seq data')
parser.add_argument('-res_rna', type = float, help='Float indicating the number resolution for Louvain community detection algorithm for scRNA-seq data')
args = parser.parse_args()

res_atac = 1.0 #Kept constant
embeddings_atac = ['umap', 'lsi']

loader = CSVLoader(SIGNAC_FOLDER)
loader.create_adata(embeddings= embeddings_atac, neighbors=True)

atac = loader.adata
atac_path = os.path.join(RESULT_FOLDER, f'10xMouse_{args.k_atac}K{args.lsi}LSI{res_atac}res.h5ad')
atac.write_h5ad(atac_path)

# Load scRNA-seq data
rna_path = os.path.join(RESULT_FOLDER, f'10xMouse_{args.k_rna}K{args.pc}PC{args.res_rna}res.h5ad')
rna = sc.read_h5ad(rna_path)

# Subset for common cells
barcodes = set(rna.obs_names).intersection(set(atac.obs_names))
atac = atac[atac.obs_names.isin(barcodes)]
rna = rna[rna.obs_names.isin(barcodes)]

# Add Louvain clusters and categorical seurat clusters
atac.obs = atac.obs.merge(rna.obs.louvain, how='left', left_index = True, right_index=True)
atac.obs.seurat_clusters = atac.obs.seurat_clusters.astype("category")

# Load annotations from 10XGenomics and keep only promoter genes
annotations = pd.read_csv(os.path.join(os.getcwd(), 'peak_annotation.tsv'), sep='\t', header = 0)
annotations = annotations[annotations.peak_type == 'promoter']
annotations['peak'] = annotations[['chrom', 'start', 'end']].apply(lambda row: '-'.join(row.values.astype(str)), axis=1)

# Gene and peaks intersections with annotations
annotations = annotations[(annotations.gene.isin(rna.var_names)) & (annotations.peak.isin(atac.var_names))]
intersection = set(annotations.peak).intersection(set(atac.var_names))
annotations = annotations[annotations.peak.isin(intersection)]
atac = atac[:, atac.var_names.isin(intersection)]

intersection = set(annotations.gene).intersection(set(rna.var_names))
annotations = annotations[annotations.gene.isin(intersection)]
rna = rna[:, rna.var_names.isin(intersection)]     

# Merge annotations with atac.var
merged_df = atac.var.merge(annotations, how='left', left_index =True, right_on='peak')
merged_df.drop_duplicates(subset= ['peak'], keep='first', inplace=True)
merged_df.set_index('peak', inplace=True)
atac.var = merged_df

print(f"Final shapes rna= {rna.shape} atac={atac.shape}")

# Save atac 
atac.write_h5ad(atac_path)
rna.write_h5ad(rna_path)


# Plot clusters 
title = f'Seurat K={args.k_atac} LSI={args.lsi} res={res_atac}'
path = f'_scATAC_seurat_{args.k_atac}K{args.lsi}LSI{res_atac}res.png'
sc.pl.umap(atac, color=['seurat_clusters'], title=title, show = False, save = path)

title = f'Louvain rna K={args.k_atac} LSI={args.lsi} res={res_atac}'
path = f'_scATAC_louvain_{args.k_atac}K{args.lsi}LSI{res_atac}res.png'
sc.pl.umap(atac, color=['louvain'], title=title, show = False, save = path)
