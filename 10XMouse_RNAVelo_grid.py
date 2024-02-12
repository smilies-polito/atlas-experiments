import os
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt

from itertools import product
from matplotlib import rcParams

seed = 52


CWD = os.getcwd()
LOOM_PATH = os.path.join(CWD, '10X_multiome_mouse_brain.loom')
CELLANNO_PATH = os.path.join(CWD, 'cell_annotations.tsv')
DATA_PATH = os.path.join(CWD, 'scRNA_adata', '10xMouse_loom_preprocessed.h5ad')

if not os.path.exists(os.path.join(CWD, 'scRNA_adata')):
    os.makedirs(os.path.join(CWD, 'scRNA_adata'))


# Data loading + Multivelo annotations
adata =sc.read_loom(LOOM_PATH)
adata.var_names_make_unique()
adata.obs_names = [obs.split(':')[1][:-1] + '-1' for obs in adata.obs_names]

cell_annot = pd.read_csv(CELLANNO_PATH, sep='\t', index_col=0)
adata = adata[cell_annot.index,:]
adata.obs['celltype'] = cell_annot['celltype']

# QC metrics 
adata.var['mt'] = adata.var_names.str.startswith('mt-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'],log1p=False, percent_top=None, inplace=True)

print("Saving QC files")
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts',  'pct_counts_mt'], jitter = 0.04, 
             multi_panel= True, save = "_scRNA.png")

sc.pl.scatter(adata, x='n_genes_by_counts', y='total_counts', save = "_scRNA_totalCounts.png")
sc.pl.scatter(adata, x='n_genes_by_counts', y='pct_counts_mt', save = "_scRNA_mitoPercent.png")

print("Filtering")
lower, upper = np.percentile(adata.obs.n_genes_by_counts, [2,98]).astype(int)
print(f'Retaining cells with {lower} < n_genes_by_counts < {upper}')
sc.pp.filter_cells(adata, min_genes=lower)
sc.pp.filter_cells(adata, max_genes=upper)
scv.pp.filter_and_normalize(adata, min_shared_counts=10, n_top_genes=2000)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# PCA
print("Computing PCA")
scv.pp.pca(adata, n_comps=50, use_highly_variable=True)
sc.pl.pca_variance_ratio(adata, n_pcs=50, save='_scRNA.png')

# Explained variance plot
cumulative_variance_ratio = np.cumsum(adata.uns['pca']['variance_ratio'])
fig, ax = plt.subplots(figsize=(7, 6))
plt.scatter(range(1,51), cumulative_variance_ratio, )
# Plotting variance explained by first 10,15,20,25,30 components according to elbow plot 
plt.axhline(cumulative_variance_ratio[9], color='silver', linestyle='dashed')  
plt.axhline(cumulative_variance_ratio[14], color='silver', linestyle='dashed') 
plt.axhline(cumulative_variance_ratio[19], color='silver', linestyle='dashed') 
plt.axhline(cumulative_variance_ratio[24], color='silver', linestyle='dashed') 
plt.axhline(cumulative_variance_ratio[29], color='silver', linestyle='dashed') 
ax.set_xticks(range(0, 100, 10))
plt.savefig(os.path.join(CWD,'figures', 'scRNA_explainedVar.png'))

# Save preprocessed adata
adata.write_h5ad(DATA_PATH) 

# Multiple K and PCs 
K = [10, 30, 50, int((adata.shape[0])**(1/2)), 80, 100]
PC = [10, 15, 20, 25, 30]
resolution = [0.5, 0.7, 1.0, 1.4]
grid = product(K, PC, resolution)

for k, pc, res in grid:
    print(f"Running with K={k}, PC={pc}, resolution={res}")
    adata = sc.read_h5ad(DATA_PATH)
    sc.pp.neighbors(adata, n_neighbors=k, n_pcs=pc, random_state = seed)
    sc.tl.louvain(adata, resolution=res, random_state=seed)
    scv.pp.moments(adata, n_neighbors= k, n_pcs=pc)
    title = f'K={k} PC={pc} res={res}'
    path =  f'_scRNA_{k}K_{pc}PC_{res}res.png'
    sc.tl.umap(adata, random_state=seed)
    sc.pl.umap(adata, color=['louvain', 'celltype'], title=title, show=False, save=path)






