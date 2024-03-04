import os
import numpy as np
import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt

seed = 52

CWD = os.getcwd()
LOOM_PATH = os.path.join(CWD, '10X_multiome_mouse_brain.loom')
ADATA_PATH = os.path.join(CWD,'scRNA_adata')

if not os.path.exists(ADATA_PATH):
    os.makedirs(ADATA_PATH)

# Data loading + Multivelo annotations
adata =sc.read_loom(LOOM_PATH)
adata.var_names_make_unique()
adata.obs_names = [obs.split(':')[1][:-1] + '-1' for obs in adata.obs_names]


# QC metrics 
adata.var['mt'] = adata.var_names.str.startswith('mt-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'],log1p=False, percent_top=None, inplace=True)

print("Saving QC files")
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts',  'pct_counts_mt'], jitter = 0.04, 
             multi_panel= True, save = "_scRNA.png", show=False)

sc.pl.scatter(adata, x='n_genes_by_counts', y='total_counts', save = "_scRNA_totalCounts.png", show=False)
sc.pl.scatter(adata, x='n_genes_by_counts', y='pct_counts_mt', save = "_scRNA_mitoPercent.png", show=False)

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
sc.pl.pca_variance_ratio(adata, n_pcs=50, save='_scRNA.png', show=False)

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
preprocessing_path = os.path.join(ADATA_PATH, '10xMouse_loom_preprocessed.h5ad')
adata.write_h5ad(preprocessing_path) 
