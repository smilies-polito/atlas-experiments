import os
import argparse
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

seed = 52

LOOM_FILE = os.path.join(os.getcwd(), "data_folder", "10X_multiome_mouse_brain.loom")
CELL_ANNOTATION_PATH = os.path.join(os.getcwd(), "cell_annotations.tsv")
RESULTS_PATH = os.path.join(os.getcwd(), "figures")

celltypes = pd.read_csv(CELL_ANNOTATION_PATH, sep='\t', header=0, index_col=0)
celltypes_key = celltypes.columns[0]

adata = sc.read_loom(LOOM_FILE)
adata.var_names_make_unique()
adata.obs_names = [name.split(":")[1][:-1] + "-1" for name in adata.obs_names]

adata =  adata[adata.obs_names.isin(celltypes.index.values)]
adata.obs = pd.merge(adata.obs, celltypes, how="left", left_index=True, right_index=True)

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.tl.pca(adata, n_comps=10)
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=10)
sc.tl.umap(adata)

fig, ax = plt.subplots(figsize=(12,10))
sc.pl.umap(adata, color=[celltypes_key], show=False, title="Cell types annotations", ax=ax, legend_loc="on data", legend_fontsize="medium")
fig.savefig(os.path.join(RESULTS_PATH, "clusters.png"))


def FindExpression(adata, gene, saving_path, celltype_key):
    """
    Function: given a gene plot its expression along with the clusters and saves the figure
    """
    if gene in adata.var_names:
        title = f'Expression of {gene}'
        path = os.path.join(saving_path, f'{gene}_expression.png')
        fig, ax = plt.subplots(figsize=(8, 6))
        sc.pl.umap(adata, color=[celltype_key], legend_loc='on data', legend_fontsize="x-small", show=False, ax=ax)
        sc.pl.umap(adata, color=gene, cmap='Reds', use_raw=False, title=title, show=False, ax=ax)
        fig.savefig(path)

# Genes from litterature
markers = {
    'cajal-retzius' : ['Reln', 'Nrxn1', 'Trp73', 'Lhx5', 'Nhlh2'],
    'layer2to4' : ['Satb2', 'Nrgn', 'Inhba',  'Cux2', 'Prox1'],
    'forebrainGABA' : ['Slc32a1'],
    'interneurons' : ['Dlx1', 'Dlx2', 'Gad1', 'Gad2', 'Bcl11b', 'Lhx6', 'Adarb2'],
    'opc' : ['Olig2', 'Pdgfra', 'Sox10'],
    'astrocytes' : ['Vim', 'Slc1a3', 'Nes', 'Aldoc'],
    'layer5to6' : [ 'Bcl11b', 'Fezf2', 'Nrgn', 'Crym', 'Rorb', 'Nr4a2'], 
    'ipc' : ['Eomes', 'Top2a', 'Elavl2', 'Elavl4'],    
    'radialGlia' : [ 'Vim',  'Nes'],
    'ependymal' : ['Ednrb', 'Sulf1'],
    'SVZ' : [ 'Sema3c', 'Eomes'],
    'Microglia': ['Trem2', 'Ctss',]
}

genes = [] 
[genes.extend(g) for g in markers.values()]

for gene in genes:
    FindExpression(adata, gene, RESULTS_PATH, celltypes_key)
