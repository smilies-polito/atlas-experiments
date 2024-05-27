import os
import argparse
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

def FindExpression(adata, gene):
    """
    Function: given a gene plot its expression along with the clusters and saves the figure
    """
    if gene in adata.var_names:
        title = f'Expression of {gene}'
        path = os.path.join(CWD, 'figures', f'{gene}_expression.png')
        fig, ax = plt.subplots(figsize=(8, 6))
        sc.pl.umap(adata, color='louvain', legend_loc='on data', show=False, ax=ax)
        sc.pl.umap(adata, color=gene, cmap='Reds', use_raw=False, title=title, show=False, ax=ax)
        fig.savefig(path)

CWD = os.getcwd()
seed= 52

parser = argparse.ArgumentParser()
parser.add_argument('-k', type=int, help='Integer describing the size of the local neighborhood')
parser.add_argument('-pc', type=int, help='Integer describing the number of principal components')
args= parser.parse_args()

ADATA_PATH = os.path.join(CWD, 'signac_folder')
DATA_PATH= os.path.join(ADATA_PATH, f"scRNA_{args.k}K{args.pc}PC.h5ad")
adata = sc.read_h5ad(DATA_PATH)

title = f'Marker genes for every cluster (Wilcoxon) K={args.k}, PC={args.pc}'
path = f'_{args.k}K{args.pc}PC.png'
sc.tl.rank_genes_groups(adata, 'louvain', method='wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, title=title, save=path, show=False)

#Dataset with 10 most espressed genes in the clusters
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
df = pd.DataFrame(
    {group + '_' + key[:1]: result[key][group]
    for group in groups for key in ['names', 'pvals']}).head(10)

path = os.path.join(ADATA_PATH, f'geneExpression_{args.k}K{args.pc}PC.csv')
df.to_csv(path, sep=',', header=True)

# Genes from litterature
markers = {
    # 'cajal-retzius' : ['Reln', 'Nrxn1', 'Trp73', 'Lhx5', 'Nhlh2'],
    'layer2to4' : ['Satb2', 'Nrgn', 'Inhba',  'Cux2', 'Prox1'],
    # 'forebrainGABA' : ['Slc32a1'],
    # 'interneurons' : ['Dlx1', 'Dlx2', 'Gad1', 'Gad2', 'Bcl11b', 'Lhx6', 'Adarb2'],
    'opc' : ['Olig2', 'Pdgfra', 'Sox10'],
    'astrocytes' : ['Vim', 'Slc1a3', 'Nes', 'Aldoc'],
    'layer5to6' : [ 'Bcl11b', 'Fezf2', 'Nrgn', 'Crym', 'Rorb', 'Nr4a2'], 
    'ipc' : ['Eomes', 'Top2a', 'Elavl2', 'Elavl4'],    
    'radialGlia' : [ 'Vim',  'Nes'],
    'ependymal' : ['Ednrb', 'Sulf1'],
    'SVZ' : [ 'Sema3c', 'Eomes'],
    # 'Microglia': ['Trem2', 'Ctss',]
}

genes = [] 
[genes.extend(g) for g in markers.values()]

for gene in genes:
    FindExpression(adata, gene)

title = f'Markers K={args.k} PC={args.pc}'
path = f'litteratureMarkers_{args.k}K{args.pc}PC.png'
sc.pl.dotplot(adata, markers, 'louvain', dendrogram=True, title = title, save = path, show=False) 