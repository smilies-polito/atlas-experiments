import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from muon import MuData
from anndata import AnnData
from scipy.sparse import issparse
from typing import Optional, Union

def plot_grouped_heatmap(data: Union[mu.MuData, AnnData], similarity_key:str="connectivities", group_key:str="celltype", save:Optional[str]=None):

	if not similarity_key in data.obsp:
		if not similarity_key in data.obsm:
			raise KeyError(f"{similarity_key} not in data.obsm or data.obsp")
		W = data.obsm[similarity_key]
		manipulate_columns = False
	else:
		W = data.obsp[similarity_key]
		manipulate_columns = True

	if not group_key in data.obs.columns:
		raise KeyError(f"{group_key} not in data.obs")
	
	if issparse(W):
		W= W.toarray()
	if isinstance(W, pd.DataFrame):
		W = W.to_numpy()
	
	celltypes = data.obs[group_key].astype("category")
	celltypes_ordered = celltypes.cat.categories

	sorted_idx = np.argsort(celltypes.cat.codes.values)
	if manipulate_columns:
		W_sorted = W[sorted_idx, :][:, sorted_idx]
	else:
	W_sorted = W[sorted_idx, :]
		
	sorted_celltypes = celltypes.iloc[sorted_idx]

	lut = dict(zip(celltypes_ordered, sns.color_palette("tab20", len(celltypes_ordered))))
	row_colors = [lut[ct] for ct in sorted_celltypes]
	col_colors = row_colors if manipulate_columns else None

	sns.set(style="white")
	g = sns.clustermap(
  	 W_sorted,
	 row_cluster=False,
   	 col_cluster=False,
   	 row_colors=row_colors,
  	 col_colors=col_colors,
	 cmap="viridis",
  	 xticklabels=False,
	 yticklabels=False,
	 figsize=(10, 10)
	)

	for label in celltypes_ordered:
	    g.ax_col_dendrogram.bar(0, 0, color=lut[label], label=label, linewidth=0)
	    g.ax_col_dendrogram.legend(
       		 loc="center",
	       	 ncol=min(6, len(celltypes_ordered)),
       		 title=group_key
    		)

	if save is not None:
		plt.savefig(save)
		plt.close()
	

