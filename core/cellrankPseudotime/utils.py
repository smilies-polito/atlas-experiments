import os
import pandas as pd
import numpy as np
import cellrank as cr
import matplotlib.pyplot as plt 

from scipy.sparse import hstack, csr_matrix
from anndata import AnnData
from cellrank.estimators import GPCCA
from typing import Literal, Optional, Union, Sequence 
seed = 42

def _check_conjugate(idx: int, eig: np.ndarray):
    """
    
    Function that check whether the value is real.

    Params
    -------
    idx: int
        index of the eigenvalue
    eig: numpy.ndarray
        eigenvalue

    Returns
    -------
    int : n_macrostates

    """
    if np.iscomplex(eig[idx]):
        print(f"Complex value needs its conjugate, returning {idx+2} states")
        return idx+2
    else: 
        return idx+1
    

def _compute_spectral_gap(eigenvalues: np.ndarray, ns: int):
	"""

    Function that computes the spectral gap.

    Parameters
    -----------
    eigenvalues: numpy.ndarray 
        sotres the eigenvalues for GPCCA 
    ns: int 
        number of macrostates

    Returns
    -------- 
    spectral gap: float (or complex value) 
    
	"""
	return eigenvalues[ns-1] - eigenvalues[ns] if ns<eigenvalues.shape[0] else None


def _check_macrostate_quality(g: cr.estimators.GPCCA, ns: int):
	"""

    Function that computes n_states quality according to DOI 10.1007/s11634-013-0134-6 (spectral gap, minChi, crispness). 

    Parameters
    ----------
    g: cellrank.estimators.GPCCA
    ns: int
        number of macrostates
    
    Returns
    -------- 
    spectral_gap: float (or complex value) 
        spectral gap for g
    minChi: float 
        minChi for g 
    crispness: float 
        crispness for g 

	"""
	spectral_gap = _compute_spectral_gap(g.eigendecomposition['D'], ns)
	minChi = np.min(g.macrostates_memberships.X)
	crispness = g._gpcca.crispness_values[0]
	return spectral_gap, minChi, crispness


def _plot_quality_gpcca(df: pd.DataFrame, saving_path:str, title: Optional[str]= None):
	"""

    Function than performs the scatter plot of minChi against crispness.
    
    Parameters
    ----------
    df: pandas.DataFrame 
        contains minChi and crispness values.
	
	"""
	plt.figure(figsize = (8,6))
	scatter = plt.scatter(df['minChi'], df['crispness'])

	if title is not None:
		plt.title(title)
	plt.xlabel("minChi")
	plt.ylabel("crispness")
	for i in df.index:
		plt.annotate(i, (df['minChi'].loc[i], df['crispness'].loc[i]))
	
	plt.savefig(os.path.join(saving_path,"gpcca_qualities.png"))
	plt.close()



def _compute_macrostates(gpcca: GPCCA,
                         n_states: int, cell_type_key: str, 
                         barcodes: Optional[Sequence]=None,
                         plot:bool = False, save_fate: bool = True,
						 saving_path : Optional[str]=None):
	"""

    Function that performs a single GPCCA computation. 

    Parameters
    ----------
    gpcca : cellrank.estimators.GPCCA
        GPCCA object
    n_states : int
        number of macrostates
    cell_type_key: str 
        key in _atac.obs.columns where celltype annotations are stored (default is "cell_type").  
    barcodes: Sequence, optional
        cell barcodes. Must be specified if save_fate is True.
    k : int
        dimensionality of the neighborhood
    lsi : int, optional
        number of dimensions for SVD (default is None)
    pc: int, optional
        number of principal components (default is None)
    plot: bool
        if to plot macrostate composition, initial and temrinal states and fate probabilities
    save_fate : bool
        if to save fate probabilities as csv_file
	"""
	if (plot or save_fate) and not os.path.exists(saving_path):
		os.mkdir(saving_path)

	gpcca.compute_macrostates(n_states=n_states, cluster_key = cell_type_key)
	gpcca.predict_initial_states()
	gpcca.predict_terminal_states(allow_overlap=True)   

	if plot:
		gpcca.plot_macrostate_composition(key=cell_type_key, show=False, 
					save = os.path.join(saving_path, f"macrostateComposition{n_states}.png"),
					title=f"Macrostate {n_states} Composition")

		gpcca.plot_coarse_T(annotate=True, 
                            save = os.path.join(saving_path, f"coarseT{n_states}.png"),
                            title=f"Coarse Transition Matrix ({n_states})")

		gpcca.plot_macrostates(which="initial", legend_loc="right", s=100, show=False, 
                               save = os.path.join(saving_path, f"initial{n_states}.png"), 
                               title=f"Initial States ({n_states})")

		gpcca.plot_macrostates(which="terminal", legend_loc="right", s=100, show=False, 
                               save = os.path.join(saving_path, f"terminal{n_states}.png"),
                               title=f"Terminal States ({n_states})")
    
	gpcca.compute_fate_probabilities(tol=1e-10, use_petsc=True, preconditioner="ilu")
	
	if plot: 
		gpcca.plot_fate_probabilities(same_plot=True,                              
                                     save = os.path.join(saving_path, f"fateProb{n_states}.png"),
                                     title=f"Fate Probabilities ({n_states})") 

	if save_fate:
		if barcodes is None:
			raise ValueError("Barcodes must be speficied when save_fate is True")
		df = pd.DataFrame(gpcca.fate_probabilities.X, columns = gpcca.fate_probabilities.names, index=barcodes)
		df['entropy'] = gpcca.compute_lineage_priming(method="entropy")
		df['KL'] = gpcca.compute_lineage_priming(method="kl_divergence")
		df.to_csv(os.path.join(saving_path, f"terminal_{n_states}.tsv"), sep="\t", header=True, index=True)
    
	return _check_macrostate_quality(gpcca, n_states)
    







 
