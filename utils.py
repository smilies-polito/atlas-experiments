import os
import pandas as pd
import numpy as np
import cellrank as cr
import matplotlib.pyplot as plt 

from scipy.sparse import hstack, csr_matrix
from anndata import AnnData
from cellrank.estimators import GPCCA
from typing import Literal, Optional, Union, Sequence 
seed = 52

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
    if np.iscomplex(eig):
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


def _plot_quality_gpcca(df: pd.DataFrame, **kwargs):
    """

    Function than performs the scatter plot of minChi against crispness.
    
    Parameters
    ----------
    df: pandas.DataFrame 
        contains minChi and crispness values.

    """

    fig = plt.figure()
    plt.scatter(df['minChi'], df['crispness'])
    title = "GPCCA macrostates quality" 
    title = title + "_" +str(kwargs['k']) + "K" if 'k' in kwargs.keys() else title
    title = title + " " +str(kwargs['pc']) + "PC" if 'pc' in kwargs.keys() else title
    title = title + " " +str(kwargs['lsi']) + "LSI" if 'lsi' in kwargs.keys() else title

    path = os.path.join(os.getcwd() , 'figures', title.replace(" ", "").lower() + '.png')

    plt.title(title)
    plt.xlabel("minChi")
    plt.ylabel("crispness")
    for i in df.index:
        plt.annotate(i, (df['minChi'].loc[i], df['crispness'].loc[i]))
    fig.savefig(path)


def _save_qualities_gpcca(df: pd.DataFrame, **kwargs):
    """

    Function that saves GPCCA quality values

    Parameters
    ----------
    df: pandas.DataFrame
        contains values for minChi, crispness and spectral_gap for all GPCCA estimates
    
    """
    df['n_states'] = df.index
    df['k'] = np.ones(df.shape[0])*kwargs['k'] if 'k' in kwargs.keys() else np.ones(df.shape[0])*-1
    df['pc'] = np.ones(df.shape[0])*kwargs['pc'] if 'pc' in kwargs.keys() else np.ones(df.shape[0])*-1
    df['lsi'] = np.ones(df.shape[0])*kwargs['lsi'] if 'lsi' in kwargs.keys() else np.ones(df.shape[0])*-1

    path = os.path.join(os.getcwd() , 'macrostate_quality.csv')

    if not os.path.exists(path):
        df.to_csv(path, header=True, index=False)
    else:
        df.to_csv(path, header=False, index=False, mode="a")


def _save_probabilities(gpcca, barcodes: Sequence, n_states: int, **kwargs):
    """
    
    Function that saves fate probabilities towards terminal states.
    
    Parameters
    -----------
    gpcca: cellrank.estimators.GPCCA 
        object containing terminal macrostates and fate probabilities. 
    barcodes: Sequence
        cell barcodes
    n_states: int
        number of macrostates

    """

    path = f"probabilities_{n_states}states_"
    path = path +str(kwargs['k']) + "K" if 'k' in kwargs.keys() else path
    path = path +str(kwargs['pc']) + "PC" if 'pc' in kwargs.keys() else path
    path = path +str(kwargs['lsi']) + "LSI" if 'lsi' in kwargs.keys() else path
    path = path + ".csv"

    path = os.path.join(os.getcwd(), 'fate_probabilities', path)

    df = pd.DataFrame(gpcca.fate_probabilities.X, columns = gpcca.fate_probabilities.names, index=barcodes)
    df['entropy'] = gpcca.compute_lineage_priming(method="entropy")
    df['KL'] = gpcca.compute_lineage_priming(method="kl_divergence")
    df.to_csv(path)


def _add_cell_type(adata:AnnData, annotations:pd.DataFrame):
    """

    Function that adds to the adata the celltype annotations.

    Parameters
    ---------
    adata: anndata.AnnData
    annotations: pandas.DataFrame
        contains cell clusters annotaitons to be added to AnnData 

    Updates
    --------
    adata.obs

    """
    adata.obs = adata.obs.merge(annotations, how="left", right_index=True, left_index=True)


def _compute_macrostates(gpcca: GPCCA,
                         n_states: int, cell_type_key: str, 
                         barcodes: Optional[Sequence]=None,
                         k=int, pc: Optional[int]=None, 
                         lsi : Optional[int]=None, plot:bool = False, save_fate: bool = True, ):
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

    Note. at least one between lsi and pc should be specified.

    """
    if lsi is None and pc is None:
        raise ValueError("At least one between lsi and pc should be specified")
    
    title, path = (f"K={k}", f"{k}K")
    title, path = (title+ f" LSI={lsi}", path + f"{lsi}LSI") if lsi is not None else (title, path)
    title, path = (title+ f" PC={pc}", path + f"{pc}PC") if pc is not None else (title, path)
    path = path+".png"

    gpcca.compute_macrostates(n_states=n_states, cluster_key = cell_type_key)
    gpcca.predict_initial_states()
    gpcca.predict_terminal_states(allow_overlap=True)   

    if plot:
        gpcca.plot_macrostate_composition(key=cell_type_key, show=False, 
                                          save = f"macrostateComposition{n_states}_{path}",
                                          title=f"Macrostate {n_states} Composition {title}")

        gpcca.plot_coarse_T(annotate=True, 
                            save = f"coarseT{n_states}_{path}",
                            title=f"Coarse Transition Matrix ({n_states}) {title}")

        gpcca.plot_macrostates(which="initial", legend_loc="right", s=100, show=False, 
                               save = f"initial{n_states}_{path}",
                               title=f"Initial States ({n_states}) {title}")
             
        gpcca.plot_macrostates(which="terminal", legend_loc="right", s=100, show=False, 
                               save = f"terminal{n_states}_{path}",
                               title=f"Terminal States ({n_states}) {title}")
    
    gpcca.compute_fate_probabilities(tol=1e-10, use_petsc=True, preconditioner="ilu")

    if plot: 
        gpcca.plot_fate_probabilities(same_plot=True,                              
                                     save = f"fateProb{n_states}_{path}",
                                     title=f"Fate Probabilities ({n_states}) {title}") 

    if save_fate:
        if barcodes is None:
            raise ValueError("Barcodes must be speficied when save_fate is True")
        _save_probabilities(gpcca, barcodes=barcodes, n_states = n_states, k=k, lsi=lsi, pc=pc)
    
    return _check_macrostate_quality(gpcca, n_states)
    







 