import os
import pandas as pd
import numpy as np
import cellrank as cr
import matplotlib.pyplot as plt 

from typing import Literal, Optional, Union, Sequence 
seed = 52

def _check_conjugate(idx: int, eig: np.ndarray):
    if np.iscomplex(eig):
        print(f"Complex value needs it conjugate, returning {idx+2} states")
    
        return idx+2
    else: 
        return idx+1
    
def _compute_spectral_gap(eigenvalues: np.ndarray, ns: int):
    """
        Function that computed the spectral gap, thence the difference between lambda_n_c and lambda_n_c+1 
        params:
            - eigenvalues: array storing the eeigenvalues for GPCCA 
            - ns: number of macrostates
        output: 
            - spectral gap: float (or complex value) indicating the spectral gap
    """
    return eigenvalues[ns-1] - eigenvalues[ns] if ns<eigenvalues.shape[0] else None


def _check_macrostate_quality(g: cr.estimators.GPCCA, ns: int):
    """
        Function that computes n_c quality according to DOI 10.1007/s11634-013-0134-6:
        Criteria are: 
        1. spectral gap
        2. minChi criterion: min_i min_j X(i,j) with X the membership matrix
        3. Optimality of PCCA+ solution (crispness): trace(S)/n_c 

        params:
            - g: an instance of cellrank GPCCA class.
            - ns: number of macrostates
        
        output: 
            - spectral_gap: float (or complex value) indicating the spectral gap for g
            - minChi: float indicating the minChi for g 
            - crispness: float indicating the crispness for g 
    """
    spectral_gap = _compute_spectral_gap(g.eigendecomposition['D'], ns)
    minChi = np.min(g.macrostates_memberships.X)
    crispness = g._gpcca.crispness_values[0]
    return spectral_gap, minChi, crispness


def _plot_quality_gpcca(df: pd.DataFrame, **kwargs):
    """
        Scatter plot of minChi against crispness:
        params:
            - df: pd.DataFrame containing minChi and crispness values.
    """
    fig = plt.figure()
    plt.scatter(df['minChi'], df['crispness'])
    title = "GPCCA macrostates quality" 
    title = title + "_" +str(kwargs['k']) + "K" if 'k' in kwargs.keys() else title
    title = title + " " +str(kwargs['pc']) + "PC" if 'pc' in kwargs.keys() else title
    title = title + " " +str(kwargs['lsi']) + "LSI" if 'lsi' in kwargs.keys() else title
    title = title + " " +str(kwargs['res']) + "res" if 'res' in kwargs.keys() else title

    path = os.path.join(os.getcwd() , 'figures', title.replace(" ", "").lower() + '.png')

    plt.title(title)
    plt.xlabel("minChi")
    plt.ylabel("crispness")
    for i in df.index:
        plt.annotate(i, (df['minChi'].loc[i], df['crispness'].loc[i]))
    fig.savefig(path)
