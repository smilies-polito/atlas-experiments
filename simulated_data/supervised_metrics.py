import os
import warnings
import numpy as np 
import pandas as pd
from muon import MuData
from scipy.stats import kendalltau
from scipy.spatial.distance import jensenshannon


def terminal_state_score(mudata:MuData,
            terminal_clusters: list,
            time_key: str = "pseudotime",
            cluster_key: str = "cluster"
            ) -> tuple:

    '''
    Terminal State Score Function. 
    
    Parameters
    ----------
    mudata
        Mudata containing the results from the trajectory inference.
    time_key
        Key in mudata.obs where pseudotime for all cells is stored.
    cluster_key
        Key in mudata.obs where cluster identifier for each cell is stored
    terminal_clusters
        List of clusters that are supposed to be terminal according 
        to scMultiSim differentiation tree

    Raises
    ------
    KeyError
        If either time_key or cluster_key is not present in mudata.obs

    Warns
    -----
    UserWarning
        If not terminal states are present

    Returns
    -------
    tuple containing
        
        - Terminal Timing Score (tts): float
        - Terminal Temporal Precision (ttp): float
        - Terminal State Recall (tsr): float
        - Terminal Temporal Concentration (ttc): float 
        - Terminal State Score (tss): float

    Notes
    -----
    If no terminal state is revcovered, then all metrics are 0.

    '''


    def _construct_terminal_df(ts: dict, p: pd.Series, m:pd.Series) -> pd.DataFrame:
        rows = []    
        for state, cells in ts.items():
            for cell in cells:
                rows.append({"terminal_state": state,
                            "cell": cell,
                            "pseudotime": p.loc[cell],
                            "cluster": m.loc[cell]})
        return pd.DataFrame(rows)

    if not time_key in mudata.obs:
        raise KeyError(f"{time_key} not in mudata.obs")
    if not cluster_key in mudata.obs:
        raise KeyError(f"{cluster_key} not in mudata.obs")

    pseudotime = mudata.obs[time_key]
    membership = mudata.obs[cluster_key]
    tau_min = pseudotime.min()

    terminal_states = mudata.uns["terminal_states"]
    terminal_df = _construct_terminal_df(
                terminal_states,
                pseudotime,
                membership,
            )
    if terminal_df.empty:
        warnings.warn("WARNING: no terminal states detected", stacklevel=2)
        return 0.0, 0.0, 0.0, 0.0, 0.0

    gt_tau = (pseudotime.groupby(membership).max().
            reindex(terminal_clusters, fill_value=tau_min))
    inferred_tau = (terminal_df[terminal_df["cluster"].isin(terminal_clusters)].
                    groupby("cluster")["pseudotime"].
                    median().
                    reindex(terminal_clusters, fill_value=tau_min)
            )
    tts_i =  1 - (gt_tau- inferred_tau) / (gt_tau - tau_min)
    tts = tts_i.mean()

    if terminal_df.empty:
        ttp = 0
    else:
        ttp = terminal_df["cluster"].isin(terminal_clusters).mean()

    if terminal_df.empty:
        ttr = 0
    else: 
        found, expected = set(terminal_df["cluster"]), set(terminal_clusters)
        missing = expected.difference(found) 
        tsr = 1 - (len(missing) / len(expected))

    df_term = terminal_df[terminal_df["cluster"].isin(terminal_clusters)]
    if df_term.empty:
        ttc = 0
    else:
        iqr = (df_term.groupby("cluster")["pseudotime"].quantile(0.75) - 
            df_term.groupby("cluster")["pseudotime"].quantile(0.25))
        iqr = iqr.reindex(terminal_clusters, fill_value=0.0)
        ttc = (1-iqr).mean()  
    
    overall = (tsr* tts * ttp * ttc) ** (1/4)
    
    return tts, ttp, tsr, ttc, overall



def js_distance(
        mudata: MuData,
        truth: pd.DataFrame, 
        fate_key: str = "fate_probabilities",
        base:float=np.e) -> pd.Series:
    '''
    Jensen Shannon Distance between inferred fate probabilities and ground truth.
    
    Parameters
    ----------
    mudata
        MuData object storing results from trajectory inference.
    truth:
        pandas.DataFrame storing ground truth fate probabilities. Should match shape of 
        mudata.obsm[fate_key].
    fate_key
        key in mudata.obsm where fate probabilities are stored as pandas.DataFrame.
    base:
        base for JSD computation.

    Raises
    ------
    KeyError
        If fate_key not in mudata.obsm.
    ValueError 
        If inferred fates and ground truth fates dataframes do not mach in shape 
        or columns (i.e., different fates), or if NaNs are present.
    
    Returns
    -------
    pandas.Series
        JSD for each cell

    '''

    if fate_key not in mudata.obsm:
        raise KeyError(f"{fate_key} not in mudata.obsm")
    pred = mudata.obsm[fate_key]

    if pred.shape!=truth.shape:
        raise ValueError(f"Instances not match {pred.shape} != {truth.shape}")
    if pred.isna().any().any() or truth.isna().any().any():
        raise ValueError(f"NaNs are not a valid input")
    if set(pred.columns) != set(truth.columns):
        raise ValueError(f"Fate columns do not match")
    pred = pred.loc[truth.index, truth.columns]    
    return pd.Series(jensenshannon(pred.values, truth.values, base=base, axis=1), index = pred.index)
    

        
def kendall_correlation(
        mudata: MuData,
        pred_time: str = "pseudotime",
        true_time: str = "true_pseudotime"
    ) -> tuple:

    '''
    Computed Kendall Correlation.

    Parameters
    ----------
    mudata
        MuData storing results from trajectory inference.
    pred_time
        Key in mudata.obs storing cell-wise predicted pseudotime.
    true_time
        Key in mudata.obs storing cell-wise ground truth pseudotime.

    Raises
    ------
    KeyError
        If pred_time or true_time are not in mudata.obs.

    Warns
    -----
    UserWarning 
        If less than 100 cells are present. In this case Kendall Tau might have high 
        variance as states in scipy.

    Returns
    -------
    tuple containing

    - kendall tau statistics.
    - kendall tau pvalue.

    Notes
    -----
    If NaNs are present returns np.nan

    '''
    
    if pred_time not in mudata.obs:
        raise KeyError("{pred_time} not in mudata.obs")
    if true_time not in mudata.obs:
        raise KeyError("{true_time} not in mudata.obs")

    truth = mudata.obs[true_time]
    pred = mudata.obs[pred_time]

    if len(truth) < 100:
        warnings.warn("Sample size < 100. Kendall's tau may have high variance")
    
    if truth.isna().any() or pred.isna().any():
        warnings.warn(f"NaNs are not a valid input")
        return np.nan, np.nan
    
    pred = pred.reindex(truth.index)
    res, pvalue = kendalltau(pred.values, truth.values, variant="b") 
    return res, pvalue


