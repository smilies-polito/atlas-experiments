import os
import atlas
import argparse
import numpy as np
import muon as mu
from typing import Literal
from muon import MuData

def _infer_organism(path:str):
    if "mouse_brain" in path:
        return "embryonic_mouse_brain"
    else:
        return "mouse_skin"

def plot_regulation(mudata: MuData,
                        regs: dict,
                        strategy: Literal["pseudotime-kernel","palantir"]):

    if strategy == "pseudotime-kernel":
        mudata.uns["fate_state_colors"] = mudata.uns["pseudotimeK_fate_colors"]
        mudata.obsm["fate_probabilities"] = mudata.obsm["pseudotimeK_fate_probabilities"]
    elif strategy == "palantir":
        mudata.uns["fate_state_colors"] = mudata.uns["palantir_fate_colors"]
        mudata.obsm["fate_probabilities"] = mudata.obsm["palantir_probabilities"]
    else: 
        raise ValueError(f"Strategy {strategy} not a valid TI inference method")

    for ptf, genes in regs.items():
        for g in genes:
            try:
                atlas.pl.plot_trends(mudata = mudata,
                                        ptf = ptf,
                                        gene = g,
                                        time_key = "pseudotime",
                                        fate_probability_key = "fate_probabilities",
                                         save = f"{strategy}_{ptf}_{g}")
            except Exception as e:
                print(ptf, g, e)

                

REGULATION = {"mouse_skin": {"Lef1": ["Dlx3", "Jag1", "Hoxc13"],
                    "Dlx3": ["Wnt3", "Hoxc13"],
                    "Hoxc13": ["Foxn1", "Foxq1", "St14"],
                    "Gata3": ["Krt71"]},
                "embryonic_mouse_brain": {"Ascl1": ["Dll1", "Hes6", "Tubb3"],
                                    "Neurod1": ["Scrt1", "Meis2"],
                                    "Pax6": ["Dll1"],
                                    "Nfia": ["Gfap"],
                                     "Nfib": ["Gfap"]}
            }


if __name__=="__main__":
    seed = 42
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default = "palantir")
    parser.add_argument("--h5mu", type=str)
    args = parser.parse_args()
    strategy, data_path = args.strategy, args.h5mu

    mudata = mu.read_h5mu(data_path) 
    
    regs = REGULATION[_infer_organism(data_path)]

    plot_regulation(mudata = mudata, regs = regs, strategy = strategy)
