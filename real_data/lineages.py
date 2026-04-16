import os
import atlas
import numpy as np
import muon as mu
from typing import Literal
from muon import MuData

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

    organism = "mouse_skin"
    strategy = "pseudotime-kernel"
    output_dir = os.path.join("output", organism)
    data_path = os.path.join(output_dir, "20:10_15:15:None_hard.h5mu")
    mudata = mu.read_h5mu(data_path) 
    
    regs = REGULATION[organism]

    plot_regulation(mudata = mudata, regs = regs, strategy = strategy)
