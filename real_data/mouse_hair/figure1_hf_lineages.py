########################################################################
# This code applies ATLAS multi-omic visualisation on the TI from      #
# the SHARE-seq Mouse Hair Data                                        # 
# Plots can be visualized in Figure 1                                  #
########################################################################
import os
import atlas
import argparse
import numpy as np
import muon as mu
from muon import MuData

def _infer_organism(path:str):
    if "mouse_brain" in path:
        return "embryonic_mouse_brain"
    else:
        return "mouse_skin"

def _get_strategy(path:str):
    if "cellrank" in path:
        return "cellrank"
    else:
        return "palantir"

def plot_regulation(mudata: MuData,
                        regs: dict,
                        strategy:str):

    for ptf, genes in regs.items():
        save = f"{strategy}_{ptf}"
        try:
            atlas.pl.trends(mudata = mudata,
                            ptf = ptf,
                            genes = genes, 
                            time_key = "pseudotime",
                            fate_probability_key = "fate_probabilities",
                            show= False,
                            save = save)
        except Exception as e:
            print("Plot Trends for {strategy} and ptf = {ptf} got Exception\n{e}\n")
                

REGULATION = {"mouse_skin": {"Lef1": ["Dlx3", "Jag1", "Hoxc13", "Notch1", "Prdm1", "Krt71", "Foxq1"],
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
    parser.add_argument("--h5mu", type=str)
    args = parser.parse_args()

    file_path = os.path.join(os.getcwd(), args.h5mu)
    mudata = mu.read_h5mu(file_path)
    
    regs = REGULATION[_infer_organism(args.h5mu)]
    strategy = _get_strategy(args.h5mu)

    plot_regulation(mudata = mudata, regs = regs, strategy = strategy)
