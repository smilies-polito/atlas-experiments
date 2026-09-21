########################################################################
# This code visualized gene expression on the E18 Mouse Brain data     #
# Plots can be visualized in Supp Figure 8                             #
########################################################################

import os
import argparse
import muon as mu
import scanpy as sc 

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--knn_rna", type=int, default = 15)
    parser.add_argument("--knn_act", type=int, default = 15)
    parser.add_argument("--n_pcs_rna", type=int, default = 20)
    parser.add_argument("--n_pcs_act", type=int, default = 10)
    parser.add_argument("--wnn", default = None)
    args = parser.parse_args()

    code = f"{args.n_pcs_rna}:{args.n_pcs_act}_{args.knn_rna}:{args.knn_act}:{args.wnn}"
    data_path = os.path.join(os.getcwd(), "output", "embryonic_mouse_brain", f"{code}.h5mu")
    data = mu.read_h5mu(data_path)

    data["rna"].obsm["X_umap"] = data.obsm["X_umap"]
    markers = {"RG": ["Vim", "Sox2", "Nes"], 
                "Astro": ["Gfap", "Aqp4", "Olig2"],
                "nIPC": ["Eomes"],
                "mGPC":["Efgr"],
                "Cyc. Prog.":["Top2a"]}
    for cluster, mlist in markers.items():
        for gene in mlist:
            if gene in data["rna"].var_names:
                title = f"{cluster}:{gene}"
                sc.pl.umap(data["rna"], color=gene, title=title, save = f"_{gene}.png")
        
