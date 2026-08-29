########################################################################
# This code visualized gene expression on the Human Fatel Brain data   #
# Plots can be visualized in Supp Figure 9                             #
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
    prefix = "non_removed" 

    data_dir = os.path.join(os.getcwd(), "output", "human_brain")
    data_path = os.path.join(data_dir, f"{prefix}_{code}.h5mu")
    data = mu.read_h5mu(data_path)

    data["rna"].obsm["X_umap"] = data.obsm["X_umap"]
    markers = {"RG": ["VIM", "SOX2", "NES"], 
                "Astro": ["GFAP", "AQP4", "OLIG2"],
                "nIPC": ["EOMES"],
                "mGPC":["EGFR"],
                "Cyc. Prog.":["TOP2A"]}
    for cluster, mlist in markers.items():
        for gene in mlist:
            if gene in data["rna"].var_names:
                title = f"{cluster}:{gene}"
                sc.pl.umap(data["rna"], color=gene, title=title, save = f"_human_brain_{gene}.png")
        
