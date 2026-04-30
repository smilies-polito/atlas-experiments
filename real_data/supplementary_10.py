import os
import muon as mu
import scanpy as sc 

if __name__=="__main__":
    data_path = os.path.join(os.getcwd(), "output", "human_brain", "brain.h5mu")
    data = mu.read_h5mu(data_path)

    data["rna"].obsm["X_umap"] = data.obsm["X_umap"]
    markers = {"RG": ["VIM", "SOX2", "NES"], 
                "Astro": ["GFAP", "AQP4", "OLIG2"],
                "nIPC": ["EOMES"],
                "mGPC":["EGFR"],
                "Cyc. Prog.":["TOP2A"]}
    for cluster, mlist in markers.items():
        for gene in mlist:
            if gene in data["rna"].var_names():
                title = f"{cluster}:{gene}"
                sc.pl.umap(data["rna", color=gene, title=title, save = f"_{gene}.png")
        
