import os
import muon as mu
import scanpy as sc 

if __name__=="__main__":
    data_path = os.path.join(os.getcwd(), "output", "embryonic_mouse_brain", "20:10_15:15:None_hard_6states.h5mu")
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
        
