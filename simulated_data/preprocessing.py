import os
import atlas
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from muon import MuData
from scipy.sparse import csr_matrix

if __name__=="__main__":
    seed = 42
    working_directory = os.getcwd() # set path to repository 
    np.random.seed(seed)

    parser = argparse.ArgumentParser() 
    parser.add_argument("--tree", type=str)
    parser.add_argument("--rd", type=float)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--knn_rna", type=int)
    parser.add_argument("--knn_act", type=int)
    parser.add_argument("--n_pcs_rna", type=int, default=20)
    parser.add_argument("--n_pcs_act", type=int, default=10)
    parser.add_argument("--wnn", default=None)
    args = parser.parse_args()

    # DATA CONSTRUCTION 
    tree = args.tree
    rd, sigma = args.rd, args.sigma
    knn_rna, knn_act= args.knn_rna, args.knn_act
    n_pcs_rna, n_pcs_act = args.n_pcs_rna, args.n_pcs_act

    # PATH DECLARATION
    data_path = os.path.join(working_directory, "data", "simulated_data", tree)
    saving_simulation_path = os.path.join(working_directory, "output", "simulations", "mudata")
    activity = pd.read_csv(os.path.join(data_path, f"{rd}_{sigma}_activity.tsv"), sep="\t", header=0, index_col=0)
    spliced = pd.read_csv(os.path.join(data_path, f"{rd}_{sigma}_spliced.tsv"), sep="\t", header=0, index_col=0)
    metadata = pd.read_csv(os.path.join(data_path, f"{rd}_{sigma}_metadata.tsv"), sep="\t", header=0, index_col=0)

    # create activity matrix
    activity = AnnData(X=csr_matrix(activity.values), 
            obs = pd.DataFrame(data=None, index=activity.index.values, columns=None), 
            var = pd.DataFrame(data=None, index=activity.columns, columns=None))    
    sc.pp.normalize_total(activity)
    sc.pp.pca(activity, random_state=seed, use_highly_variable=False)
    # create rna matrix
    rna = AnnData(X=csr_matrix(spliced.values.T), 
            obs=pd.DataFrame(data=None, index=spliced.columns, columns=None), 
            var=pd.DataFrame(data=None, index=spliced.index.values, columns=None))
    rna.obs = rna.obs.merge(metadata, how="left", left_index=True, right_index=True)
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.pca(rna, random_state=seed, use_highly_variable=False)

    mudata = MuData({"rna":rna, "activity":activity})

    # PREPROCESSING 
    knn = {"rna": knn_rna, "activity": knn_act}
    n_pcs = {"rna": n_pcs_rna, "activity": n_pcs_act}
    modalities = ("rna", "activity")
    use_rep = "X_pca"
    n_neighbors = None if (args.wnn is None or args.wnn == "None") else int(args.wnn) 
    code = f"{tree}_{rd}:{sigma}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{n_neighbors}"

    atlas.pp.wnn(mudata = mudata,
                        knn = knn,
                        n_pcs = n_pcs,
                        use_rep = use_rep,
                        modalities = modalities,
                        n_neighbors=n_neighbors,
                        random_state=seed)

    atlas.tl.umap(mudata=mudata,
                    random_state = seed,
                    neighbors_key="wnn")


    if not os.path.exists(saving_simulation_path):
        os.mkdir(saving_simulation_path) 

    mudata.write(os.path.join(saving_simulation_path,
                                f"{code}.h5mu")
                )
                    



