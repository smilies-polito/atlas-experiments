import os
import argparse
import pandas as pd
import numpy as np
import muon as mu
from scipy import sparse

if __name__=="__main__":
    working_dir = os.getcwd()
    data_dir = os.path.join(working_dir, "output", "embryonic_mouse_brain")
    output_dir = os.path.join(data_dir, f"vitae_input")
    os.makedirs(output_dir, exist_ok=True)
    layer="log1p"

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pcs_rna", type=int, default=20)
    parser.add_argument("--n_pcs_act", type=int, default=10)
    parser.add_argument("--knn_rna", type=int, default=15)
    parser.add_argument("--knn_act", type=int, default=15)
    parser.add_argument("--wnn", default=None)
    args = parser.parse_args()
    code = f"{args.n_pcs_rna}:{args.n_pcs_act}_{args.knn_rna}:{args.knn_act}:{args.wnn}"

    mudata = mu.read_h5mu(os.path.join(data_dir, f"{code}.h5mu"))
    glob = mudata.obs
    for mod in ["rna", "activity"]:
        a = mudata[mod]
        X = sparse.csr_matrix(a.X, dtype=np.float32)
        sparse.save_npz(os.path.join(output_dir, f"{mod}_X.npz"), X)
        glob.reindex(a.obs_names).to_csv(os.path.join(output_dir, f"{mod}_obs.csv"))
        a.var.to_csv(os.path.join(output_dir, f"{mod}_var.csv"))
