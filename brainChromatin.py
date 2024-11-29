#######################
# Brain Chromatin Preprocessing
# Data from https://github.com/GreenleafLab/brainchromatin/blob/main/links.txt
####################### 

import os
import anndata
import mudata
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
from mudata import MuData

os.chdir("...") #Set working directory 

data_dc1r3_r1= os.path.join(os.getcwd(), "dc1r3_r1")
data_dc2r2_r1= os.path.join(os.getcwd(), "dc2r2_r1")
data_dc2r2_r2= os.path.join(os.getcwd(), "dc2r2_r2")
metadata_path = os.path.join(os.getcwd(), "multiome_cell_metadata.txt")
cluster_path = os.path.join(os.getcwd(), "multiome_cluster_names.txt")
results_path= os.path.join(os.getcwd(), "results")
if not os.path.exists(results_path):
    os.mkdir(results_path)
    
dc1r3_r1 = mu.read_10x_h5(os.path.join(data_dc1r3_r1, "filtered_feature_bc_matrix.h5"))
dc2r2_r1 = mu.read_10x_h5(os.path.join(data_dc2r2_r1, "filtered_feature_bc_matrix.h5"))
dc2r2_r2 = mu.read_10x_h5(os.path.join(data_dc2r2_r2, "filtered_feature_bc_matrix.h5"))

def preprocess_names(multimodalData, experiment_string):
    for modality in ["rna", "atac"]:
        multimodalData[modality].obs_names = multimodalData[modality].obs_names.str.replace("-1","",regex=False)
        multimodalData[modality].obs_names = experiment_string + "_" + multimodalData[modality].obs_names
        if modality=="rna":
            multimodalData[modality].var_names_make_unique()
            multimodalData[modality].obs["experiment"] = experiment_string

    multimodalData.update_obs()

preprocess_names(dc1r3_r1, "hft_ctx_w21_dc1r3_r1")
preprocess_names(dc2r2_r1, "hft_ctx_w21_dc2r2_r1")
preprocess_names(dc2r2_r2, "hft_ctx_w21_dc2r2_r2")

merged_rna = anndata.concat([dc1r3_r1["rna"], dc2r2_r1["rna"], dc2r2_r2["rna"]]) 
merged_atac = anndata.concat([dc1r3_r1["atac"], dc2r2_r1["atac"], dc2r2_r2["atac"]], join="outer")

data = MuData({"rna":merged_rna,"atac":merged_atac})

metadata = pd.read_csv(metadata_path, sep="\t", header=0)
data = data[data.obs_names.isin(metadata["Cell.ID"])]

# TO DO: 
# 1. log normalize rna counts
# 2. tfidf atac counts
# 3. find var features in both 
# 4. pca and lsi 
# 5. compute nearest neighbor graph + umap
# 6. Add cluster ids
# 7. plot expression of genes of interest
# 8. read and subset spliced and unspliced counts
# 9. velocyto 

