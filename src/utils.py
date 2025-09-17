import json
import requests
import pandas as pd 
from tqdm import tqdm
from muon import MuData
from typing import Optional, Union
from anndata import AnnData
from muon import atac as ac 
from scipy.spatial.distance import cdist


def search_cells(data: MuData, embedding_key:str="X_umap", grouping_key:str="celltype", n_select:int=1) -> dict:
	centroids = compute_centroids(data, embedding_key=embedding_key, grouping_key=grouping_key)
	nearest_cells = find_nearest_cells(data, centroids, embedding_key = embedding_key, grouping_key = grouping_key, n_select=n_select)
	return nearest_cells


def compute_centroids(data: MuData, embedding_key: str="X_umap", grouping_key:str="celltype") -> pd.DataFrame:
	if not embedding_key in data.obsm:
		raise KeyError(f"{embedding_key} not in data.obsm")
	if not grouping_key in data.obs:
		raise KeyError(f"{grouping_key} not in data.obs")

	embedding = data.obsm[embedding_key]
	columns = [f"col{n}" for n in range(embedding.shape[1])]

	df = pd.DataFrame(embedding, index=data.obs_names, columns=columns)
	df[grouping_key] = data.obs[grouping_key]
	
	centroids = df.groupby(by=grouping_key).mean()
	return centroids


def find_nearest_cells(data: MuData, centroids:pd.DataFrame, embedding_key: str="X_umap", grouping_key:str="celltype", n_select:int=30) -> dict:
	if not embedding_key in data.obsm:
		raise KeyError(f"{embedding_key} not in data.obsm")
	if not grouping_key in data.obs:
		raise KeyError(f"{grouping_key} not in data.obs")

	embedding = data.obsm[embedding_key]	

	nearest_cells = {}
	
	for group in centroids.index:
		mask = data.obs[grouping_key] == group
		group_coordinates = embedding[mask.values, :]
		group_observations = data.obs_names[mask.values]
		group_centroid = centroids.loc[group].values.reshape(1, -1)
		dists = cdist(group_coordinates, group_centroid).flatten()
		top_indices = np.argsort(dists)[:n_select]
		nearest_cells[group] = group_observations[top_indices].tolist()

	return nearest_cells 


def get_state_lineage_tracing(x:str) -> str:
	mapping = {"initial": ["HSC", "Refined.HSC"], 
		"erythroid": ["EryP"],
		"megakaryocyte": ["MKP"], 
		"monocyte": ["Mono"],
		"NK": ["NK"],
		"B": ["B", "Plasma"],
		"dendritic" : ["cDC", "pDC"],
		"T": ["CD4", "CD8"], 
		"intermedate": ["MDP", "GMP", "CMP", "MEP", "MPP", "LMPP", "CLP", "ProB"]}
	for k,v in mapping.items():
		if x in v:
			return k
	return None

def intermediate_probability(frequencies, reachable_lineages, cell):
	'''
		Function that identifies the lineages involved in the differentiation process for intermediate progenitor cells.
	'''
	clade = cell["ClonalGroup"]
	celltype = cell["STD.CellType"]
	reachable = reachable_lineages.get(celltype , []) # get the lineages the cell cen develop into
	clade_row = frequencies[(frequencies.clade == clade)].drop(columns="clade") # identifies the cells belonging to the clade
	mask = [col for col in clade_row if col not in reachable] # identifies the lineages 
	clade_row.loc[:, mask] = 0 # sets to 0 the lineages the cell cannot reach 
	totals = clade_row.sum(axis=1).values[0] # normalizes the other values
	clade_row /= totals 
	return clade_row.iloc[0].copy() 

def process_coordinates(response):
	results = []
	if not response.ok:
		return results
	data = response.json()
	print(len(data.items()))
	for symbol,info in data.items():
		results.append(( symbol, info.get("seq_region_name", None),
					info.get("start", None),
					info.get("end", None),
					info.get("strand", None)))
	return results

def retrieve_ensembl_coordinates(genes: list, offset:int =200, organism:str="mus_musculus", skip_chromosomes:Optional[list]=None) -> pd.DataFrame:
	server = "https://rest.ensembl.org"
	ext = f"/lookup/symbol/{organism}"
	headers = {"Content-Type": "application/json", "Accept": "application/json"}
	coordinates = []
	for i in tqdm(range(0, len(genes), offset), desc = "Extracting Ensembl Coordinates"):
		payload = json.dumps({ "symbols": genes[i: i+offset]})
		response = requests.post(server+ext, headers=headers, data=payload)
		coordinates = coordinates + process_coordinates(response)

	coordinates = pd.DataFrame(coordinates, columns = ["Symbol", "Chromosome", "Start", "End", "Strand"])
	mask = coordinates.isna().any(axis=1)
	coordinates = coordinates[~mask]
	coordinates["Strand"] = coordinates["Strand"].map(lambda strand: "-" if strand =="-1" else "+")
	if skip_chromosomes is not None:
		coordinates["Chromosome"] = coordinates["Chromosome"].map(lambda x: "chr"+x if x not in skip_chromosomes else x)
	coordinates = coordinates[~coordinates["Chromosome"].isin(skip_chromosomes)]
	coordinates = coordinates.astype({"Start":int, "End":int})
	return coordinates


def create_feature_map(features: pd.DataFrame, strand:bool = False) -> pd.DataFrame:
	columns = ["Chromosome", "Start", "End"]
	if strand:
		columns.append("Strand")
	features = features[columns]
	features = features.loc[~features.Start.isnull()]
	features.Start = features.Start.astype(int)
	features.End = features.End.astype(int) 
	return features

def assign_lineage(celltype: str) -> str:
	map_celltype_to_lineage = {"MEP": "erythroid", 
								"EryP": "erythroid", 
								"MKP": "megakaryocyte",
								"CMP": "intermediate", 
								"LMPP": "intermediate",
								"MPP": "intermediate",
								"MDP": "myeloid",
								"GMP": "myeloid",
								"cDC": "myeloid",
								"pDC": "myeloid",
								"Mono": "myeloid",
								"CLP": "lymphoid",
								"CD4": "lymphoid",
								"CD8": "lymphoid",
								"ProB": "lymphoid",
								"B": "lymphoid",
								"Plasma": "lymphoid",
								"NK": "lymphoid",
								"CBD": "lymphoid",
								"HSC": "hsc",
								"Refined.HSC": "hsc"}


def aggregate_lineage_fate(df:pd.DataFrame, terminal:dict):
		lineages = {"megakaryocyte": ["megakaryocyte"], "erythroid": ["erythroid"], "myeloid": ["dendritic","monocyte"], "lymphoid": ["B", "T", "NK"]}
		for lineage, cell_type in lineages.items():
			barcodes = [terminal[t] for t in cell_type]
			df[lineage] =  df[barcodes].sum(axis=1)

def process_single_atac_experiment(experiment: AnnData, fragment_path:str, features: pd.DataFrame) -> AnnData:
	ac.tl.locate_file(data=experiment, file=fragment_path, key="fragments")
	ac.tl.nucleosome_signal(experiment)
	ac.tl.tss_enrichment(experiment, features)
	return experiment


def compute_single_activity_experiment(experiment:AnnData, fragment_path: str, features: pd.DataFrame, stranded:bool=False) -> AnnData:
	ac.tl.locate_file(data=experiment, file=fragment_path, key ="fragments")
	activity = ac.tl.count_fragments_features(data=experiment, features=features, stranded=stranded)
	return activity 


def _check_keys(data: Union[AnnData, MuData], modality_key:Optional[str]=None, 
				embedding_key: Optional[str]=None, pseudo_time_key: Optional[str]=None,
				entropy_key: Optional[str]=None, fate_prob_key: Optional[str]=None, obs_key:Optional[str]=None):

	is_mudata = isinstance(data, MuData) and modality_key is None
	is_anndata = isinstance(data, AnnData)
	is_modality = isinstance(data, MuData) and modality_key is not None
	if is_modality and modality_key not in data.mod.keys():
		raise KeyError(f"{modality_key} not in data.mod.keys()")
	if (is_mudata or is_modality) and embedding_key is not None and embedding_key not in data.obsm.keys():
 		raise KeyError(f"{embedding_key} not in data.obsm") 
	if (is_mudata or is_anndata) and (pseudo_time_key is not None and pseudo_time_key not in data.obs.columns):
		raise KeyError(f"{pseudo_time_key} not in data.obs")
	if is_modality and (pseudo_time_key is not None and pseudo_time_key not in data[modality_key].obs.columns):
		raise KeyError(f"{pseudo_time_key} not in data.obs")
	if (is_mudata or is_anndata) and (fate_prob_key is not None and fate_prob_key not in data.obsm.keys()):
		raise KeyError(f"{fate_prob_key} not in data.obsm")
	if is_modality and (fate_prob_key is not None and fate_prob_key not in data[modality_key].obsm.keys()): 
		raise KeyError(f"{fate_prob_key} not in data.obsm") 
	if (is_mudata or is_anndata) and (entropy_key is not None and entropy_key not in data.obs.columns):
		raise KeyError(f"{entropy_key} not in data.obs")
	if is_modality and entropy_key is not None and entropy_key not in data[modality_key].obs.columns:
		raise KeyError(f"{entropy_key} not in data[{modality_key}].obs")
	if (is_anndata or is_mudata) and obs_key is not None and obs_key not in data.obs.columns:
		raise KeyError(f"{obs_key} not in data.obs")
	if is_modality and obs_key is not None and obs_key not in data[modality_key].obs.columns:
		raise KeyError(f"{obs_key} not in data[{modality_key}].obs")
