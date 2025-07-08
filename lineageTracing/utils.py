import pandas as pd
from anndata import AnnData
from muon import atac as ac 

def aggregate_lineage_fate(df:pd.DataFrame, terminal:dict):
		lineages = {"megakaryocyte": ["megakaryocyte"], "erythroid": ["erythroid"], "myeloid": ["dendritic","monocyte"], "lymphoid": ["B", "T", "NK"]}
		for lineage, cell_type in lineages.items():
			barcodes = [terminal[t] for t in cell_type]
			df[lineage] =  df[barcodes].sum(axis=1)

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
	return map_celltype_to_lineage.get(celltype, "")

def create_feature_map(features: pd.DataFrame, strand:bool = False) -> pd.DataFrame:
	columns = ["Chromosome", "Start", "End"]
	if strand:
		columns.append("Strand")
	features = features[columns]
	features = features.loc[~features.Start.isnull()]
	features.Start = features.Start.astype(int)
	features.End = features.End.astype(int) 
	return features

def process_single_atac_experiment(experiment: AnnData, fragment_path:str, features: pd.DataFrame) -> AnnData:
	ac.tl.locate_file(data=experiment, file=fragment_path, key="fragments")
	ac.tl.nucleosome_signal(experiment)
	ac.tl.tss_enrichment(experiment, features)
	return experiment


def compute_single_activity_experiment(experiment:AnnData, fragment_path: str, features: pd.DataFrame, stranded:bool=False) -> AnnData:
	ac.tl.locate_file(data=experiment, file=fragment_path, key ="fragments")
	activity = ac.tl.count_fragments_features(data=experiment, features=features, stranded=stranded)
	return activity 
	
