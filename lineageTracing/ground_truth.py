#####################################################
# Generate ground truth for lineage tracing dataset #
#####################################################

import os
import numpy as np
import pandas as pd
import muon as mu

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
	
def softmax(row):
	e_x = np.sum(row - np.max(row))
	return e_x/e_x.sum()
	

if __name__=="__main__":
	seed = 42
	np.random.seed(seed)

	# read dataset
	working_dir = ... # set to repository directory 
	donor = ... #either donor1 or donor2
	data_path = os.patth.join(working_dir, "data", "lineage_tracing", f"{donor}", "data.h5mu")
	saving_path = os.path.join(working_dir, "data", "lineage_tracing", f"{donor}", "branch_assignment.csv")
	data = mu.read_h5mu(data_path)
	threshold = 0.7
	epsilon = 1e-6 

	# Subset for clonal probability > threshold and assign lineage
	high_data = data.obs[(data.obs["ClonalGroup.Prob"] > threshold)].copy()

	# Remove progenitors lineage for ground truth computations
	lineages_to_be_removed = ["intermediate", "hsc"]
	subdata = high_data[~high_data.lineage.isin(lineages_to_be_removed)]
	
	# Compute lineage frequency across the dataset, i.e. the number of cells belonging to the lineage 
	global_frequency = subdata["lineage"].value_counts(normalize=True) 

	# Compute lineage frequency for every clade + normalize accounting for lineage size
	clade_frequency = subdata[["lineage", "ClonalGroup"]].groupby(["lineage", "ClonalGroup"]).size().reset_index()
	clade_frequency.columns= ["lineage", "clade", "frequency"]
	clade_frequency["adj"] = clade_frequency.apply(lambda row: row.frequency/global_frequency[row.lineage], axis=1)

	# DEVELOPMENTAL PROBABILITIES HSC CELLS 
	clade_frequency["probability"] = clade_frequency.groupby("clade")["adj"].transform(lambda x: x/x.sum())
	clade_frequency_pivot = clade_frequency.pivot(index="clade", columns = "lineage", values="probability").reset_index()

	probabilities_hsc = high_data[high_data.lineage == "hsc"][["ClonalGroup"]].reset_index(names="barcode").merge(clade_frequency_pivot, how="left", left_on="ClonalGroup", right_on = "clade")
	probabilities_hsc.drop("clade", axis=1, inplace=True)

	# DEVELOPMENTAL PROBABILITIES TERMINAL CELLS 
	probabilities_terminal = high_data[~high_data.lineage.isin(lineages_to_be_removed)][["ClonalGroup", "lineage"]]
	probabilities_terminal = pd.get_dummies(probabilities_terminal, columns = ["lineage"], dtype=float, prefix='', prefix_sep='').reset_index(names="barcode")

	# DEVELOPMENTAL PROBABILITIES INTERMEDIATE PROGENITORS 
	# identify which lineages can be reached for every intermediate
	reachable_lineages = {"CMP": ["erythroid", "myeloid", "megakaryocyte"], 
						  "MPP": ["erythroid", "myeloid", "megakaryocyte", "lymphoid"],
						  "LMPP": ["myeloid", "lymphoid"]}

	intermediate_cells = high_data[high_data.lineage=="intermediate"][["ClonalGroup", "STD.CellType"]].reset_index(names="barcode")
	clade_frequency_pivot = clade_frequency.pivot(index="clade", columns="lineage", values="adj").reset_index()
	probabilities_intermediate = intermediate_cells.apply(lambda x: intermediate_probability(clade_frequency_pivot, reachable_lineages, x), axis=1)
	probabilities_intermediate = pd.concat((intermediate_cells[["barcode", "ClonalGroup"]], probabilities_intermediate), axis=1)

	total_probs = pd.concat([probabilities_intermediate, probabilities_terminal, probabilities_hsc], axis=0).set_index("barcode")
	total_probs = total_probs.loc[high_data.index.tolist()]	
	total_probs.to_csv(saving_path, sep=",", header=True, index=True)


