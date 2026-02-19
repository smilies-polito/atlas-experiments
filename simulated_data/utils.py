import os
import muon as mu
import numpy as np
import pandas as pd
from muon import MuData
from anndata import AnnData
from typing import Literal, Union
from itertools import product

POTENCY_DICT = {"three_branches": {"4_1": "committed",
								"5_2": "committed",
								"5_3": "committed",
								"4_5": "totipotent"},
				"five_branches": {"6_7": "totipotent",
								"6_1": "committed",
								"7_9": "multipotent",	
								"7_8": "multipotent",
								"8_2": "committed",
								"8_3": "committed",
								"9_4": "committed",
								"9_5": "committed"}
				}

TERM_DICT = {"three_branches": ["4_1", "5_2", "5_3"],
			"five_branches": ["9_4", "9_5", "8_2", "8_3", "6_1"]}

PHYLA5 = pd.DataFrame([[1,0,0,0,0],[0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0],
			 [0,0,0,0,1], [1,1,1,1,1], [0,1,1,0,0], [0,0,0,1,1]],
			columns=["6_1", "8_2", "8_3", "9_4", "9_5"],
			index=["6_1", "8_2", "8_3", "9_4", "9_5", "6_7", "7_8", "7_9"])


PHYLA3 = pd.DataFrame([[1, 1, 1], [1,0,0], [0,1,0], [0,0,1]],
			index = ["4_5", "4_1", "5_2", "5_3"],
			columns = ["4_1", "5_2", "5_3"])

DEV_DICT = {"three_branches": PHYLA3, 
		"five_branches": PHYLA5}

				
def truth_like_fates(pseudotime: pd.Series, 
			membership: pd.Series, 
			tree:Literal["three_branches", "five_branches"]="three_branches",
			alpha: float = 2.0)-> pd.DataFrame:

	development = DEV_DICT[tree]
	accessibility = development.loc[membership.values].values
	W = pd.DataFrame(accessibility,
			index = pseudotime.index,
			columns = development.columns)
	row_sum = W.sum(axis=1)
	if np.any(row_sum==0):
		raise ValueError("Some cells have no accessible terminal states")
	return  W.divide(row_sum, axis=0)
		

def initial_macrostate(mudata:Union[AnnData, MuData], pseudotime_key: str, n_cells: int) -> list:
	pseudotime = mudata.obs[pseudotime_key]
	return pseudotime.nsmallest(n_cells).index.tolist()

def terminal_macrostate(mudata: Union[AnnData, MuData],
			pseudotime_key: str,
			cluster_key: str,
			terminal_state: str,
			n_cells: int) -> list:
	pseudotime = mudata[mudata.obs[cluster_key] == terminal_state].obs[pseudotime_key]
	return pseudotime.nlargest(n_cells).index.tolist()
	
