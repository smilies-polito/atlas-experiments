import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from muon import MuData
from pygam import LinearGAM, s
from typing import Union
from matplotlib.colors import to_hex


def _assign_state_colors(mudata:MuData,
						cmap: str = "tab20"):
	all_states = set()
	if "fate_state_colors" not in mudata.uns:
		mudata.uns["fate_state_colors"] = {}

	color_map = mudata.uns["fate_state_colors"]

	for key in ["terminal_states", "initial_states", "intermediate_states"]:
		states = mudata.uns.get(key, None)

		if isinstance(states, dict):
			all_states.update(states.keys())

	new_states = [s for s in  all_states if s not in color_map]
	if not new_states:
		return 

	base_colors = plt.get_cmap(cmap).colors
	for state in sorted(new_states):
		idx = len(color_map)
		color = base_colors[idx % len(base_colors)]
		color_map[state] = to_hex(color)


def _weighted_quantile(x, w, q):
	idx = np.argsort(x)
	x_sorted = x[idx]
	w_sorted = w[idx]
	cw = np.cumsum(w_sorted)
	cw /= cw[-1]
	return np.interp(q, cw, x_sorted)

def _finite_difference(y, x, order:int=1):
	if order==1:
		return np.gradient(x,y)
	elif order==2:
		return np.gradient(np.gradient(x,y), x)
	else:
		raise ValueError("Only order 1 and 2 supported")


class MultiBranchGAM: 
	def __init__(self, 
				mudata: MuData, 	
				ptf: Union[str, int],
				gene: Union[str, int], 
				pseudotime_key: str = "pseudotime",
				fate_prob_key: str = "fate_probabilities",
				rna_modality: str = "rna", 
				activity_modality: str = "activity",
				n_splines: int = 8):
		self.mudata = mudata
		if ptf not in mudata.mod[rna_modality].var_names:
			raise KeyError(f"{ptf} not a valid pTF")
		self.ptf = ptf 
		if gene not in mudata.mod[activity_modality].var_names:
			raise KeyError(f"{gene} not a valid gene")
		self.gene = gene
		self.rna_mod = rna_modality
		self.activity_mod = activity_modality
		self.n_splines = n_splines
		if pseudotime_key not in self.mudata.obs.columns:
			raise KeyError(f"{pseudotime_key} not a valid key.")
		self.pseudotime_key = pseudotime_key
		if fate_prob_key not in self.mudata.obsm.keys():
			raise KeyError(f"{fate_prob_key} not a valid key.")
		self.fate_key = fate_prob_key 
		
		self.prepare_data()


	def prepare_data(self, eps:float = 1e-6):
		self.pseudotime = (self.mudata.obs[self.pseudotime_key].
							values.reshape(-1,1) )

		fate_df = self.mudata.obsm[self.fate_key]
		self.fate_prob = fate_df.values
		self.branch_names = fate_df.columns

		gex = self.mudata.mod[self.rna_mod][:, self.ptf].X.toarray().ravel()
		act = self.mudata.mod[self.activity_mod][:, self.gene].X.toarray().ravel()

		# standardization: comparsability among curves
		self.gex = (gex - gex.mean()) / (gex.std() + eps)
		self.act = (act - act.mean()) / (act.std() + eps)

	def fit(self):
		self.models = {}
		for i, branch in enumerate(self.branch_names):
			weights = self.fate_prob[:, i]
			eGam = LinearGAM( 
						s(0, n_splines = self.n_splines)
				).fit(self.pseudotime, self.gex, weights = weights)
			aGam = LinearGAM(
						s(0, n_splines = self.n_splines)
				).fit(self.pseudotime, self.act, weights = weights)
			self.models[branch] = {
				"weights": weights,
				"gam_exp": eGam,
				"gam_act": aGam
			}

	def predict(self, 
				n_points: int = 200,
				q_low: float = 0.02,
				q_high: float = 0.98):
		
		results = {}
		t = self.pseudotime.ravel()
		for branch, model in self.models.items():
			weights = model["weights"]
			t_min = _weighted_quantile(t, weights, q_low)
			t_max = _weighted_quantile(t, weights, q_high)

			t_grid = np.linspace(t_min, t_max, n_points).reshape(-1,1)
			results[branch] = {
					"t_grid" : t_grid.ravel(),
					"gex": model["gam_exp"].predict(t_grid),
					"act": model["gam_act"].predict(t_grid),
					"t_min": t_min,
					"t_max": t_max
			}
		self.predictions = results 
		
	def derivative(self,
					n_points: int= 200,
					q_low: float = 0.02,			
					q_high: float = 0.98,
					order: int = 1):
		results = {}
		t = self.pseudotime.ravel()
		for branch, pred in self.predictions.items():
			t = pred["t_grid"]
			results[branch] = {
					"t_grid": t,
					"gex_deriv": _finite_difference(pred["gex"], t, order),
					"act_deriv": _finite_difference(pred["act"], t, order)
			}
		self.derivatives = results
				



	
		
		
	



			
	




