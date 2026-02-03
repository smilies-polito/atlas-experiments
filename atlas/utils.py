import matplotlib.pyplot as plt
from muon import MuData
from matplotlib.colors import to_hex


def _assign_state_colors(mudata:MuData,
						cmap: str = "tab20"):
	all_states = set()
	if "fate_state_colors" not in mudata.uns:
		mudata.uns["fate_state_colors"] = {}

	color_map = mudata.uns["fate_state_colors"]

	for key in ["terminal_states", "initial_states", "macrostates"]:
		states = mudata.uns.get(key, None)

		if isinstance(states, dict):
			all_states.update(states.keys())

	new_states = [s for s in  all_states if s not in color_map]
	if not new_states:
		return 

	cmap_obj = plt.get_cmap(cmap, len(new_states))
	for i, state in enumerate(sorted(new_states)):
		color_map[state] = to_hex(cmap_obj(i))

	
