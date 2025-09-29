import os
import json
import numpy as np
from src.metrics import compare_transition_matrices

if __name__=="__main__":
	path = os.path.join("/scvemo", "output", "e18_mouse_no_rescale")
	rna_path = os.path.join(path, "pseudotime_rna", "matrix_analysis.json")
	if os.path.exists(rna_path):
		with open(rna_path, "r") as f:
			rna = json.load(f)
			f.close()
	else:
		print("Rna not existing")
		exit()


	multiomics_path = os.path.join(path, "pseudotime_multiomics", "matrix_analysis.json")
	if os.path.exists(multiomics_path):
		with open(multiomics_path, "r") as f:
			multiomics = json.load(f)
			f.close()
	else:
		print("Multiomics not existing")
		exit()

	compare_transition_matrices(rna=rna, multiomics=multiomics)


	
