import numpy as np

def _check_conjugate(idx, eig):
    if np.iscomplex(eig):
        print(f"Complex value needs it conjugate, returning {idx+2} states")
        return idx+2
    else: 
        return idx+1