import numpy as np

def _check_conjugate(idx, eig):
    """
        Function supporting the identification of real and complex eigenvalues for the selection of the macrostates
        params:
            idx: integer indicating the position of the eigenvalue. If eigenvalue is real then need to compute idx+1 macrostates; if eigenvalue is  complex, then returns idx+2 to get the conjugate.
            eig: eigenvalue.
    """
    if np.iscomplex(eig):
        print(f"Complex value needs it conjugate, returning {idx+2} states")
        return idx+2
    else: 
        return idx+1
