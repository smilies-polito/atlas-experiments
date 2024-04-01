import numpy as np

def _check_conjugate(idx, eig):
    if np.iscomplex(eig):
        print(f"Complex value needs it conjugate, returning {idx+2} states")
        return idx+2
    else: 
        return idx+1
    
def _compute_spectral_gap(eigenvalues: np.ndarray):
    """
        Function that computed the spectral gap, thence the difference between lambda_n_c and lambda_n_c+1 
        params:
            - eigenvalues: array storing the eeigenvalues for GPCCA 
        output: 
            - spectral gap: float (or complex value) indicating the spectral gap 
    """
    return eigenvalues[0] - eigenvalues[1]


def _check_macrostate_quality(g):
    """
        Function that computes n_c quality according to DOI 10.1007/s11634-013-0134-6:
        Criteria are: 
        1. spectral gap
        2. minChi criterion: min_i min_j X(i,j) with X the membership matrix
        3. Optimality of PCCA+ solution (crispness): trace(S)/n_c 

        params:
            - g: an instance of cellrank GPCCA class.
        
        output: 
            - spectral_gap: float (or complex value) indicating the spectral gap for g
            - minChi: float indicating the minChi for g 
            - crispness: float indicating the crispness for g 
    """
    spectral_gap = _compute_spectral_gap(g.eigendecomposition['D'])
    minChi = np.min(g.macrostates_memberships.X)
    crispness = g._gpcca.crispness_values[0]
    return spectral_gap, minChi, crispness

