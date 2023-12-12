import os
from itertools import product


Ks = [10,30,63,100]
PCs = [10,20,30]
grid = product(Ks, PCs)

for k, pc in grid:
    os.system(f"python 10xMouse_KPC.py 10xMouse_preprocessed.h5ad {k} {pc}")