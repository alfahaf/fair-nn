# Convert the pickled datasets into txt files used by the C++ code
import numpy as np
import pickle
import sys


fn = sys.argv[1]
print(fn)
with open(fn, 'rb') as f:
    data, queries, _, _ = pickle.load(f)

gn = fn.split(".")[0]

np.savetxt(f"{gn}-data.txt", data)
np.savetxt(f"{gn}-queries.txt", queries)



