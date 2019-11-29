import multiprocessing as mp
import numpy as np
import sys

import run

def run_main(arg):
    print(arg)
    run.main(arg[0], arg[1], arg[2], arg[3], arg[4])


learningrates = np.geomspace(0.01, 1.0, 10)
scales = np.geomspace(0.1, 1.0, 3)
nhidden = [3, 5, 7]
nsamples = [1000, 10000, 100000]
rseeds = [424242424, 42424243]
args = []

for rs in rseeds:
    for lr in learningrates:
        for s in scales:
            for n in nhidden:
                for ns in nsamples:
                    args.append((n, s, lr, ns, rs))


nprocs = 20
if len(sys.argv) == 2:
    i = int(sys.argv[1])
    args = args[i*nprocs:(i+1)*nprocs]

pool = mp.Pool(nprocs)
pool.map(run_main, args)
pool.close()
pool.join()
