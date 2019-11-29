import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import os
import sys
from run import targetstates


statesfile = sys.argv[1]
states = np.load(statesfile).flatten()[0]

vstates = np.zeros(16)
for k, v in states.items():
    # states are encoded littleendian
    vstates[k % 16] += v

vstates /= vstates.sum()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(np.arange(16)+0.4, vstates, width=0.4)
ax.bar(targetstates.keys(), targetstates.values(), width=0.4)
ax.set_xlabel('visiblestate')
ax.set_ylabel('probability')
#ax.set_yscale('log')
plt.savefig('state.pdf')
plt.close()




