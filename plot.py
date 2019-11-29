import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys


root = sys.argv[1]
for name in os.listdir(root):
    directory = os.path.join(root, name)
    if os.path.isdir(directory):
        try:
            with open(os.path.join(directory, 'dkls.txt')) as f:
                dkls = [float(line.strip().split()[1]) for line in f]
        except:
            raise
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(dkls)
        ax.set_xlabel('iteration')
        ax.set_ylabel('dkl')
        ax.set_yscale('log')
        plt.savefig(os.path.join('plots', directory + '_dkls.pdf'))
        plt.close()




