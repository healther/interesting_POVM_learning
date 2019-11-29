import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import sys

targetstates = {
                 0: 0.083333,
                 1: 0.083333,
                 2: 0.083333,
                 3: 0.083333,
                 4: 0.111111,
                 5: 0.027778,
                 6: 0.027778,
                 7: 0.083333,
                 8: 0.027778,
                 9: 0.111111,
                 10: 0.027778,
                 11: 0.083333,
                 12: 0.027778,
                 13: 0.027778,
                 14: 0.111111,
                 }

ts = np.zeros(16)
for k, v in targetstates.items():
    i1 = k % 4
    i2 = k // 4
    ts[k + 1] = v
print(ts)

Qz = [3., -1., -1., -1.]
Qx = [0., 2. * np.sqrt(2.), - np.sqrt(2.), - np.sqrt(2.)]

def magz_eval(S, Qz):
    mag1z = 0.
    mag2z = 0.

    for i in range(4):
        for k in range(16):
            if k%4==i:
                mag1z += Qz[i] * S[k]
            if k//4==i:
                mag2z += Qz[i] * S[k]

    return (mag1z, mag2z)

def corrz_eval(S, Qz):
    corrz = 0.
    for l in range(16):
        corrz += Qz[l//4] * Qz[l%4] * S[l]

    return corrz


def main():
    folder = sys.argv[1]

    mag1z = []
    mag2z = []
    mag1x = []
    mag2x = []
    corrz = []
    corrx = []
    for i in range(100000):
        try:
            data = np.load(folder + '/states_{:05d}.txt'.format(i))
            data = data.flatten()[0]
            vstates = np.zeros(2**4)
            for j, d in data.items():
                vstates[(j+1)%16] += d
            vs = vstates / vstates.sum()

            m1z, m2z = magz_eval(vs, Qz)
            m1x, m2x = magz_eval(vs, Qx)
            corz = corrz_eval(vs, Qz)
            corx = corrz_eval(vs, Qx)
            mag1z.append(m1z)
            mag2z.append(m2z)
            mag1x.append(m1x)
            mag2x.append(m2x)
            corrz.append(corz)
            corrx.append(corx)
        except:
            print(i)
            break

    dkls = np.loadtxt(folder + '/dkls.txt')[:, 1]
    mag1z = np.array(mag1z)
    mag2z = np.array(mag2z)
    mag1x = np.array(mag1x)
    mag2x = np.array(mag2x)
    corrz = np.array(corrz)
    corrx = np.array(corrx)
    np.save('quantumness.npy', {'mag1z': mag1z, 'mag2z': mag2z, 'mag1x': mag1x, 'mag2x': mag2x, 'corrz': corrz, 'corrx': corrx, 'dkls': dkls})

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    ax2.plot(dkls, label='dkls')
    ax.plot(mag1z, label='mag1z')
    ax.plot(mag2z, label='mag2z')
    ax.plot(mag1x, label='mag1x')
    ax.plot(mag2x, label='mag2x')
    ax.plot(corrz, label='corrz')
    ax.plot(corrx, label='corrx')
    ax.plot(np.sqrt(2) * (corrx-corrz), label='quantumness')
    ax.legend()
    ax.set_xlabel('update')
    ax.set_ylabel('')
    ax2.set_ylabel('dkl')
    ax2.set_yscale('log')

    plt.savefig('quantumness.pdf')


if __name__=='__main__':
    main()
