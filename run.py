import sys
import os
import subprocess
import json
import yaml
import numpy as np
from collections import Counter, defaultdict
import datetime


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


def run_once(weight, bias, initialstate, rseedoffset=0):
    outdict = {}
    outdict["Config"] = {
                         "synapseType": "exp",
                         "nupdates": 1000000,
                         "randomSeed": 42424242+rseedoffset, "randomSkip": 1000000,
                         "tauref": 100,
                         "tausyn": 100,
                         "tauref": 1,
                         "tausyn": 1,
                         "output": {"outputScheme": "BinaryState", "outputEnv": False},
                         "subsampling": 100,
                         "delay": 1, "networkUpdateScheme": "InOrder"}
    outdict["bias"] = bias.tolist()
    outdict["initialstate"] = initialstate.astype(int).tolist()
    outdict["temperature"] = {"type": "Const", "times": [0, 10000000], "values": [1., 1.]}
    outdict["outfile"] = "out"
    outdict["externalCurrent"] = {"type": "Const", "times": [0, 10000000], "values": [0., 0.]}
    outdict["weight"] = []
    for i, wline in enumerate(weight):
        for j, w in enumerate(wline):
            if w == 0:
                continue
            outdict["weight"].append([i, j, float(np.real(w))])

    yaml.dump(outdict, open('sim.json', 'w'))

    subprocess.call(['/home/hd/hd_hd/hd_wv385/git/neuralsampling/neuralsampler/bin/neuralsampler', 'sim.json'])


def get_updates(states, n, m):
    updates = {'w': np.zeros((n, m)), 'b': np.zeros(n+m)}
    for istate in states:
        state = np.array([np.bitwise_and(2**i, istate)!=0 for i in range(n+m)])
        updates['w'] += np.outer(state[:n], state[n:]) / float(len(states))
        updates['b'] += state / float(len(states))

    return updates


def get_model_dkl(modelstates, nhidden):
    origc = Counter(modelstates)
    relstates = defaultdict(float)
    for k, v in origc.items():
        ind = k % 16 # 2**nvisible
        # ind = int(k * 2**-nhidden)
        relstates[ind] += v
    dkl = 0.
    outstring = ""
    for ts, p in targetstates.items():
        outstring += "{} {} {}\n".format(ts, p, relstates[ts]/float(sum(relstates.values())))
        if relstates[ts] > 0:
            dkl += p*np.log(p/relstates[ts]*sum(relstates.values()))
        else:
            print("Didn't sample state {}".format(ts))
    print(outstring)
    return dkl


def get_states(n, m):
    states = []
    with open('out', 'r') as f:
        for i, line in enumerate(f):
            if i<4:
                continue
            try:
                state = sum(int(s)*2**j for j, s in enumerate(line[:n+m]))
                states.append(state)
            except ValueError:
                break
    return states


def initialstate(state=None, nvisible=4, nhidden=5):
    if state is not None:
        istate = np.zeros_like(state)
        istate[state>100.] = np.random.randint(0, 100, size=sum(state>100.))
        istate[state<100.] = np.random.randint(100, 200, size=sum(state<100.))
        istate[nvisible:] = np.random.randint(0, 200, size=nhidden)
        return istate
    else:
        return np.random.randint(0, 200, size=nhidden+nvisible)



def main(nhidden=3, scale=0.3, learningrate=0.1, rseed=42424242):
    n = 4
    m = nhidden
    nupdates = 100000
    np.random.seed(rseed)

    wdir = 'sims/{}_{}_{}_{}'.format(m, scale, learningrate, rseed)
    try:
        os.mkdir(wdir)
    except OSError:
        pass
    os.chdir(wdir)

    b = np.random.normal(0, scale, size=n+m)
    w = np.zeros((n+m, n+m))
    w[n:,:n] = np.random.normal(0, scale, size=(m,n))
    w += w.T

    try:
        w = np.load('weights.txt')
        b = np.load('bias.txt')
    except:
        pass

    with open('dkls.txt', 'a', buffering=0) as dklfile:
        for i in range(nupdates):
            modelstates = []
            for _ in range(4):
                run_once(w, b, initialstate(nvisible=n, nhidden=m), rseedoffset=i+int(np.random.randint(100)))
                modelstates += get_states(n, m)

            update = get_updates(modelstates, n, m)
            for istate, p in targetstates.items():
                state = np.zeros(n+m)
                state[:n] = [-500+1000*(np.bitwise_and(2**j, istate)!=0) for j in range(n)]
                state[-m:] = 0
                run_once(w, b+state, initialstate(state, n, m), rseedoffset=i+int(np.random.randint(100)))
                clampedstates = get_states(n, m)
                stateupdate = get_updates(clampedstates, n, m)
                update['w'] -= stateupdate['w'] * p
                update['b'] -= stateupdate['b'] * p

            b         -= learningrate * update['b']
            w[n:, :n] -= learningrate * update['w'].T
            w[:n, n:] -= learningrate * update['w']

            dkl = get_model_dkl(modelstates, m)
            dklfile.write('{:05d} {}\n'.format(i, dkl))
            print("{} Update {}: DKL {}".format(datetime.datetime.now(), i, dkl))

            with open('states_{:05d}.txt'.format(i), 'w') as f:
                np.save(f, Counter(modelstates))
            with open('weights_{:05d}.txt'.format(i), 'w') as f:
                np.save(f, w)
            with open('bias_{:05d}.txt'.format(i), 'w') as f:
                np.save(f, b)
            with open('weights.txt', 'w') as f:
                np.save(f, w)
            with open('bias.txt', 'w') as f:
                np.save(f, b)

if __name__=="__main__":
    if len(sys.argv) == 5:
        nhidden = int(sys.argv[1])
        scale = float(sys.argv[2])
        learingrate = float(sys.argv[3])
        rseed = int(sys.argv[4])
        main(nhidden, scale, learningrate, rseed)
    else:
        print("Using default values")
        main()
