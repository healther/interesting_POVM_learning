import sys
import os
import subprocess
import json
import yaml
import numpy as np
from collections import Counter, defaultdict
import datetime
import sbs

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

targetstates = {tuple(int(kk) for kk in np.binary_repr(k, 4)): v
                for k, v in targetstates.items()}


def get_model_dkl(modelstates, nhidden):
    relstates = defaultdict(float)
    for k, v in modelstates.items():
        ind = k[:4]
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


def setup_sbs(nneurons):
    sampler_config = sbs.db.SamplerConfiguration.load(
            "tutorial_calibration.json")

    bm = sbs.network.ThoroughBM(num_samplers=nneurons,
                                sim_name='nest',
                                sampler_config=sampler_config)

    bm.saturating_synapses_enabled = True
    bm.use_proper_tso = True
    return bm


def run_once(bm, w, b, duration=1e5, burn_in_time=1000., rseed=np.random.randint(2**32-1)):
    bm.biases_theo = b
    bm.weights_theo = w
    bm.gather_spikes(duration=duration, dt=0.1, burn_in_time=burn_in_time,
                     sim_setup_kwargs={'rng_seeds': [rseed]})

    return {k: v for k, v in np.ndenumerate(bm.dist_joint_sim)}


def get_updates(states, n, m):
    updates = {'w': np.zeros((n, m)), 'b': np.zeros(n+m)}
    for state, frequency in states.items():
        updates['w'] += frequency * np.outer(state[:n], state[n:])
        print(frequency, state)
        updates['b'] += frequency * np.array(state)

    return updates


def main(nhidden=3, scale=0.3, learningrate=0.1, nsamples=10000, rseed=42424242):
    n = 4
    m = nhidden
    nupdates = 100000
    np.random.seed(rseed)
    network = setup_sbs(nneurons=n+m)

    wdir = 'sims/{}_{}_{}_{}_{}'.format(m, scale, learningrate, nsamples, rseed)
    try:
        os.mkdir(wdir)
    except OSError:
        pass
    os.chdir(wdir)

    # get initial parameters, if not use normal distributed ones
    try:
        w = np.load('weights.txt')
        b = np.load('bias.txt')
    except:
        b = np.random.normal(0, scale, size=n+m)
        w = np.zeros((n+m, n+m))
        w[n:,:n] = np.random.normal(0, scale, size=(m, n))
        w += w.T

    network.weights_theo = w
    network.biases_theo = b

    with open('dkls.txt', 'a', buffering=0) as dklfile:
        for i in range(nupdates):
            # using 4x samples for the free run
            modelstates = run_once(network, w, b, duration=nsamples*10.*4.)
            update = get_updates(modelstates, n, m)

            for istate, p in targetstates.items():
                state = np.zeros_like(b)
                print(istate)
                state[:n] = -500 + 1000*np.array(istate)
                state[-m:] = 0
                clampedstates = run_once(network, w, b+state, duration=nsamples*10.)
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
    if len(sys.argv) == 6:
        nhidden = int(sys.argv[1])
        scale = float(sys.argv[2])
        learingrate = float(sys.argv[3])
        nsamples = float(sys.argv[4])
        rseed = int(sys.argv[5])
        main(nhidden, scale, learningrate, nsamples, rseed)
    else:
        print("Using default values")
        main()
