from __future__ import print_function

import sys
import numpy as np
from pprint import pformat as pf
from collections import defaultdict
import json

import sbs
sbs.gather_data.set_subprocess_silent(False)
log = sbs.log

# The backend of choice. Both should work but when using neuron, we need to
# disable saturating synapses for now.
sim_name = "pyNN.nest"
#sim_name = "pyNN.neuron"

# some example neuron parameters
neuron_params = {
        "cm": .2,
        "tau_m": 1.,
        "e_rev_E": 0.,
        "e_rev_I": -100.,
        "v_thresh": -50.,
        "tau_syn_E": 10.,
        "v_rest": -50.,
        "tau_syn_I": 10.,
        "v_reset": -50.001,
        "tau_refrac": 10.,
        "i_offset": 0.,
    }

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

def calibration():
    """
        A sample calibration procedure.
    """
    # Since we only have the neuron parameters for now, lets create those first
    nparams = sbs.db.NeuronParametersConductanceExponential(**neuron_params)

    # Now we create a sampler object. We need to specify what simulator we want
    # along with the neuron model and parameters.
    # The sampler accepts both only the neuron parameters or a full sampler
    # configuration as argument.
    sampler = sbs.samplers.LIFsampler(nparams, sim_name=sim_name)

    # Now onto the actual calibration. For this we only need to specify our
    # source configuration and how long/with how many samples we want to
    # calibrate.

    source_config = sbs.db.PoissonSourceConfiguration(
            rates=np.array([3000.] * 2),
            weights=np.array([-1., 1]) * 0.001,
        )

    # We need to specify the remaining calibration parameters
    calibration = sbs.db.Calibration(
            duration=1e5, num_samples=150, burn_in_time=500., dt=0.01,
            source_config=source_config,
            sim_name=sim_name,
            sim_setup_kwargs={"spike_precision": "on_grid"})
    # Do not forget to specify the source configuration!

    # here we could give further kwargs for the pre-calibration phase when the
    # slope of the sigmoid is searched for
    sampler.calibrate(calibration)

    # Afterwards, we need to save the calibration.
    sampler.write_config("tutorial_calibration")

    # Finally, the calibration function can be plotted using the following
    # command ("calibration.png" in the current folder):
    sampler.plot_calibration(save=True)

def prob(w, b, v):
    p = 0.

    for i in range(8):
        p += b[i] * v[i]
        for j in range(8):
            p += w[i, j] * v[i] * v[j]

    ret = np.exp(p)
    return ret


def train_network(nhidden=3, scale=0.3, learningrate=0.1, rseed=42424242):
    """
        How to setup and evaluate a Boltzmann machine. Please note that in
        order to instantiate BMs all needed neuron parameters need to be in the
        database and calibrated.

        Does the same thing as sbs.tools.sample_network(...).
    """

    targetdist = {tuple(int(i) for i in '{:04b}'.format(int(k))): v for k, v in targetstates.items()}
    duration = 1e5

    sampler_config = sbs.db.SamplerConfiguration.load(
            "tutorial_calibration.json")

    # @Stefanie: num_samplers ist die Gesamtzahl der Neuronen (i.e. hidden + visible)
    nvisible = 4
    nunits = nvisible + nhidden
    bm = sbs.network.ThoroughBM(num_samplers=nunits,
                                sim_name=sim_name,
                                sampler_config=sampler_config)

    weights = np.zeros((nunits, nunits))
    weights[nvisible:,:nvisible] = np.normal(0., scale, (nhidden, nvisible))
    weights += weights.T
    bm.weights_theo = weights

    # Set random biases.
    bm.biases_theo = np.normal(0., scale, nunits)

    bm.saturating_synapses_enabled = True
    bm.use_proper_tso = True

    for iternumber in range(10000):
        # Korrelationen fuer CD-Training
        wcorrelations = np.zeros((nvisible, nhidden))
        bcorrelations = np.zeros(bm.num_samplers)

        for state, p in targetdist.items():
            sampleprobs = get_samples(bm, 1000.*(np.array(state)-.5))
            wcorrelations += p * get_wcorrelations(sampleprobs, bm.num_samplers-4)
            bcorrelations += p * get_bcorrelations(sampleprobs, bm.num_samplers-4)
        sampleprobs = get_samples(bm, np.zeros(4))

        wcorrelations -= get_wcorrelations(sampleprobs, bm.num_samplers-4)
        bcorrelations -= get_bcorrelations(sampleprobs, bm.num_samplers-4)

        visibleprobs = defaultdict(float)
        for state, p in sampleprobs.items():
            visibleprobs[tuple(int(i) for i in state[:4])] += p

        (mag1z, mag2z, mag1x, mag2x, corrzz, corrxx) = evaluate(visibleprobs)
        vp = [visibleprobs[k] for k in targetdist.keys()]
        tp = [targetdist[k] for k in targetdist.keys()]
        print('\n\n\n\n\n\n\n\n\n\n')
        print(vp, tp)
        print(dkl(tp, vp))
        with open('dkls_{}.txt'.format(name), 'a') as f:
            f.write('{}\n'.format(dkl(tp, vp)))
        np.savetxt('weights_{}_{:05d}.txt'.format(name, iternumber),
                   bm.weights_theo)
        np.savetxt('bias_{}_{:05d}.txt'.format(name, iternumber),
                   bm.weights_theo)
        with open('mag1z_{}.txt'.format(name), 'a') as f:
            f.write('{}\n'.format(mag1z))
        with open('mag2z_{}.txt'.format(name), 'a') as f:
            f.write('{}\n'.format(mag2z))
        with open('mag1x_{}.txt'.format(name), 'a') as f:
            f.write('{}\n'.format(mag1x))
        with open('mag2x_{}.txt'.format(name), 'a') as f:
            f.write('{}\n'.format(mag2x))
        with open('corrzz_{}.txt'.format(name), 'a') as f:
            f.write('{}\n'.format(corrzz))
        with open('corrxx_{}.txt'.format(name), 'a') as f:
            f.write('{}\n'.format(corrxx))

        # Lernrate (muss moeglicherweise angepasst werden)
        bm.weights_theo[:4, 4:] += 0.01 * wcorrelations
        bm.weights_theo[4:, :4] += 0.01 * wcorrelations.T
        bm.biases_theo += 0.01 * bcorrelations


def dkl(ps, qs):
    return np.sum(p*np.log(p/q) for p, q in zip(ps, qs) if p != 0.)

def evaluate(v):
    mag1z = 0.
    mag2z = 0.
    mag1x = 0.
    mag2x = 0.
    corrzz = 0.
    corrxx = 0.

    Qz = [3., -1., -1., -1.]
    Qx = [0., 2. * np.sqrt(2.), - np.sqrt(2.), - np.sqrt(2.)]

    for state, p in v.items():
        state1 = 3 - (state[1] + 2 * state[0])
        state2 = 3 - (state[3] + 2 * state[2])

        mag1z += p * Qz[state1]
        mag2z += p * Qz[state2]
        mag1x += p * Qx[state1]
        mag2x += p * Qx[state2]
        corrzz += p * Qz[state1] * Qz[state2]
        corrxx += p * Qx[state1] * Qx[state2]

    return (mag1z, mag2z, mag1x, mag2x, corrzz, corrxx)



def get_bcorrelations(sampleprobs, nhidden):
    correlations = np.zeros(4 + nhidden)
    for state, p in sampleprobs.items():
        for i in range(4+nhidden):
            if state[i]:
                correlations[i] += p
    return correlations


def get_wcorrelations(sampleprobs, nhidden):
    correlations = np.zeros((4, nhidden))
    ps = 0.
    for state, p in sampleprobs.items():
        ps += p
        for i in range(4):
            for j in range(nhidden):
                if state[i] and state[j+4]:
                    correlations[i, j] += p
    return correlations


def get_samples(bm, bias):
    oldbias = bm.biases_theo
    newbias = np.array(oldbias)
    newbias[:4] += bias
    bm.biases_theo = newbias
    bm.gather_spikes(duration=1e5, dt=0.1, burn_in_time=1000.)
    bm.biases_theo = oldbias
    print(bm.spike_data)

    return {k: v for k, v in np.ndenumerate(bm.dist_joint_sim)}


if __name__=="__main__":
    if len(sys.argv) == 2:
        calibration()
    if len(sys.argv) == 4:
        nhidden = int(sys.argv[1])
        scale = float(sys.argv[2])
        learingrate = float(sys.argv[3])
        rseed = int(sys.argv[4])
        train_network(nhidden=nhidden, scale=scale, learningrate=learningrate)
    else:
        print("Using default values")
        train_network()
