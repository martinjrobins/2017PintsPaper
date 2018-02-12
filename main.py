import pints
from tasks import optimise
import argparse
from math import floor
from subprocess import call
import pickle
import matplotlib.pyplot as plt
import numpy as np


models = [pints.toy.LogisticModel, pints.toy.HodgkinHuxleyIKModel]
optimisers = [pints.CMAES, pints.PSO, pints.XNES, pints.SNES]
noise_levels = [10.0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run matrix.')
    parser.add_argument('--integer', type=int,
                        help='an integer to run a matrix entry')
    parser.add_argument('--max', action="store_true",
                        help='returns maximum integer')
    parser.add_argument('--execute', action="store_true",
                        help='start running the matrix on arcus')
    parser.add_argument('--plot', action="store_true",
                        help='do plotting')

    args = parser.parse_args()
    size = len(models) * len(optimisers) * len(noise_levels)

    if args.max:
        print(size - 1)
    elif args.execute:
        call('ssh arcus-b < run_arcus_job.sh', shell=True)
    elif args.integer is not None:
        i = args.integer
        ni = i % len(noise_levels)
        i = floor(i / len(noise_levels))
        no = i % len(optimisers)
        i = floor(i / len(optimisers))
        nm = i % len(models)
        print('running matrix (%d,%d,%d)' % (nm, no, ni))
        output = optimise(models[nm], optimisers[no], noise_levels[ni])
        pickle.dump(output, 'output_%d_%d_%d.pickle' % (nm, no, ni))
    elif args.plot:
        score = np.zeros(len(models), len(optimisers))
        time = np.zeros(len(models), len(optimisers))
        for nm, model in enumerate(models):
            for no, optimiser in enumerate(optimisers):
                for ni, noise in enumerate(noise_levels):
                    output = pickle.load(
                        'output_%d_%d_%d.pickle' % (nm, no, ni))
                    score[nm, no] = output[1]
                    time[nm, no] = output[2]
        f = plt.figure()
        plt.imshow(score, cmap='hot', interpolation='nearest')
        plt.savefig('score.pdf')
        plt.close(f)
        f = plt.figure()
        plt.imshow(time, cmap='hot', interpolation='nearest')
        plt.savefig('time.pdf')
        plt.close(f)
