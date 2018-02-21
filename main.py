import pints
from tasks import optimise_sampler
import argparse
from math import floor
from subprocess import call
import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np

models = [pints.toy.LogisticModel, pints.toy.HodgkinHuxleyIKModel]
parameters = [[0.015, 500.0],
              pints.toy.HodgkinHuxleyIKModel().suggested_parameters()]
times = [np.linspace(0, 1000, 1000),
         pints.toy.HodgkinHuxleyIKModel().suggested_times()]
optimisers = [pints.CMAES, pints.PSO, pints.XNES, pints.SNES]
noise_levels = [0.01, 0.1]
num_samples = 5

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
    parser.add_argument('--all', action="store_true",
                        help='run entire matrix')

    args = parser.parse_args()
    size = len(models) * len(optimisers) * len(noise_levels)

    if args.max:
        print(size - 1)
    elif args.execute:
        call('ssh arcus-b < run_arcus_job.sh', shell=True)
    elif args.integer is not None:
        i = args.integer
        ni = i % len(noise_levels)
        i = int(floor(i / len(noise_levels)))
        no = i % len(optimisers)
        i = int(floor(i / len(optimisers)))
        nm = i % len(models)
        print('running matrix (%d,%d,%d)' % (nm, no, ni))
        output = optimise_sampler(num_samples, optimisers[no], models[nm],
                                  noise_levels[ni], times[nm], parameters[nm])
        fname = 'output_%d_%d_%d.pickle' % (nm, no, ni)
        print('writing ' + fname)
        pickle.dump(output, open(fname, 'wb'))
    elif args.all:
        for ni, noise in enumerate(noise_levels):
            for nm, model in enumerate(models):
                for no, optimiser in enumerate(optimisers):
                    print('running matrix (%d,%d,%d)' % (nm, no, ni))
                    try:
                        output = optimise_sampler(num_samples, optimiser,
                                                  model, noise, times[nm],
                                                  parameters[nm])
                        fname = 'output_%d_%d_%d.pickle' % (nm, no, ni)
                        print('writing ' + fname)
                        pickle.dump(output, open(fname, 'wb'))
                    except Exception:
                        pass
    elif args.plot:
        f = plt.figure()
        y = range(len(models))
        y_labels = [m.__name__ for m in models]
        x = range(len(optimisers))
        x_labels = [o.__name__ for o in optimisers]
        for ni, noise in enumerate(noise_levels):
            score = np.zeros((len(models), len(optimisers), num_samples))
            time = np.zeros((len(models), len(optimisers), num_samples))
            for nm, model in enumerate(models):
                for no, optimiser in enumerate(optimisers):
                    fname = 'output_%d_%d_%d.pickle' % (
                        nm, no, ni)
                    print('reading ' + fname)
                    if os.path.exists(fname):
                        output = pickle.load(open(fname, 'rb'))
                        assert(len(output[:, 1] == num_samples))
                        score[nm, no, :] = output[:, 1]
                        time[nm, no, :] = output[:, 2]
                    else:
                        score[nm, no, :] = float('nan')
                        time[nm, no, :] = float('nan')

            for nm, model in enumerate(models):
                min_score = np.min(score[nm, :, :], axis=(0, 1))
                max_score = np.max(score[nm, :, :], axis=(0, 1))
                score[nm, :, :] = (score[nm, :, :] -
                                   min_score) / (max_score - min_score)
                min_time = np.min(time[nm, :, :], axis=(0, 1))
                max_time = np.max(time[nm, :, :], axis=(0, 1))
                time[nm, :, :] = (time[nm, :, :] -
                                  min_time) / (max_time - min_time)

            plt.clf()
            imshow = plt.imshow(np.mean(score, axis=2), cmap='RdYlBu_r',
                                interpolation='nearest')
            plt.xticks(x, x_labels, rotation='vertical')
            plt.yticks(y, y_labels)
            plt.colorbar(label='score (mean)')
            plt.tight_layout()
            plt.savefig('score_mean_with_noise_%d.pdf' % ni)
            plt.clf()
            imshow = plt.imshow(np.min(score, axis=2), cmap='RdYlBu_r',
                                interpolation='nearest')
            plt.xticks(x, x_labels, rotation='vertical')
            plt.yticks(y, y_labels)
            plt.colorbar(label='score (min)')
            plt.tight_layout()
            plt.savefig('score_min_with_noise_%d.pdf' % ni)
            plt.clf()
            plt.imshow(np.mean(time, axis=2),
                       cmap='RdYlBu_r', interpolation='nearest')
            plt.xticks(x, x_labels, rotation='vertical')
            plt.yticks(y, y_labels)
            plt.colorbar(label='time (mean)')
            plt.tight_layout()
            plt.savefig('time_mean_with_noise_%d.pdf' % ni)
            plt.clf()
            plt.imshow(np.min(time, axis=2),
                       cmap='RdYlBu_r', interpolation='nearest')
            plt.xticks(x, x_labels, rotation='vertical')
            plt.yticks(y, y_labels)
            plt.colorbar(label='time (min)')
            plt.tight_layout()
            plt.savefig('time_mean_with_noise_%d.pdf' % ni)
