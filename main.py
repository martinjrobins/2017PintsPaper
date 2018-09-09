#!/usr/bin/env python3
import pints
import pints.toy
from tasks import optimise_sampler, mcmc_sampler, run_matrix, plot_matrix
from HyperOptimiser import HyperOptimiser, HyperCMAES, HyperPSO, HyperXNES, HyperSNES
from HyperSampler import HyperSampler, HyperMCMC, HyperAdaptiveMCMC, HyperDifferentialEvolutionMCMC, HyperPopulationMCMC
import argparse
from math import floor
from subprocess import call
import os
import sys
import math

models = [pints.toy.LogisticModel, pints.toy.HodgkinHuxleyIKModel,
          pints.toy.GoodwinOscillatorModel, pints.toy.FitzhughNagumoModel,
          pints.toy.Hes1Model, pints.toy.LotkaVolterraModel, pints.toy.RepressilatorModel,
          pints.toy.SIRModel]

optimisers = [pints.CMAES, pints.PSO, pints.XNES, pints.SNES]

mcmcs = [pints.MetropolisRandomWalkMCMC,
         pints.AdaptiveCovarianceMCMC, pints.DifferentialEvolutionMCMC, pints.PopulationMCMC]

noise_levels = [0.01, 0.1]

num_samples = 20
max_tuning_runs = 20

hyper_optimisers = [HyperCMAES, HyperPSO, HyperXNES, HyperSNES]

hyper_mcmcs = [HyperMCMC, HyperAdaptiveMCMC, HyperDifferentialEvolutionMCMC, HyperPopulationMCMC]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run matrix.')
    parser.add_argument('--run', nargs=3,
                        help='three integers indicating the model, method and noise level')
    parser.add_argument('--list', action="store_true",
                        help='list all possible models, methods and noise levels')
    parser.add_argument('--max', action="store_true",
                        help='returns maximum integer')
    parser.add_argument('--arcus', action="store_true",
                        help='start running the matrix on arcus')
    parser.add_argument('--plot', action="store_true",
                        help='plot matrix results')
    parser.add_argument('--run_all', action="store_true",
                        help='run entire matrix')

    args = parser.parse_args()
    size = len(models) * (len(hyper_optimisers) + len(hyper_mcmcs)) * len(noise_levels)

    if args.list:
        print('Models:')
        for i, model in enumerate(models):
            print('\t', i, ':', model)
        print('\nMethods:')
        for i, hyper_optimiser in enumerate(hyper_optimisers):
            print('\t', i, ':', hyper_optimiser)
        for i, hyper_mcmc in enumerate(hyper_mcmcs):
            print('\t', i+len(hyper_optimisers), ':', hyper_mcmc)
        print('\nNoise Levels:')
        for i, noise in enumerate(noise_levels):
            print('\t', i, ':', noise)
    elif args.max:
        print(size - 1)
    elif args.arcus:
        call('ssh arcus-b < run_arcus_job.sh', shell=True)
    elif args.run is not None:
        ni, no, nm = args.run
        print('running matrix (%d,%d,%d)' % (nm, no, ni))
        if no >= len(hyper_optimisers):
            output = mcmc_sampler(num_samples, max_tuning_runs, hyper_mcmcs[no - len(hyper_optimisers)](models[nm],
                                                                                                        noise_levels[ni], times[nm], parameters[nm], lower[nm], upper[nm]))
        else:
            output = optimise_sampler(num_samples, max_tuning_runs, hyper_optimisers[no](models[nm],
                                                                                         noise_levels[ni], times[nm], parameters[nm], lower[nm], upper[nm]))
        fname = 'output_%d_%d_%d.pickle' % (nm, no, ni)
        print('writing ' + fname)
        pickle.dump(output, open(fname, 'wb'))

    elif args.run_all:
        run_matrix(noise_levels, models, hyper_optimisers, max_tuning_runs, num_samples)

    elif args.plot:
        plot_matrix(noise_levels, models, hyper_optimisers, max_tuning_runs, num_samples)
