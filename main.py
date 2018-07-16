import pints
from tasks import HyperOptimiser, HyperSampler, optimise_sampler, mcmc_sampler
import argparse
from math import floor
from subprocess import call
import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np


models = [pints.toy.LogisticModel, pints.toy.HodgkinHuxleyIKModel,
          pints.toy.GoodwinOscillatorModel]


analytic_models = [pints.toy.RosenbrockLogPDF,
                   pints.toy.TwistedGaussianLogPDF,
                   pints.toy.MultimodalNormalLogPDF,
                   pints.toy.HighDimensionalNormalLogPDF]
parameters = [pints.toy.LogisticModel().suggested_parameters(),
              pints.toy.HodgkinHuxleyIKModel().suggested_parameters(),
              pints.toy.GoodwinOscillatorModel().suggested_parameters()]
lower = [[p / 10.0 for p in parameters[0]],
         [p / 10.0 for p in parameters[1]], [p / 10.0 for p in parameters[2]]]
upper = [[p * 10.0 for p in parameters[0]],
         [p * 10.0 for p in parameters[1]], [p * 10.0 for p in parameters[2]]]
times = [pints.toy.LogisticModel().suggested_times(),
         pints.toy.HodgkinHuxleyIKModel().suggested_times(),
         pints.toy.GoodwinOscillatorModel().suggested_times()]
optimisers = [pints.CMAES, pints.PSO, pints.XNES, pints.SNES]
mcmcs = [pints.MetropolisRandomWalkMCMC,
         pints.AdaptiveCovarianceMCMC, pints.DifferentialEvolutionMCMC, pints.PopulationMCMC]
noise_levels = [0.01, 0.1]
num_samples = 20


class HyperCMAES(HyperOptimiser):

    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperCMAES, self).__init__(pints.CMAES, model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 1

    def bounds(self):
        suggest = self.optimiser().suggested_population_size()
        return [{'name': 'pop_size', 'type': 'discrete', 'domain': range(1, 10*suggest)}]

    def __call__(self, x):
        print('calling CMAES with', x)
        self.optimiser.set_population_size(x)
        return super(HyperCMAES, self)()[1]


class HyperPSO(HyperOptimiser):
    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperPSO, self).__init__(pints.PSO, model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 2

    def bounds(self):
        suggest = self.optimiser().suggested_population_size()
        return [{'name': 'pop_size', 'type': 'discrete', 'domain': range(1, 10*suggest)},
                {'name': 'balance', 'type': 'continuous', 'domain': (0, 1)}]

    def __call__(self, x):
        print('calling PSE with', x)
        self.optimiser.set_population_size(x[0])
        self.optimiser.set_local_global_balance(x[1])
        return super(HyperPSO, self)()[1]


class HyperXNES(HyperOptimiser):
    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperXNES, self).__init__(pints.XNES, model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 1

    def bounds(self):
        suggest = self.optimiser().suggested_population_size()
        return [{'name': 'pop_size', 'type': 'discrete', 'domain': range(1, 10*suggest)}]

    def __call__(self, x):
        print('calling XNES with', x)
        self.optimiser.set_population_size(x)
        return super(HyperXNES, self)()[1]


class HyperSNES(HyperOptimiser):
    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperSNES, self).__init__(pints.SNES, model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 1

    def bounds(self):
        suggest = self.optimiser().suggested_population_size()
        return [{'name': 'pop_size', 'type': 'discrete', 'domain': range(1, 10*suggest)}]

    def __call__(self, x):
        print('calling SNES with', x)
        self.optimiser.set_population_size(x)
        return super(HyperSNES, self)()[1]


hyper_optimisers = [HyperCMAES, HyperPSO, HyperXNES, HyperSNES]


class HyperMCMC(HyperOptimiser):
    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperMCMC, self).__init__(pints.MetropolisRandomWalkMCMC, model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 0

    def bounds(self):
        return None

    def __call__(self, x):
        print('calling MCMC with', x)
        return super(HyperCMAES, self)()[1]


class HyperAdaptiveMCMC(HyperOptimiser):
    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperAdaptiveMCMC, self).__init__(pints.AdaptiveCovarianceMCMC, model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 0

    def bounds(self):
        return None

    def __call__(self, x):
        print('calling Adapt MCMC with', x)
        return super(HyperAdaptiveMCMC, self)()[1]


class HyperDifferentialEvolutionMCMC(HyperOptimiser):
    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperDifferentialEvolutionMCMC, self).__init__(pints.DifferentialEvolutionMCMC, model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 2

    def bounds(self):
        return [{'name': 'gamma', 'type': 'continuous', 'domain': (0, 20.38/np.sqrt(2*self.model.n_parameters()))},
                {'name': 'normal_scale', 'type': 'continuous', 'domain': (0, 1)}]

    def __call__(self, x):
        print('calling diff evolution MCMC with', x)
        return super(HyperDifferentialEvolutionMCMC, self)()[1]


class HyperPopulationMCMC(HyperOptimiser):
    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperPopulationMCMC, self).__init__(pints.PopulationMCMC, model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 1

    def bounds(self):
        return [{'name': 'pop_size', 'type': 'discrete', 'domain': range(1, 20)}]

    def __call__(self, x):
        print('calling pop MCMC with', x)
        return super(HyperPopulationMCMC, self)()[1]


hyper_mcmcs = [HyperMCMC, HyperAdaptiveMCMC, HyperDifferentialEvolutionMCMC, HyperPopulationMCMC]


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
    size = len(models) * (len(hyper_optimisers) + len(hyper_mcmcs)) * len(noise_levels)

    if args.max:
        print(size - 1)
    elif args.execute:
        call('ssh arcus-b < run_arcus_job.sh', shell=True)
    elif args.integer is not None:
        i = args.integer
        ni = i % len(noise_levels)
        i = int(floor(i / len(noise_levels)))
        no = i % (len(hyper_optimisers) + len(hyper_mcmcs))
        i = int(floor(i / (len(hyper_optimisers) + len(hyper_mcmcs))))
        nm = i % len(models)
        print('running matrix (%d,%d,%d)' % (nm, no, ni))
        if no >= len(hyper_optimisers):
            output = mcmc_sampler(num_samples, hyper_mcmcs[no - len(hyper_optimisers)], models[nm],
                                  noise_levels[ni], times[nm], parameters[nm], lower[nm], upper[nm])
        else:
            output = optimise_sampler(num_samples, hyper_optimisers[no](models[nm],
                                                                        noise_levels[ni], times[nm], parameters[nm], lower[nm], upper[nm]))
        fname = 'output_%d_%d_%d.pickle' % (nm, no, ni)
        print('writing ' + fname)
        pickle.dump(output, open(fname, 'wb'))
    elif args.all:
        for ni, noise in enumerate(noise_levels):
            for nm, model in enumerate(models):
                for no, optimiser in enumerate(hyper_optimisers):
                    print('running matrix (%d,%d,%d)' % (nm, no, ni))
                    try:
                        output = optimise_sampler(num_samples, optimiser[no](optimiser, model, noise, times[nm], parameters[nm], lower[nm], upper[nm]))
                        fname = 'output_%d_%d_%d.pickle' % (nm, no, ni)
                        print('writing ' + fname)
                        pickle.dump(output, open(fname, 'wb'))
                    except Exception:
                        pass
                for no, mcmc in enumerate(hyper_mcmcs):
                    print('running matrix (%d,%d,%d)' % (nm, no, ni))
                    try:
                        output = mcmc_sampler(num_samples, optimiser(
                                              model, noise, times[nm],
                                              parameters[nm], lower[nm], upper[nm]))
                        fname = 'output_%d_%d_%d.pickle' % (
                            nm, no + len(hyper_optimisers), ni)
                        print('writing ' + fname)
                        pickle.dump(output, open(fname, 'wb'))
                    except Exception:
                        pass

    elif args.plot:
        f = plt.figure()
        y = range(len(models))
        y_labels = [m.__name__ for m in models]
        x = range(len(hyper_optimisers))
        x_labels = [o.optimiser.__name__ for o in hyper_optimisers]
        x_mcmc = range(len(hyper_mcmcs))
        x_mcmc_labels = [m.__name__ for m in hyper_mcmcs]
        for ni, noise in enumerate(noise_levels):
            score = np.zeros((len(models), len(hyper_optimisers), num_samples))
            time = np.zeros((len(models), len(hyper_optimisers), num_samples))
            rhat = np.zeros((len(models), len(hyper_mcmcs), num_samples))
            ess = np.zeros((len(models), len(hyper_mcmcs), num_samples))
            time_mcmc = np.zeros((len(models), len(hyper_mcmcs), num_samples))
            for nm, model in enumerate(models):
                for no, optimiser in enumerate(hyper_optimisers):
                    fname = 'output_%d_%d_%d.pickle' % (
                        nm, no, ni)
                    print('reading ' + fname)
                    if os.path.exists(fname):
                        output = pickle.load(open(fname, 'rb'))
                        assert(len(output[:, 1]) == num_samples)
                        score[nm, no, :] = output[:, 1]
                        time[nm, no, :] = output[:, 2]
                    else:
                        score[nm, no, :] = float('nan')
                        time[nm, no, :] = float('nan')
                for no, mcmc in enumerate(hyper_mcmcs):
                    fname = 'output_%d_%d_%d.pickle' % (
                        nm, no + len(hyper_optimisers), ni)
                    print('reading ' + fname)
                    if os.path.exists(fname):
                        output = pickle.load(open(fname, 'rb'))
                        assert(len(output[:, 1] == num_samples))
                        rhat[nm, no, :] = output[:, 0]
                        ess[nm, no, :] = output[:, 1]
                        time_mcmc[nm, no, :] = output[:, 2]
                    else:
                        rhat[nm, no, :] = float('nan')
                        ess[nm, no, :] = float('nan')
                        time_mcmc[nm, no, :] = float('nan')

            normalise = False
            if normalise:
                for nm, model in enumerate(models):
                    min_score = np.min(score[nm, :, :], axis=(0, 1))
                    max_score = np.max(score[nm, :, :], axis=(0, 1))
                    score[nm, :, :] = (score[nm, :, :] -
                                       min_score) / (max_score - min_score)
                    min_time = np.min(time[nm, :, :], axis=(0, 1))
                    max_time = np.max(time[nm, :, :], axis=(0, 1))
                    time[nm, :, :] = (time[nm, :, :] -
                                      min_time) / (max_time - min_time)
                    min_rhat = np.min(rhat[nm, :, :], axis=(0, 1))
                    max_rhat = np.max(rhat[nm, :, :], axis=(0, 1))
                    rhat[nm, :, :] = (rhat[nm, :, :] -
                                      min_rhat) / (max_rhat - min_rhat)
                    min_ess = np.min(ess[nm, :, :], axis=(0, 1))
                    max_ess = np.max(ess[nm, :, :], axis=(0, 1))
                    ess[nm, :, :] = (ess[nm, :, :] -
                                     min_ess) / (max_ess - min_ess)
                    min_time_mcmc = np.min(time_mcmc[nm, :, :], axis=(0, 1))
                    max_time_mcmc = np.max(time_mcmc[nm, :, :], axis=(0, 1))
                    time_mcmc[nm, :, :] = (time_mcmc[nm, :, :] -
                                           min_time_mcmc) / (max_time_mcmc - min_time_mcmc)

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
            plt.savefig('time_min_with_noise_%d.pdf' % ni)

            plt.clf()
            imshow = plt.imshow(np.mean(rhat, axis=2), cmap='RdYlBu_r',
                                interpolation='nearest')
            plt.xticks(x_mcmc, x_mcmc_labels, rotation='vertical')
            plt.yticks(y, y_labels)
            plt.colorbar(label='rhat (mean)')
            plt.tight_layout()
            plt.savefig('rhat_mean_with_noise_%d.pdf' % ni)
            plt.clf()
            imshow = plt.imshow(np.min(rhat, axis=2), cmap='RdYlBu_r',
                                interpolation='nearest')
            plt.xticks(x_mcmc, x_mcmc_labels, rotation='vertical')
            plt.yticks(y, y_labels)
            plt.colorbar(label='rhat (min)')
            plt.tight_layout()
            plt.savefig('rhat_min_with_noise_%d.pdf' % ni)
            plt.clf()
            plt.imshow(np.mean(ess, axis=2),
                       cmap='RdYlBu_r', interpolation='nearest')
            plt.xticks(x_mcmc, x_mcmc_labels, rotation='vertical')
            plt.yticks(y, y_labels)
            plt.colorbar(label='ess (mean)')
            plt.tight_layout()
            plt.savefig('ess_mean_with_noise_%d.pdf' % ni)
            plt.clf()
            plt.imshow(np.min(ess, axis=2),
                       cmap='RdYlBu_r', interpolation='nearest')
            plt.xticks(x_mcmc, x_mcmc_labels, rotation='vertical')
            plt.yticks(y, y_labels)
            plt.colorbar(label='ess (min)')
            plt.tight_layout()
            plt.savefig('ess_min_with_noise_%d.pdf' % ni)
            plt.clf()
            plt.imshow(np.mean(time_mcmc, axis=2),
                       cmap='RdYlBu_r', interpolation='nearest')
            plt.xticks(x_mcmc, x_mcmc_labels, rotation='vertical')
            plt.yticks(y, y_labels)
            plt.colorbar(label='time_mcmc (mean)')
            plt.tight_layout()
            plt.savefig('time_mcmc_mean_with_noise_%d.pdf' % ni)
            plt.clf()
            plt.imshow(np.min(time_mcmc, axis=2),
                       cmap='RdYlBu_r', interpolation='nearest')
            plt.xticks(x_mcmc, x_mcmc_labels, rotation='vertical')
            plt.yticks(y, y_labels)
            plt.colorbar(label='time_mcmc (min)')
            plt.tight_layout()
            plt.savefig('time_mcmc_min_with_noise_%d.pdf' % ni)
