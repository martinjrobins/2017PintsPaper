import pints
import pints.toy
import numpy as np
from timeit import default_timer as timer
import multiprocessing
from itertools import repeat
from GPyOpt.methods import BayesianOptimization
import os
import math


class HyperOptimiser:
    def __init__(self, optimiser, model, noise, times, real_parameters, lower, upper):
        self.optimiser = optimiser
        self.model = model
        self.noise = noise
        self.times = times
        self.real_parameters = real_parameters
        self.lower = lower
        self.upper = upper

    def optimise(self, set_hyper_params, parallel=False):
        the_model = self.model()
        print('model = ', the_model)
        values = the_model.simulate(self.real_parameters, self.times)
        value_range = np.max(values) - np.min(values)
        values += np.random.normal(0, self.noise * value_range, values.shape)
        problem = pints.MultiOutputProblem(the_model, self.times, values)
        score = pints.SumOfSquaresError(problem)
        middle = [0.5 * (u + l) for l, u in zip(self.lower, self.upper)]
        sigma = [(1.0/6.0)*(u - l) for l, u in zip(self.lower, self.upper)]
        print('sigma = ', sigma)
        boundaries = pints.Boundaries(self.lower, self.upper)

        optimisation = pints.Optimisation(
            score,
            middle,
            sigma0=sigma,
            boundaries=boundaries,
            method=self.optimiser
        )
        set_hyper_params(optimisation.optimiser())
        if parallel:
            optimisation.set_parallel(int(os.environ['OMP_NUM_THREADS']))

        start = timer()
        found_parameters, found_value = optimisation.run()
        end = timer()
        N = 10
        start_score = timer()
        for i in range(N):
            minimum_value = score(self.real_parameters)
        end_score = timer()
        score_duration = (end_score - start_score) / N

        # found_values = the_model.simulate(found_parameters, times)

        # plt.figure()
        # plt.xlabel('Time')
        # plt.ylabel('Value')
        # plt.plot(times, values, c='b', label='Noisy data')
        # plt.plot(times, found_values, c='r', label='Fit')
        # plt.title('score = %f' % (found_value))
        # plt.savefig('fit_for_optimiser_%s_and_model_%s_with_noise_%f.pdf' %
        #            (optimiser.__name__, model.__name__, noise))

        return found_parameters,  \
            found_value / minimum_value, \
            (end - start) / score_duration


class HyperSampler:
    def __init__(self, mcmc_method, model, noise, times, real_parameters, lower, upper):
        self.mcmc_method = mcmc_method
        self.model = model
        self.noise = noise
        self.times = times
        self.real_parameters = real_parameters
        self.lower = lower
        self.upper = upper

    def sample(self, set_hyper_params):
        the_model = self.model()
        values = the_model.simulate(self.real_parameters, self.times)
        value_range = np.max(values) - np.min(values)
        values += np.random.normal(0, self.noise * value_range, values.shape)
        problem = pints.MultiOutputProblem(the_model, self.times, values)
        log_likelihood = pints.UnknownNoiseLogLikelihood(problem)
        lower = list(self.lower) + [value_range * self.noise / 10.0]*the_model.n_outputs()
        upper = list(self.upper) + [value_range * self.noise * 10]*the_model.n_outputs()
        middle = [0.5 * (u + l) for l, u in zip(lower, upper)]
        sigma = [u - l for l, u in zip(lower, upper)]
        log_prior = pints.UniformLogPrior(lower, upper)
        log_posterior = pints.LogPosterior(log_likelihood, log_prior)
        n_chains = 3
        xs = [[np.random.uniform() * (u - l) + l for l, u in zip(lower, upper)]
              for c in range(n_chains)]
        mcmc = pints.MCMCSampling(log_posterior, 3, xs, method=self.mcmc_method)
        [set_hyper_params(sampler) for sampler in mcmc.samplers()]
        # mcmc.set_max_iterations(10000)
        # mcmc.set_verbose(False)
        # logger_fn = 'logger_info.out'
        # mcmc.set_log_to_file(logger_fn)

        start = timer()
        chains = mcmc.run()
        end = timer()

        rhat = np.max(pints._diagnostics.rhat_all_params(chains))
        ess = 0
        for chain in chains:
            ess += pints._diagnostics.effective_sample_size(chain)
        ess = np.min(ess)
        print('rhat:', rhat)
        print('ess:', ess)
        print('time:', end - start)
        return rhat, ess, end - start


def optimise(sample_num, hyper, x):
    print('optimise for sample', sample_num)
    return hyper.run(x)


def optimise_sampler(num_samples, max_tuning_runs, hyper):
    # tune hyper
    print("TUNING HYPER-PARAMETERS for hyper=", hyper)
    if (hyper.n_parameters() > 0):
        #myBopt = BayesianOptimization(f=hyper, domain=hyper.bounds(), num_cores=os.environ['OMP_NUM_THREADS'])
        myBopt = BayesianOptimization(f=hyper, domain=hyper.bounds())
        myBopt.run_optimization(max_iter=max_tuning_runs)
        x_opt = myBopt.x_opt
    else:
        x_opt = []

        # take samples
    print("TAKING SAMPLES")
    p = multiprocessing.Pool(int(os.environ['OMP_NUM_THREADS']))
    args = zip(range(num_samples), repeat(hyper), repeat(x_opt))
    results = p.starmap(optimise, args)
    return np.array(results)


def sample(sample_num, hyper, x):
    print('sampling for sample', sample_num)
    return hyper.run(x)


def mcmc_sampler(num_samples, max_tuning_runs, hyper):
    # tune hyper
    print("TUNING HYPER-PARAMETERS for hyper=", hyper)
    if (hyper.n_parameters() > 0):
        #myBopt = BayesianOptimization(f=hyper, domain=hyper.bounds(), num_cores=os.environ['OMP_NUM_THREADS'])
        myBopt = BayesianOptimization(f=hyper, domain=hyper.bounds())
        myBopt.run_optimization(max_iter=max_tuning_runs)
        x_opt = myBopt.x_opt
    else:
        x_opt = []

    print("TAKING SAMPLES")
    p = multiprocessing.Pool(int(os.environ['OMP_NUM_THREADS']))
    args = zip(range(num_samples), repeat(hyper), repeat(x_opt))
    results = p.starmap(sample, args)
    return np.array(results)
