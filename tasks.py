import pints
import pints.toy
import numpy as np
from timeit import default_timer as timer
import multiprocessing
from itertools import repeat


def optimise(sample, optimiser, model, noise, times, real_parameters, lower, upper):
    print('Running sample' + str(sample))
    the_model = model()

    values = the_model.simulate(real_parameters, times)
    value_range = np.max(values) - np.min(values)
    values += np.random.normal(0, noise * value_range, values.shape)
    problem = pints.SingleSeriesProblem(the_model, times, values)
    score = pints.SumOfSquaresError(problem)
    middle = [0.5 * (u + l) for l, u in zip(lower, upper)]
    sigma = [u - l for l, u in zip(lower, upper)]
    boundaries = pints.Boundaries(lower, upper)

    start = timer()
    found_parameters, found_value = pints.optimise(
        score,
        middle,
        sigma0=sigma,
        boundaries=boundaries,
        method=optimiser,
    )
    end = timer()
    N = 10
    start_score = timer()
    for i in range(N):
        minimum_value = score(real_parameters)
    end_score = timer()
    score_duration = (end_score - start_score) / N

    #found_values = the_model.simulate(found_parameters, times)

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


def mcmc(sample, mcmc_method, model, noise, times, real_parameters, lower, upper):
    print('Running sample' + str(sample))
    the_model = model()
    values = the_model.simulate(real_parameters, times)
    value_range = np.max(values) - np.min(values)
    values += np.random.normal(0, noise * value_range, values.shape)
    problem = pints.SingleSeriesProblem(the_model, times, values)
    log_likelihood = pints.UnknownNoiseLogLikelihood(problem)
    lower = list(lower) + [value_range * noise / 10.0]
    upper = list(upper) + [value_range * noise * 10]
    middle = [0.5 * (u + l) for l, u in zip(lower, upper)]
    sigma = [u - l for l, u in zip(lower, upper)]
    log_prior = pints.UniformLogPrior(lower, upper)
    log_posterior = pints.LogPosterior(log_likelihood, log_prior)
    n_chains = 3
    xs = [[np.random.uniform() * (u - l) + l for l, u in zip(lower, upper)]
          for c in range(n_chains)]
    mcmc = pints.MCMCSampling(log_posterior, 3, xs, method=mcmc_method)
    # mcmc.set_max_iterations(10000)
    # mcmc.set_verbose(False)
    #logger_fn = 'logger_info.out'
    # mcmc.set_log_to_file(logger_fn)

    start = timer()
    chains = mcmc.run()
    end = timer()

    rhat = np.max(pints._diagnostics.rhat_all_params(chains))
    ess = 0
    for chain in chains:
        ess += pints._diagnostics.effective_sample_size(chain)
    ess = np.min(ess)
    print(rhat)
    print(ess)
    print(end - start)
    return rhat, ess, end - start


def optimise_sampler(num_samples, optimiser, model, noise, times, real_parameters, lower, upper):
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    args = zip(range(num_samples), repeat(optimiser), repeat(
        model), repeat(noise), repeat(times), repeat(real_parameters), repeat(lower), repeat(upper))
    results = p.starmap(optimise, args)
    return np.array(results)


def mcmc_sampler(num_samples, mcmc_method, model, noise, times, real_parameters, lower, upper):
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    args = zip(range(num_samples), repeat(mcmc_method), repeat(
        model), repeat(noise), repeat(times), repeat(real_parameters), repeat(lower), repeat(upper))
    results = p.starmap(mcmc, args)
    return np.array(results)
