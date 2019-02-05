import pints
from timeit import default_timer as timer
import os
import numpy as np
import math


class HyperSampler:
    """
    A hyper base class that is used to run a single mcmc sampler method

    Args:

    ``mcmc_method``
        The type of the optimiser method to use
    ``model``
        The type of the model to use
    ``noise``
        The level of noise added to the simulated data
    ``times``
        The time points to use for the simulated data
    ``real_parameters``
        The real parameters to use for the simulated data
    ``lower`` ``upper``
        The sampler is given these bounds to search between
     """

    def __init__(self, mcmc_method, model, noise, times, real_parameters, lower, upper):
        self.method = mcmc_method
        self.model = model
        self.noise = noise
        self.times = times
        self.real_parameters = real_parameters
        self.lower = lower
        self.upper = upper

    def sample(self, x, parallel=False):
        """
        Runs the sampler, this method:
            (1) generates simulated data and adds noise
            (2) sets up the sampler with the method given,
                using an KnownNoiseLogLikelihood, and a UniformLogPrior
            (3) runs the sampler
            (4) returns:
                - the calculated rhat value
                - the average of ess across all chains, returning the
                  minimum result across all parameters
                - the total time taken by the sampler
        """

        the_model = self.model()
        values = the_model.simulate(self.real_parameters, self.times)
        value_range = np.max(values) - np.min(values)
        values += np.random.normal(0, self.noise * value_range, values.shape)
        problem = pints.MultiOutputProblem(the_model, self.times, values)
        log_likelihood = pints.KnownNoiseLogLikelihood(problem, value_range*self.noise)
        # lower = list(self.lower) + [value_range *
        #                            self.noise / 10.0]*the_model.n_outputs()
        #upper = list(self.upper) + [value_range * self.noise * 10]*the_model.n_outputs()
        lower = list(self.lower)
        upper = list(self.upper)
        middle = [0.5 * (u + l) for l, u in zip(lower, upper)]
        sigma = [u - l for l, u in zip(lower, upper)]
        log_prior = pints.UniformLogPrior(lower, upper)
        log_posterior = pints.LogPosterior(log_likelihood, log_prior)
        n_chains = int(x[-1])
        xs = [[np.random.uniform() * (u - l) + l for l, u in zip(lower, upper)]
              for c in range(n_chains)]
        mcmc = pints.MCMCSampling(log_posterior, n_chains, xs, method=self.method)
        [sampler.set_hyper_parameters(x[:-1]) for sampler in mcmc.samplers()]
        if parallel:
            mcmc.set_parallel(int(os.environ['OMP_NUM_THREADS']))

        mcmc.set_log_interval(1000)

        start = timer()
        chains = mcmc.run()
        end = timer()

        rhat = np.max(pints._diagnostics.rhat_all_params(chains))
        ess = np.zeros(chains[0].shape[1])
        for chain in chains:
            ess += np.array(pints._diagnostics.effective_sample_size(chain))
        ess /= n_chains
        ess = np.min(ess)
        print('rhat:', rhat)
        print('ess:', ess)
        print('time:', end - start)
        return rhat, ess, end - start

    def __call__(self, x):
        """
        call method used by the GPyOpt to optimise the hyper-parameters

        optimises on 1/ess. i.e. GPyOpt tries to maximise the ess

        """
        ess = self.sample(x[0], parallel=True)[1]
        if math.isnan(ess):
            return math.inf
        return 1.0/ess

    def constraints(self):
        return None

    def uses_gradients(self):
        return False


class HyperMCMC(HyperSampler):
    method_name = 'MetropolisRandomWalkMCMC'

    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperMCMC, self).__init__(pints.MetropolisRandomWalkMCMC,
                                        model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 1

    def bounds(self):
        return [{'name': 'nchains', 'type': 'discrete', 'domain': range(2, 20), 'dimensionality': 1}]


class HyperAdaptiveMCMC(HyperSampler):
    method_name = 'AdaptiveCovarianceMCMC'

    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperAdaptiveMCMC, self).__init__(pints.AdaptiveCovarianceMCMC,
                                                model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 1

    def bounds(self):
        return [{'name': 'nchains', 'type': 'discrete', 'domain': range(2, 20), 'dimensionality': 1}]


class HyperDifferentialEvolutionMCMC(HyperSampler):
    method_name = 'DifferentialEvolutionMCMC'

    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperDifferentialEvolutionMCMC, self).__init__(
            pints.DifferentialEvolutionMCMC, model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 6

    def bounds(self):
        return [{'name': 'gamma', 'type': 'continuous', 'domain': (0, 20.38/np.sqrt(2*self.model().n_parameters()))},
                {'name': 'normal_scale', 'type': 'continuous', 'domain': (0, 1)},
                {'name': 'gamma_switch_rate',
                    'type': 'discrete', 'domain': range(1, 100)},
                {'name': 'normal_error', 'type': 'discrete', 'domain': (False, True)},
                {'name': 'relative_scaling', 'type': 'discrete',
                    'domain': (False, True)},
                {'name': 'nchains', 'type': 'discrete',
                    'domain': range(3, 20), 'dimensionality': 1}
                ]


class HyperPopulationMCMC(HyperSampler):
    method_name = 'PopulationMCMC'

    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperPopulationMCMC, self).__init__(pints.PopulationMCMC,
                                                  model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 2

    def bounds(self):
        return [{'name': 'pop_size', 'type': 'discrete', 'domain': range(1, 20)},
                {'name': 'nchains', 'type': 'discrete',
                    'domain': range(2, 20), 'dimensionality': 1}
                ]


class HyperDreamMCMC(HyperSampler):
    method_name = 'DreamMCMC'

    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperDreamMCMC, self).__init__(pints.DreamMCMC,
                                             model, noise, times,
                                             real_parameters, lower, upper)

    def n_parameters(self):
        return 9

    def bounds(self):
        return [
            {'name': 'b', 'type': 'continuous', 'domain': (0, 5)},
            {'name': 'b_star', 'type': 'continuous', 'domain': (0, 5)},
            {'name': 'p_g', 'type': 'continuous', 'domain': (0, 1)},
            {'name': 'delta_max', 'type': 'discrete', 'domain': range(1, 18)},
            {'name': 'init_phase', 'type': 'discrete', 'domain': (False, True)},
            {'name': 'constant_crossover', 'type': 'discrete', 'domain': (False, True)},
            {'name': 'CR', 'type': 'continuous', 'domain': (0, 1)},
            {'name': 'nCR', 'type': 'discrete', 'domain': range(2, 20)},
            {'name': 'nchains', 'type': 'discrete', 'domain': range(3, 20)},
        ]

    def constraints(self):
        return [
            {'name': 'const_1', 'constraint': 'x[:, 3] - x[:, 8] + 2'},
        ]


class HyperEmceeHammerMCMC(HyperSampler):
    method_name = 'EmceeHammerMCMC'

    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperEmceeHammerMCMC, self).__init__(pints.EmceeHammerMCMC,
                                                   model, noise, times,
                                                   real_parameters, lower, upper)

    def n_parameters(self):
        return 2

    def bounds(self):
        return [
            {'name': 'scale', 'type': 'continuous', 'domain': (0.01, 20)},
            {'name': 'nchains', 'type': 'discrete', 'domain': range(2, 20)},
        ]


class HyperHamiltonianMCMC(HyperSampler):
    method_name = 'HamiltonianMCMC'

    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperHamiltonianMCMC, self).__init__(pints.HamiltonianMCMC,
                                                   model, noise, times,
                                                   real_parameters, lower, upper)

    def n_parameters(self):
        return 3

    def bounds(self):
        return [
            {'name': 'leapfrog_steps', 'type': 'discrete', 'domain': range(20, 21)},
            {'name': 'leapfrog_step_size', 'type': 'continuous', 'domain': (0.1, 0.3)},
            {'name': 'nchains', 'type': 'discrete', 'domain': range(2, 20)},
        ]

    def uses_gradients(self):
        return True
