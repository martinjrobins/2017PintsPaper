import pints
from timeit import default_timer as timer


class HyperSampler:
    def __init__(self, mcmc_method, model, noise, times, real_parameters, lower, upper):
        self.mcmc_method = mcmc_method
        self.model = model
        self.noise = noise
        self.times = times
        self.real_parameters = real_parameters
        self.lower = lower
        self.upper = upper

    def sample(self, x, parallel=False):
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
        [sampler.set_hyper_params(x) for sampler in mcmc.samplers()]
        if parallel:
            mcmc.set_parallel(int(os.environ['OMP_NUM_THREADS']))

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

    def __call__(self, x):
        ess = self.sample(x, parallel=True)[1]
        if math.isnan(ess):
            return math.inf
        return 1.0/ess


class HyperMCMC(HyperSampler):
    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperMCMC, self).__init__(pints.MetropolisRandomWalkMCMC, model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 0

    def bounds(self):
        return None


class HyperAdaptiveMCMC(HyperSampler):
    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperAdaptiveMCMC, self).__init__(pints.AdaptiveCovarianceMCMC, model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 0

    def bounds(self):
        return None


class HyperDifferentialEvolutionMCMC(HyperSampler):
    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperDifferentialEvolutionMCMC, self).__init__(pints.DifferentialEvolutionMCMC, model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 2

    def bounds(self):
        return [{'name': 'gamma', 'type': 'continuous', 'domain': (0, 20.38/np.sqrt(2*self.model().n_parameters()))},
                {'name': 'normal_scale', 'type': 'continuous', 'domain': (0, 1)}]


class HyperPopulationMCMC(HyperSampler):
    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperPopulationMCMC, self).__init__(pints.PopulationMCMC, model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 1

    def bounds(self):
        return [{'name': 'pop_size', 'type': 'discrete', 'domain': range(1, 20)}]
