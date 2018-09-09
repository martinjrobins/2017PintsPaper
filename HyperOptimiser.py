import pints
from timeit import default_timer as timer


class HyperOptimiser:
    def __init__(self, optimiser, model, noise, times, real_parameters, lower, upper):
        self.optimiser = optimiser
        self.model = model
        self.noise = noise
        self.times = times
        self.real_parameters = real_parameters
        self.lower = lower
        self.upper = upper

    def optimise(self, x, parallel=False):
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
        optimisation.optimiser().set_hyper_params(x)
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

        return found_parameters,  \
            found_value / minimum_value, \
            (end - start) / score_duration

    def __call__(self, x):
        return self.optimise(x, parallel=True)[1]


class HyperCMAES(HyperOptimiser):

    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperCMAES, self).__init__(pints.CMAES, model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 1

    def bounds(self):
        suggest = self.optimiser(np.zeros(self.model().n_parameters())).suggested_population_size()
        return [{'name': 'pop_size', 'type': 'discrete', 'domain': range(2, 10*suggest)}]


class HyperPSO(HyperOptimiser):
    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperPSO, self).__init__(pints.PSO, model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 2

    def bounds(self):
        suggest = self.optimiser(np.zeros(self.model().n_parameters())).suggested_population_size()
        return [{'name': 'pop_size', 'type': 'discrete', 'domain': range(2, 10*suggest)},
                {'name': 'balance', 'type': 'continuous', 'domain': (0, 1)}]


class HyperXNES(HyperOptimiser):
    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperXNES, self).__init__(pints.XNES, model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 1

    def bounds(self):
        suggest = self.optimiser(np.zeros(self.model().n_parameters())).suggested_population_size()
        return [{'name': 'pop_size', 'type': 'discrete', 'domain': range(2, 10*suggest)}]


class HyperSNES(HyperOptimiser):
    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperSNES, self).__init__(pints.SNES, model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 1

    def bounds(self):
        suggest = self.optimiser(np.zeros(self.model().n_parameters())).suggested_population_size()
        return [{'name': 'pop_size', 'type': 'discrete', 'domain': range(2, 10*suggest)}]
