import pints
from timeit import default_timer as timer
import numpy as np
import os


class HyperOptimiser:
    """
    A hyper base class that is used to run a single optimisation method

    Args:

    ``optimiser``
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
        The optimiser is given these bounds to search between
     """

    def __init__(self, optimiser, model, noise, times, real_parameters, lower, upper):
        self.method = optimiser
        self.model = model
        self.noise = noise
        self.times = times
        self.real_parameters = real_parameters
        self.lower = lower
        self.upper = upper

    def optimise(self, x, parallel=False):
        """
        Runs the optimisation, this method:
            (1) generates simulated data and adds noise
            (2) sets up the optimiser with the method given, trying to
                optimise the function f(x) = sum of squared error
            (3) runs the optimisation
            (4) returns:
                - the found parameters x,
                - the ratio of f(x) / f(x_0), where x_0 are the real parameters
                - time total time taken divided by the time taken to evaluate a
                  single evaluation of f(x)
        """
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
        boundaries = pints.RectangularBoundaries(self.lower, self.upper)

        optimisation = pints.Optimisation(
            score,
            middle,
            sigma0=sigma,
            boundaries=boundaries,
            method=self.method
        )
        optimisation.optimiser().set_hyper_parameters(x)
        if parallel:
            optimisation.set_parallel(int(os.environ['OMP_NUM_THREADS']))
        else:
            optimisation.set_parallel(False)

        optimisation.set_log_interval(1000)

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
        """
        call method used by the GPyOpt to optimise the hyper-parameters.

        optimises on the ratio of f(x)/f(x_0)
        """
        return self.optimise(x[0], parallel=True)[1]


class HyperCMAES(HyperOptimiser):
    """
    *Extends:* :class:`HyperOptimiser`
    """
    method_name = 'CMAES'

    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperCMAES, self).__init__(pints.CMAES, model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 1

    def bounds(self):
        suggest = self.method(np.zeros(self.model().n_parameters())).suggested_population_size()
        return [{'name': 'pop_size', 'type': 'discrete', 'domain': range(2, 10*suggest)}]


class HyperPSO(HyperOptimiser):
    """
    *Extends:* :class:`HyperOptimiser`
    """
    method_name = 'PSO'

    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperPSO, self).__init__(pints.PSO, model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 2

    def bounds(self):
        suggest = self.method(np.zeros(self.model().n_parameters())).suggested_population_size()
        return [{'name': 'pop_size', 'type': 'discrete', 'domain': range(2, 10*suggest)},
                {'name': 'balance', 'type': 'continuous', 'domain': (0, 1)}]


class HyperXNES(HyperOptimiser):
    """
    *Extends:* :class:`HyperOptimiser`
    """
    method_name = 'XNES'

    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperXNES, self).__init__(pints.XNES, model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 1

    def bounds(self):
        suggest = self.method(np.zeros(self.model().n_parameters())).suggested_population_size()
        return [{'name': 'pop_size', 'type': 'discrete', 'domain': range(2, 10*suggest)}]


class HyperSNES(HyperOptimiser):
    """
    *Extends:* :class:`HyperOptimiser`
    """
    method_name = 'SNES'

    def __init__(self, model, noise, times, real_parameters, lower, upper):
        super(HyperSNES, self).__init__(pints.SNES, model, noise, times, real_parameters, lower, upper)

    def n_parameters(self):
        return 1

    def bounds(self):
        suggest = self.method(np.zeros(self.model().n_parameters())).suggested_population_size()
        return [{'name': 'pop_size', 'type': 'discrete', 'domain': range(2, 10*suggest)}]
