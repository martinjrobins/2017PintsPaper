import pints
import pints.toy
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import argparse


def optimise(optimiser, model, noise, times, real_parameters):
    the_model = model()
    values = the_model.simulate(real_parameters, times)
    value_range = np.max(values) - np.min(values)
    values += np.random.normal(0, noise * value_range, values.shape)
    problem = pints.SingleSeriesProblem(the_model, times, values)
    score = pints.SumOfSquaresError(problem)
    lower = [x / 10.0 for x in real_parameters]
    upper = [x * 10.0 for x in real_parameters]
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
    found_values = the_model.simulate(found_parameters, times)

    plt.figure()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.plot(times, values, c='b', label='Noisy data')
    plt.plot(times, found_values, c='r', label='Fit')
    plt.title('score = %f' % (found_value))
    plt.savefig('fit_for_optimiser_%s_and_model_%s_with_noise_%f.pdf' %
                (optimiser.__name__, model.__name__, noise))

    return found_parameters, found_value, end - start


if __name__ == "__main__":
    optimise(pints.PSO, pints.toy.HodgkinHuxleyIKModel, 10.0)
