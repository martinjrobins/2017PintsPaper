import pints
import pints.toy
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import argparse


def optimise(optimiser, model, noise):
    model = model()
    real_parameters = model.suggested_parameters()
    times = model.suggested_times()
    values = model.simulate(real_parameters, times)
    values += np.random.normal(0, noise, values.shape)
    problem = pints.SingleSeriesProblem(model, times, values)
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

    plt.figure()
    plt.xlabel('Time')
    plt.ylabel('Value')
    for t, v in model.fold(times, values):
        plt.plot(t, v, c='b', label='Noisy data')
    for t, v in model.fold(times, found_value):
        plt.plot(t, v, c='r', label='Fit')
    plt.title('Fit for %s, score = %f' % (optimiser, found_value))
    plt.savefig('fit_for_%s' % (optimiser))

    return found_parameters, found_value, end - start


if __name__ == "__main__":
    optimise(pints.PSO, pints.toy.HodgkinHuxleyIKModel, 10.0)
