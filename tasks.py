from celery import Celery
import pints
import pints.toy
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

app = Celery('tasks', broker='pyamqp://guest@localhost//')


@app.task
def optimise(optimiser):
    model = pints.toy.HodgkinHuxleyIKModel()
    real_parameters = model.suggested_parameters()
    times = model.suggested_times()
    values = model.simulate(real_parameters, times)
    values += np.random.normal(0, 10, values.shape)
    problem = pints.SingleSeriesProblem(model, times, values)
    score = pints.SumOfSquaresError(problem)
    lower = [x / 10.0 for x in real_parameters]
    upper = [x * 10.0 for x in real_parameters]
    middle = [0.5 * (u + l) for l, u in zip(lower, upper)]
    sigma = [u - l for l, u in zip(lower, upper)]
    boundaries = pints.Boundaries(lower, upper)

    start = timer()
    if optimiser == 'cmaes':
        found_parameters, found_value = pints.cmaes(
            score,
            boundaries,
            middle,
            sigma,
        )
    elif optimiser == 'pso':
        found_parameters, found_value = pints.pso(
            score,
            boundaries,
            middle
        )
    elif optimiser == 'snes':
        found_parameters, found_value = pints.snes(
            score,
            boundaries,
            middle,
            sigma,
        )
    elif optimiser == 'xnes':
        found_parameters, found_value = pints.xnes(
            score,
            boundaries,
            middle,
            sigma,
        )
    end = timer()
    found_score = score(found_parameters)
    found_values = problem.evaluate(found_parameters)

    plt.figure()
    plt.xlabel('Time')
    plt.ylabel('Value')
    for t, v in model.fold(times, values):
        plt.plot(t, v, c='b', label='Noisy data')
    for t, v in model.fold(times, found_value):
        plt.plot(t, v, c='r', label='Fit')
    plt.title('Fit for %s, score = %f' % (optimiser, found_score))
    plt.savefig('fit_for_%s' % (optimiser))

    return found_score, end - start


if __name__ == "__main__":
    optimise('pso')
