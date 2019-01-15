import numpy as np
import multiprocessing
from itertools import repeat
from GPyOpt.methods import BayesianOptimization
import pickle
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os

import pmatrix


def to_filename(noise_level, model, hyper_method):
    noise_str = str(noise_level).replace('.', '_')
    model_str = model.__name__
    hyper_method_str = hyper_method.method_name
    return pmatrix.DIR_RESULT + '/' + 'results-%s-%s-%s.pickle' % (model_str, hyper_method_str, noise_str)


def run_single(noise_level, model, hyper_method, max_tuning_runs, num_samples, only_if_not_exist=False):
    fname = to_filename(noise_level, model, hyper_method)

    # dont run if flag set and file already exists
    if only_if_not_exist and os.path.isfile(fname):
        print('WARNING: skipping since results file',fname,'already exists')
        return

    parameters = model().suggested_parameters()
    lower = np.asarray(parameters) / 10.0
    upper = np.asarray(parameters) * 10.0
    times = model().suggested_times()
    if issubclass(hyper_method, pmatrix.HyperSampler):
        output = mcmc_sampler(num_samples,
                              max_tuning_runs,
                              hyper_method(model,
                                           noise_level,
                                           times,
                                           parameters, lower, upper))
    elif issubclass(hyper_method, pmatrix.HyperOptimiser):
        output = optimise_sampler(num_samples,
                                  max_tuning_runs,
                                  hyper_method(model,
                                               noise_level,
                                               times,
                                               parameters, lower, upper))
    else:
        raise TypeError("hyper_method must be an instance of HyperSampler or HyperOptimiser")

    print('writing ' + fname)
    pickle.dump(output, open(fname, 'wb'))


def plot_matrix(noise_levels, models, hyper_optimisers, hyper_mcmcs, max_tuning_runs, num_samples):
    f = plt.figure()
    y = range(len(models))
    y_labels = [m.__name__ for m in models]
    x = range(len(hyper_optimisers))
    x_labels = [o.method_name for o in hyper_optimisers]
    x_mcmc = range(len(hyper_mcmcs))
    x_mcmc_labels = [m.method_name for m in hyper_mcmcs]
    for ni, noise in enumerate(noise_levels):
        score = np.zeros((len(models), len(hyper_optimisers), num_samples))
        time = np.zeros((len(models), len(hyper_optimisers), num_samples))
        rhat = np.zeros((len(models), len(hyper_mcmcs), num_samples))
        ess = np.zeros((len(models), len(hyper_mcmcs), num_samples))
        time_mcmc = np.zeros((len(models), len(hyper_mcmcs), num_samples))
        for nm, model in enumerate(models):
            for no, optimiser in enumerate(hyper_optimisers):
                fname = to_filename(noise, model, optimiser)
                if os.path.exists(fname):
                    print('reading results for (', model.__name__, ',', optimiser.method_name, ',', noise, ')')
                    output = pickle.load(open(fname, 'rb'))
                    assert(len(output[:, 1]) == num_samples)
                    score[nm, no, :] = output[:, 1]
                    time[nm, no, :] = output[:, 2]
                else:
                    print('WARNING: no results for (', model.__name__, ',', optimiser.method_name, ',', noise, ')')
                    score[nm, no, :] = float('nan')
                    time[nm, no, :] = float('nan')
            for no, mcmc in enumerate(hyper_mcmcs):
                fname = to_filename(noise, model, mcmc)
                if os.path.exists(fname):
                    print('reading ' + fname)
                    output = pickle.load(open(fname, 'rb'))
                    assert(len(output[:, 1] == num_samples))
                    rhat[nm, no, :] = output[:, 0]
                    ess[nm, no, :] = output[:, 1]
                    time_mcmc[nm, no, :] = output[:, 2]
                else:
                    print('WARNING: no results for (', model.__name__, ',', mcmc.method_name, ',', noise, ')')
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
        plt.xticks(x, x_labels, rotation=45)
        plt.yticks(y, y_labels)
        plt.colorbar(label='score (mean)')
        plt.tight_layout()
        plt.savefig(pmatrix.DIR_PLOT+'/'+'score_mean_with_noise_%d.pdf' % ni)
        plt.clf()
        imshow = plt.imshow(np.min(score, axis=2), cmap='RdYlBu_r',
                            interpolation='nearest')
        plt.xticks(x, x_labels, rotation=45)
        plt.yticks(y, y_labels)
        plt.colorbar(label='score (min)')
        plt.tight_layout()
        plt.savefig(pmatrix.DIR_PLOT+'/'+'score_min_with_noise_%d.pdf' % ni)
        plt.clf()
        plt.imshow(np.mean(time, axis=2),
                   cmap='RdYlBu_r', interpolation='nearest')
        plt.xticks(x, x_labels, rotation=45)
        plt.yticks(y, y_labels)
        plt.colorbar(label='time (mean)')
        plt.tight_layout()
        plt.savefig(pmatrix.DIR_PLOT+'/'+'time_mean_with_noise_%d.pdf' % ni)
        plt.clf()
        plt.imshow(np.min(time, axis=2),
                   cmap='RdYlBu_r', interpolation='nearest')
        plt.xticks(x, x_labels, rotation=45)
        plt.yticks(y, y_labels)
        plt.colorbar(label='time (min)')
        plt.tight_layout()
        plt.savefig(pmatrix.DIR_PLOT+'/'+'time_min_with_noise_%d.pdf' % ni)

        plt.clf()
        imshow = plt.imshow(np.mean(rhat, axis=2), cmap='RdYlBu_r',
                            interpolation='nearest')
        plt.xticks(x_mcmc, x_mcmc_labels, rotation=45)
        plt.yticks(y, y_labels)
        plt.colorbar(label='rhat (mean)')
        plt.tight_layout()
        plt.savefig(pmatrix.DIR_PLOT+'/'+'rhat_mean_with_noise_%d.pdf' % ni)
        plt.clf()
        imshow = plt.imshow(np.min(rhat, axis=2), cmap='RdYlBu_r',
                            interpolation='nearest')
        plt.xticks(x_mcmc, x_mcmc_labels, rotation=45)
        plt.yticks(y, y_labels)
        plt.colorbar(label='rhat (min)')
        plt.tight_layout()
        plt.savefig(pmatrix.DIR_PLOT+'/'+'rhat_min_with_noise_%d.pdf' % ni)
        plt.clf()
        plt.imshow(np.mean(ess, axis=2),
                   cmap='RdYlBu_r', interpolation='nearest')
        plt.xticks(x_mcmc, x_mcmc_labels, rotation=45)
        plt.yticks(y, y_labels)
        plt.colorbar(label='ess (mean)')
        plt.tight_layout()
        plt.savefig(pmatrix.DIR_PLOT+'/'+'ess_mean_with_noise_%d.pdf' % ni)
        plt.clf()
        plt.imshow(np.min(ess, axis=2),
                   cmap='RdYlBu_r', interpolation='nearest')
        plt.xticks(x_mcmc, x_mcmc_labels, rotation=45)
        plt.yticks(y, y_labels)
        plt.colorbar(label='ess (min)')
        plt.tight_layout()
        plt.savefig(pmatrix.DIR_PLOT+'/'+'ess_min_with_noise_%d.pdf' % ni)
        plt.clf()
        plt.imshow(np.mean(time_mcmc, axis=2),
                   cmap='RdYlBu_r', interpolation='nearest')
        plt.xticks(x_mcmc, x_mcmc_labels, rotation=45)
        plt.yticks(y, y_labels)
        plt.colorbar(label='time_mcmc (mean)')
        plt.tight_layout()
        plt.savefig(pmatrix.DIR_PLOT+'/'+'time_mcmc_mean_with_noise_%d.pdf' % ni)
        plt.clf()
        plt.imshow(np.min(time_mcmc, axis=2),
                   cmap='RdYlBu_r', interpolation='nearest')
        plt.xticks(x_mcmc, x_mcmc_labels, rotation=45)
        plt.yticks(y, y_labels)
        plt.colorbar(label='time_mcmc (min)')
        plt.tight_layout()
        plt.savefig(pmatrix.DIR_PLOT+'/'+'time_mcmc_min_with_noise_%d.pdf' % ni)


def optimise(sample_num, hyper, x):
    print('optimise for sample', sample_num)
    return hyper.optimise(x)


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
    return hyper.sample(x)


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
