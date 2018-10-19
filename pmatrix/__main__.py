#
# Pints performance testing command line utility.
#
# This file is part of Pints Functional Testing.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the Pints
#  functional testing software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import argparse
import pmatrix
import os


def main():
    parser = argparse.ArgumentParser(description='Run matrix.')
    parser.add_argument('--num_threads', nargs='?', default=1, type=int,
                        help='number of cpu threads to use')
    parser.add_argument('--list', action="store_true",
                        help='list all possible models, methods and noise levels')
    parser.add_argument('--max', action="store_true",
                        help='returns maximum integer')
    parser.add_argument('--run', nargs=3, metavar=('Model', 'Method', 'Noise_Level'),
                        help='three integers indicating the model, method and noise level')
    parser.add_argument('--run_all', action="store_true",
                        help='run entire matrix')
    parser.add_argument('--plot', action="store_true",
                        help='plot matrix results')

    args = parser.parse_args()

    os.environ['OMP_NUM_THREADS'] = str(args.num_threads)

    if args.list:
        print('Models:')
        for i, model in enumerate(pmatrix.models):
            print('\t', i, ':', model.__name__)
        print('\nMethods:')
        for i, hyper_optimiser in enumerate(pmatrix.hyper_optimisers):
            print('\t', i, ':', hyper_optimiser.method_name)
        for i, hyper_mcmc in enumerate(pmatrix.hyper_mcmcs):
            print('\t', i+len(pmatrix.hyper_optimisers), ':', hyper_mcmc.method_name)
        print('\nNoise Levels:')
        for i, noise in enumerate(pmatrix.noise_levels):
            print('\t', i, ':', noise)

    if args.max:
        size = len(pmatrix.models)
        size *= (len(pmatrix.hyper_optimisers) + len(pmatrix.hyper_mcmcs))
        size *= len(pmatrix.noise_levels)
        print(size - 1)

    if args.run is not None:
        nm, no, ni = [int(arg) for arg in args.run]
        if no < 0 or no >= len(pmatrix.hyper_optimisers) + len(pmatrix.hyper_mcmcs):
            raise ValueError('method index must be less than ' +
                             str(len(pmatrix.hyper_optimisers) + len(pmatrix.hyper_mcmcs)))
        if no < len(pmatrix.hyper_optimisers):
            print('running matrix (%s,%s,%s)' % (pmatrix.models[nm].__name__,
                                                 pmatrix.hyper_optimisers[no].method_name,
                                                 pmatrix.noise_levels[ni]))
            pmatrix.run_single(pmatrix.noise_levels[ni],
                               pmatrix.models[nm],
                               pmatrix.hyper_optimisers[no],
                               pmatrix.max_tuning_runs, pmatrix.num_samples)
        else:
            no -= len(pmatrix.hyper_optimisers)
            print('running matrix (%s,%s,%s)' % (pmatrix.models[nm].__name__,
                                                 pmatrix.hyper_mcmcs[no].method_name,
                                                 pmatrix.noise_levels[ni]))
            pmatrix.run_single(pmatrix.noise_levels[ni],
                               pmatrix.models[nm],
                               pmatrix.hyper_mcmcs[no],
                               pmatrix.max_tuning_runs, pmatrix.num_samples)

    if args.run_all:
        for noise in pmatrix.noise_levels:
            for model in pmatrix.models:
                for method in pmatrix.hyper_optimisers:
                    pmatrix.run_single(noise, model, method,
                                       pmatrix.max_tuning_runs, pmatrix.num_samples)
                for method in pmatrix.hyper_mcmcs:
                    pmatrix.run_single(noise, model, method,
                                       pmatrix.max_tuning_runs, pmatrix.num_samples)

    if args.plot:
        pmatrix.plot_matrix(pmatrix.noise_levels, pmatrix.models,
                            pmatrix.hyper_optimisers, pmatrix.hyper_mcmcs,
                            pmatrix.max_tuning_runs,
                            pmatrix.num_samples)


if __name__ == '__main__':
    main()
