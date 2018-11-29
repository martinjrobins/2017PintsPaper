#
# Pints performance testing module.
#
# This file is part of Pints Functional Testing.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the Pints
#  functional testing software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import pints
import pints.toy
import os

from ._tasks import (
    optimise_sampler,
    mcmc_sampler,
    plot_matrix,
    run_single,
)
from ._HyperOptimiser import (
    HyperOptimiser,
    HyperCMAES,
    HyperPSO,
    HyperXNES,
    HyperSNES,
)
from ._HyperSampler import (
    HyperSampler,
    HyperMCMC,
    HyperAdaptiveMCMC,
    HyperDifferentialEvolutionMCMC,
    HyperPopulationMCMC,
)


models = [pints.toy.LogisticModel,
          pints.toy.GoodwinOscillatorModel, pints.toy.FitzhughNagumoModel,
          pints.toy.Hes1Model, pints.toy.LotkaVolterraModel, pints.toy.RepressilatorModel,
          pints.toy.SIRModel]

optimisers = [pints.CMAES, pints.PSO, pints.XNES, pints.SNES]

mcmcs = [pints.MetropolisRandomWalkMCMC,
         pints.AdaptiveCovarianceMCMC, pints.DifferentialEvolutionMCMC, pints.PopulationMCMC]

noise_levels = [0.01, 0.1]

num_samples = 20
max_tuning_runs = 20

hyper_optimisers = [HyperCMAES, HyperPSO, HyperXNES, HyperSNES]

hyper_mcmcs = [HyperMCMC, HyperAdaptiveMCMC, HyperDifferentialEvolutionMCMC, HyperPopulationMCMC]

DIR_PMATRIX = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DIR_RESULT = os.path.join(DIR_PMATRIX, 'results')
DIR_PLOT = os.path.join(DIR_PMATRIX, 'plots')

# Ensure output dirs exist
if not os.path.isdir(DIR_RESULT):
    print('Creating results dir: ' + DIR_RESULT)
    os.makedirs(DIR_RESULT)
if not os.path.isdir(DIR_PLOT):
    print('Creating plot dir: ' + DIR_PLOT)
    os.makedirs(DIR_PLOT)
