import pints
from tasks import optimise
import argparse
from math import floor
from subprocess import call

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run matrix.')
    parser.add_argument('--integer', type=int,
                        help='an integer to run a matrix entry')
    parser.add_argument('--size', action="store_true",
                        help='returns size of the matrix')
    parser.add_argument('--execute', action="store_true",
                        help='start running the matrix on arcus')

    args = parser.parse_args()

    models = [pints.toy.LogisticModel, pints.toy.HodgkinHuxleyIKModel]
    optimisers = [pints.CMAES, pints.PSO, pints.XNES, pints.SNES]
    noise_levels = [10.0]

    size = len(models) * len(optimisers) * len(noise_levels)

    if (args.size):
        print(size)
    elif(args.execute):
        machine = 'arcus-b'
        script = 'run_arcus_job.sh'
        call(['ssh', machine, '<', script])
    elif (args.integer is not None):
        i = args.integer
        ni = i % len(noise_levels)
        i = floor(i / len(noise_levels))
        no = i % len(optimisers)
        i = floor(i / len(optimisers))
        nm = i % len(models)
        print('running matrix (%d,%d,%d)' % (nm, no, ni))
        optimise(models[nm], optimisers[no], noise_levels[ni])
