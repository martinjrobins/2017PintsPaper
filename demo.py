from tasks import optimise

if __name__ == "__main__":
    optimise('cmaes')
    #optimisers = ['cmaes', 'pso', 'snes', 'xnes']
    #optimise_results = [optimise.delay(o) for o in optimisers]
