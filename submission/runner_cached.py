import numpy as np
from bandit import main, pfloat

def instance(task, i):
    return f"../instances/instances-task{task}/i-{i}.txt"

# default args
def default_kwargs():
    return {'instance': None, 'algorithm': 'epsilon-greedy-t1', 'randomSeed': 42, 'epsilon': 0.02, 'scale': 2.0, 'threshold': 0.0, 'horizon': 10000}

horizon_list = [100, 400, 1600, 6400, 25600, 102400]

columns = ['instance', 'algorithm', 'randomSeed', 'epsilon', 'scale', 'threshold', 'horizon', 'regret', 'highs']
types = [str, str, int, float, float, float, int, float, float]

with open('task1.txt', 'r') as fp:
    lines = [l.strip() for l in fp.readlines() if l.strip()]

lines = [[types[i](t) for i, t in enumerate(l.split(', '))] for l in lines]
dicts = [dict(zip(columns, l)) for l in lines]

def check(kwarg):
    ret = []
    horizons = kwarg.get('horizon')
    for d in dicts:
        if all(v == d[k] or k == 'horizon' for k, v in kwarg.items()):
            ret.append(d)
    ret = sorted(ret, key=lambda x: x['horizon'])
    return ret


# TASK 1
kwargs = default_kwargs()
kwargs.update(horizon=horizon_list)
kwargs.update(epsilon=0.02)
for i in range(1, 3):
    kwargs.update(instance=instance(1, i))
    for alg in ['epsilon-greedy-t1', 'ucb-t1', 'kl-ucb-t1', 'thompson-sampling-t1']:
        kwargs.update(algorithm=alg)
        for seed in range(50):
            kwargs.update(randomSeed=seed)
            matches = check(kwargs)
            if not len(matches):
                main(**kwargs)
            else:
                for d in matches:
                    inst = d.get('instance', None)
                    algorithm = d.get('algorithm', None)
                    randomSeed = d.get('randomSeed', None)
                    epsilon = d.get('epsilon', None)
                    scale = d.get('scale', None)
                    threshold = d.get('threshold', None)
                    h = d.get('horizon', None)
                    regret = d.get('regret', None)
                    high = d.get('highs', None)
                    print(inst, algorithm, int(randomSeed), pfloat(epsilon), pfloat(scale), pfloat(threshold), int(h), pfloat(regret), int(high), sep=', ', flush=True)
