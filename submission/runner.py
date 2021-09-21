import numpy as np
from bandit import main

def instance(task, i):
    return f"../instances/instances-task{task}/i-{i}.txt"

# default args
def default_kwargs():
    return {'instance': None, 'algorithm': 'epsilon-greedy-t1', 'randomSeed': 42, 'epsilon': 0.02, 'scale': 2.0, 'threshold': 0.0, 'horizon': 10000}

horizon_list = [100, 400, 1600, 6400, 25600, 102400]


# TASK 1
kwargs = default_kwargs()
kwargs.update(horizon=horizon_list)
kwargs.update(epsilon=0.02)
for i in range(1, 2):
    kwargs.update(instance=instance(1, i))
    for alg in ['epsilon-greedy-t1', 'ucb-t1', 'kl-ucb-t1', 'thompson-sampling-t1']:
        kwargs.update(algorithm=alg)
        for seed in range(50):
            kwargs.update(randomSeed=seed)
            main(**kwargs)


# TASK 2
# kwargs = default_kwargs()
# kwargs.update(algorithm='ucb-t2')
# kwargs.update(horizon=10000)
# for i in range(1, 6):
#     kwargs.update(instance=instance(2, i))
#     for c in np.arange(0.02, 0.32, 0.02):
#         kwargs.update(scale=c)
#         for seed in range(50):
#             kwargs.update(randomSeed=seed)
#             main(**kwargs)


# TASK 3
# kwargs = default_kwargs()
# kwargs.update(algorithm='alg-t3')
# kwargs.update(horizon=horizon_list)
# for i in range(1, 3):
#     kwargs.update(instance=instance(3, i))
#     for seed in range(50):
#         kwargs.update(randomSeed=seed)
#         main(**kwargs)


# TASK 4
# kwargs = default_kwargs()
# kwargs.update(algorithm='alg-t4')
# kwargs.update(horizon=horizon_list)
# for i in range(1, 3):
#     kwargs.update(instance=instance(4, i))
#     for threshold in [0.2, 0.6]:
#         kwargs.update(threshold=threshold)
#         for seed in range(50):
#             kwargs.update(randomSeed=seed)
#             main(**kwargs)

