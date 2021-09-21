import os
import argparse
import numpy as np
import random


class Arm:
    def __init__(self, support, probs):
        self.support = np.array(support)
        self.probs = np.array(probs)
        self.k = len(self.support)

    # for [0, 1] support, idx == reward
    def pull(self):
        return np.random.choice(self.k, p=self.probs)

    def mean(self):
        return np.inner(self.support, self.probs)


class Bandit:
    def __init__(self, support, p_list):
        self.support = np.array(support)
        self.arms = [Arm(support, p) for p in p_list]
        self.pull = np.vectorize(lambda x: self.arms[x].pull())
        self.n = len(self.arms)

    def num_arms(self):
        return self.n

    def supports(self):
        return self.support

    def random_pulls(self, t):
        s = np.zeros((self.n, ), dtype=float)
        u = np.zeros((self.n, ), dtype=int)
        indexes = np.random.choice(self.n, size=t, replace=True)
        r = self.pull(indexes)
        np.add.at(s, indexes, r)
        np.add.at(u, indexes, 1)
        return s, u

    def regret(self, r, T):
        pmax = np.max([arm.mean() for arm in self.arms])
        return pmax * T - r

    def threshold_regret(self, h, T, threshold):
        mask = self.support > threshold
        pmax = np.max([np.sum(arm.probs[mask]) for arm in self.arms])
        return pmax * T - h


def parse_input(file, task):
    with open(file, 'r') as fp:
        lines = [l.strip() for l in fp.readlines() if l.strip()]
        p_list = [list(map(float, l.split())) for l in lines]

    if task in [1, 2]:
        supports = [0.0, 1.0]
        p_list = [[1 - l[0], l[0]] for l in p_list]
    elif task in [3, 4]:
        supports = p_list[0]
        p_list = p_list[1:]
    else:
        raise Exception('Task should be in [1, 2, 3, 4]')

    return supports, p_list


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def random_argmax(a):
    assert type(a) is np.ndarray
    return np.random.choice(np.flatnonzero(a == np.max(a)))


def ucb(s, u, c, t):
    if u == 0: return np.inf
    score = s / u + np.sqrt((c * np.log(t + 1)) / u)
    return score


def entropy(x, y):
    if x == 0: return 0
    return x * np.log(x / y)


def kl(x, y):
    return entropy(x, y) + entropy(1 - x, 1 - y)


def kl_ucb(s, u, c, t, max_iter=50, precision=1e-6):
    if u == 0: return np.inf
    lnt = np.log(t + 1)
    pcap = s / u
    target = 0 if t == 0 else (lnt + c * np.log(lnt)) / u
    lo, hi = pcap, 1
    for it in range(max_iter):
        q = (lo + hi) / 2
        current = kl(pcap, q)
        if abs(current - target) < precision:
            break
        if current < target:
            lo = q
        else:
            hi = q
    return q


def dirichlet_sample(alphas):
    r = np.random.standard_gamma(alphas)
    return r / r.sum(axis=-1, keepdims=True)


def epsilon_greedy_t1(bandit, **kwargs):
    eps = kwargs.get('epsilon')
    T = kwargs.get('horizon')
    n = bandit.num_arms()

    p = np.zeros((n, ), dtype=float)
    u = np.zeros((n, ), dtype=int)

    total = 0.0
    rewards = []
    for t in range(T):
        explore = np.random.random() < eps
        arm = np.random.choice(n) if explore else random_argmax(p)
        r = bandit.pull(arm)
        rewards.append(r)
        u[arm] += 1
        p[arm] += (r - p[arm]) / u[arm]
        total += r
    return rewards, [0] * T


def ucb_t1(bandit, **kwargs):
    task = kwargs.get('task')
    T = kwargs.get('horizon')
    n = bandit.num_arms()
    c = 2.0 if task == 1 else kwargs.get('scale')

    ucbs = np.full((n, ), np.inf, dtype=float)
    s = np.zeros((n, ), dtype=float)
    u = np.zeros((n, ), dtype=int)

    rewards = []
    for t in range(T):
        arm = random_argmax(ucbs)
        r = bandit.pull(arm)
        rewards.append(r)
        s[arm] += r
        u[arm] += 1
        ucbs = np.fromiter((ucb(s[i], u[i], c, t) for i in range(n)), dtype=float)
    return rewards, [0] * T


def kl_ucb_t1(bandit, **kwargs):
    c = 3.0
    T = kwargs.get('horizon')
    n = bandit.num_arms()

    kl_ucbs = np.full((n, ), np.inf, dtype=float)
    s = np.zeros((n, ), dtype=float)
    u = np.zeros((n, ), dtype=int)

    rewards = []
    for t in range(T):
        arm = random_argmax(kl_ucbs)
        r = bandit.pull(arm)
        rewards.append(r)
        s[arm] += r
        u[arm] += 1
        kl_ucbs = np.fromiter((kl_ucb(s[i], u[i], c, t) for i in range(n)), dtype=float)
    return rewards, [0] * T


def thompson_sampling_t1(bandit, **kwargs):
    T = kwargs.get('horizon')
    n = bandit.num_arms()

    sf = np.ones((2, n), dtype=int)
    rewards = []
    for t in range(T):
        samples = np.random.beta(a=sf[1], b=sf[0])
        arm = random_argmax(samples)
        r = bandit.pull(arm)
        rewards.append(r)
        sf[r, arm] += 1
    return rewards, [0] * T


def ucb_t2(bandit, **kwargs):
    return ucb_t1(bandit, **kwargs)


def alg_t3(bandit, **kwargs):
    T = kwargs.get('horizon')
    n = bandit.num_arms()
    support = bandit.supports()
    k = len(support)

    alpha = np.ones((n, k), dtype=int)
    rewards = []
    for t in range(T):
        samples = dirichlet_sample(alpha)
        means = np.matmul(samples, support)
        arm = random_argmax(means)
        idx = bandit.pull(arm)
        rewards.append(support[idx])
        alpha[arm, idx] += 1
    return rewards, [0] * T


def alg_t4(bandit, **kwargs):
    T = kwargs.get('horizon')
    threshold = kwargs.get('threshold')
    n = bandit.num_arms()
    support = bandit.supports()

    sf = np.ones((2, n), dtype=int)
    rewards, highs = [], []
    for t in range(T):
        samples = np.random.beta(a=sf[1], b=sf[0])
        arm = random_argmax(samples)
        idx = bandit.pull(arm)
        rewards.append(support[idx])
        idx = int(support[idx] > threshold)
        highs.append(idx)
        sf[idx, arm] += 1
    return rewards, highs


def pfloat(f):
    ret =  "{0:0.2f}".format(f)
    if ret.endswith('.00'):
        ret = ret[:-3]
    return ret


def main(**kwargs):

    instance = kwargs.get('instance', None)
    algorithm = kwargs.get('algorithm', None)
    randomSeed = kwargs.get('randomSeed', None)
    epsilon = kwargs.get('epsilon', None)
    scale = kwargs.get('scale', None)
    threshold = kwargs.get('threshold', None)
    horizon = kwargs.get('horizon', None)

    set_seed(randomSeed)
    task = int(algorithm[-1])
    support, p_list = parse_input(instance, task)
    alg = globals()[algorithm.replace('-', '_')]
    bandit = Bandit(support, p_list)
    
    kwargs.update(task=task)
    if isinstance(horizon, list):
        kwargs.update(horizon=max(horizon))
    else:
        horizon = [horizon]
    
    rewards, highs = alg(bandit, **kwargs)

    for h in horizon:
        reward = np.sum(rewards[:h])
        high = np.sum(highs[:h])
        regret = bandit.regret(reward, h)
        high_regret = bandit.threshold_regret(high, h, threshold)
        print(instance, algorithm, int(randomSeed), pfloat(epsilon), pfloat(scale), pfloat(threshold), int(h), pfloat(regret), int(high), sep=', ', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--instance", metavar='in', required=True, help="Path to the instance file")
    parser.add_argument("--algorithm", metavar='al', default='epsilon-greedy-t1', choices=['epsilon-greedy-t1', 'ucb-t1', 'kl-ucb-t1', 'thompson-sampling-t1', 'ucb-t2', 'alg-t3', 'alg-t4'], help="Algorithm to use")
    parser.add_argument("--randomSeed", metavar='rs', default=42, type=int, help="Number to set as seed for RNG")
    parser.add_argument("--epsilon", metavar='ep', default=0.02, type=float, help="Epsilon for epsilon greedy")
    parser.add_argument("--scale", metavar='c', default=2, type=float, help="Scale factor for exploration bonus of UCB")
    parser.add_argument("--threshold", metavar='th', default=0, type=float, help="Threshold for Task 4")
    parser.add_argument("--horizon", metavar='hz', default=10000, type=int, help="Total number of time steps")
    args = parser.parse_args()

    main(**vars(args))
