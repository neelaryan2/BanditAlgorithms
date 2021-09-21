from bandit import parse_input
import sys
import pandas as pd
import numpy as np
import re

import matplotlib
matplotlib.use('Agg')
from matplotlib import style
style.use('ggplot')
import matplotlib.pyplot as plt


def parse_instance_name(inst):
    regex = r'.*task([0-9]+)/i-([0-9]+).*'
    task, inst_num = re.findall(regex, inst)[0]
    return int(task), int(inst_num)


def threshold_regret(instance, threshold, highs, horizon):
    task, inst_num = parse_instance_name(instance)
    support, p_list = parse_input(instance, task)
    mask = np.array(support) > threshold
    pmax = max([np.sum(np.array(p)[mask]) for p in p_list])
    return pmax * horizon - highs


def get_df(output_file):
    columns = ['instance', 'algorithm', 'randomSeed', 'epsilon', 'scale', 'threshold', 'horizon', 'regret', 'highs']
    types = [str, str, int, float, float, float, int, float, int]

    task_number = int(output_file[4])

    with open(output_file, 'r') as fp:
        lines = [l.strip().split(', ') for l in fp.readlines() if l.strip()]
        lines = [dict(zip(columns, l)) for l in lines]

    df = pd.DataFrame.from_records(lines).astype(dict(zip(columns, types)))

    return task_number, df


task_number, df = get_df(sys.argv[1])

# TASK 2
if task_number == 2:
    new_df = df[['instance', 'scale', 'regret', 'algorithm']].groupby(['instance', 'scale', 'algorithm'], as_index=False).agg('mean')
    for alg, alg_df in new_df.groupby(['algorithm']):
        fig = plt.figure(dpi=300)
        task = None

        for inst, inst_df in alg_df.groupby(['instance']):
            task, inst_num = parse_instance_name(inst)
            scale = inst_df['scale'].to_numpy()
            regret = inst_df['regret'].to_numpy()
            mn_point = np.argmax(-regret)
            p = plt.plot(scale, regret, marker='.', label=f'Instance{inst_num}')
            plt.plot(scale[mn_point], regret[mn_point], color=p[0].get_color(), marker='*', markersize=12)

        filename = f'Task{task}_{alg}'
        plt.title(filename)
        plt.xlabel('Scale')
        plt.ylabel('Regret')
        plt.legend()
        plt.savefig('plots/' + filename + '.png', bbox_inches='tight')
        plt.close()

# TASK 1 and 3
if task_number in [1, 3]:
    new_df = df[['instance', 'horizon', 'algorithm', 'regret']].groupby(['instance', 'horizon', 'algorithm'], as_index=False).agg('mean')

    for inst, grp_df in new_df.groupby(['instance']):
        task, inst_num = parse_instance_name(inst)
        fig = plt.figure(dpi=300)
        filename = f'Task{task}_Instance{inst_num}'

        for alg, alg_df in grp_df.groupby(['algorithm']):
            horizon = alg_df['horizon'].to_numpy()
            regret = alg_df['regret'].to_numpy()
            plt.plot(horizon, regret, marker='.', label=alg)

        plt.xscale('log')
        plt.title(filename)
        plt.xlabel('Horizon')
        plt.ylabel('Regret')
        plt.legend()
        plt.savefig('plots/' + filename + '.png', bbox_inches='tight')
        plt.close()

# TASK 4

if task_number == 4:
    new_df = df[['instance', 'horizon', 'algorithm', 'threshold', 'highs']].groupby(['instance', 'horizon', 'algorithm', 'threshold'], as_index=False).agg('mean')

    for inst_alg, grp_df in new_df.groupby(['instance', 'algorithm']):
        inst, alg = inst_alg
        task, inst_num = parse_instance_name(inst)
        
        for thresh, thresh_df in grp_df.groupby(['threshold']):
            fig = plt.figure(dpi=300)
            filename = f'Task{task}_Instance{inst_num}_{alg}_threshold{thresh:.1f}'
        
            horizon = thresh_df['horizon'].to_numpy()
            highs = thresh_df['highs'].to_numpy()
            high_regrets = threshold_regret(inst, thresh, highs, horizon)
            plt.plot(horizon, high_regrets, marker='.', label=f'Threshold={thresh}')

            plt.xscale('log')
            plt.title(filename)
            plt.xlabel('Horizon')
            plt.ylabel('HIGH Regret')
            plt.legend()
            plt.savefig('plots/' + filename + '.png', bbox_inches='tight')
            plt.close()

