import argparse
import json
import numpy as np
import os
import random

import plt_utils
import matplotlib.pyplot as plt 
from matplotlib.ticker import FuncFormatter

def get_gt():
    gt_size = 0
    f_file = open('data/stats.txt', 'r')
    f_file.readline().rstrip().split()
    for line in f_file:
        if line.split('/')[0] in metadata["val"]:
            gt_size += 1
    return gt_size


def load_metadata(filename):
    metadata = {}
    metadata['val'] = []
    metadata['train'] = []
    metadata['test'] = []
    with open(filename, 'r') as f:
        metadata_json = json.load(f)
        for scene in metadata_json:
            metadata['train'].append(scene['train'])
            metadata['val'].append(scene['val'])
            for test in scene['test']:
                metadata['test'].append(test)
    return metadata

metadata = load_metadata('data/metadata.json')
len_gt = get_gt()

def print_table(config_json, methods_folder):
    methods_list = get_methods(methods_folder, config_json['overview']['methods'])
    methods = {}
    for file in methods_list:
        f_file = open(os.path.join(methods_folder, file + '.txt'), 'r')
        errors = []
        for line in f_file:
            values = line.rstrip().split()
            is_test = not values[0].split('/')[0] in metadata['val']
            if is_test:
                continue
            errors.append([float(values[1]), float(values[2]), float(values[3])])
        errors = np.array(errors, dtype=np.float32)
        DCRE_outlier = (errors[:,2] >= 0.5).sum() / len_gt
        DCRE_5 = (errors[:,2] < 0.05).sum() / len_gt
        DCRE_15 = (errors[:,2] < 0.15).sum() / len_gt
        pose_5 = np.logical_and((errors[:,1] < 5), (errors[:,0] < 0.05)).sum() / len_gt
        pose_outlier = np.logical_or((errors[:,1] >= 25), (errors[:,0] >= 0.5)).sum() / len_gt

        print(config_json['methods'][file]['title'] + ' \t & ' +
              '{:.4}'.format(pose_5) + ' & ({:.4}'.format(np.median(errors[:,0])) + ', {:.4}'.format(np.median(errors[:,1])) + ')' +
              ' & {:.3}'.format(DCRE_5) + ' & ' + '{:.3}'.format(DCRE_15) +
              ' & {:.3}'.format(1 - len(errors) / len_gt) + ' & {:.3}'.format(pose_outlier) + ' & {:.3}'.format(DCRE_outlier) + '\\\\')
        f_file.close()


def read_stats(stats_file):
    stats = {}
    f_file = open(stats_file, 'r')
    header = f_file.readline().rstrip().split()
    for line in f_file:
        line = line.rstrip().split()
        if line[0].split('/')[0] in metadata["test"]:
            continue
        stats[line[0]] = line
    return stats, header


def correlation_data(errors, methods, stats, stats_index, from_to_step, ax_limit):
    curr_from = from_to_step[0]
    output = {} 
    for method in methods:
        output[method] = [[],[],[],{}]
    while (curr_from < from_to_step[1]):
        count = 0
        count_correct = [0] * len(methods)
        for stat in stats:
            stat_value = float(stats[stat][stats_index])
            if (stat_value < (curr_from + from_to_step[2])) and stat_value > curr_from:
                count += 1
                for m in range(len(methods)):
                    if stat in errors[methods[m]] and float(errors[methods[m]][stat]) < 0.15:
                        count_correct[m] += 1

        for m in range(len(methods)):
            acc = float(count_correct[m] / max(count,1))
            if (round(curr_from, 3) in ax_limit):
                output[methods[m]][3][curr_from] = acc
                output[methods[m]][2].append(30)
            else:
                output[methods[m]][2].append(1)
            output[methods[m]][0].append(curr_from)
            output[methods[m]][1].append(acc)
        curr_from += from_to_step[2]
    return output 


def change_correlation(config_json, prediction_path):
    plt_utils.data_size = len_gt
    plot_config = config_json['change_corr']
    stats, header = read_stats('data/stats.txt')
    errors = {}
    methods = get_methods(prediction_path, config_json['change_corr']['methods'])
    for method in methods:
        f_file = open(os.path.join(prediction_path, method + '.txt'), 'r')
        errors[method] = {}
        for line in f_file:
            values = line.rstrip().split()
            is_val = values[0].split('/')[0] in metadata['val']
            is_test = not is_val
            if is_test:
                continue
            errors[method][values[0]] = values[3]
        f_file.close()
    fig = plt.figure(figsize=(13, 2.5))
    axis = plt_utils.add_plots(fig, 4)
    axis[0].set_ylabel(plot_config['axis_ylabel'])
    for ax in range(len(axis)):
        data = correlation_data(errors, methods, stats, header.index(plot_config['stats_keys'][ax]), plot_config['step'][ax], plot_config['limits'][ax])
        axis[ax].set_ylim(plot_config['axis_ylimit'])
        axis[ax].set_xlabel(plot_config['axis_xlabel'][ax])
        axis[ax].set_xlim(plot_config['axis_xlimit'][ax])
        for m in methods:
            axis[ax].scatter(data[m][0], data[m][1], marker='.', color=config_json['methods'][m]['color'], zorder=3, s=data[m][2])
            axis[ax].plot(data[m][0], plt_utils.movingaverage(data[m][1], 10),'r--',linewidth=1.0, color=config_json['methods'][m]['color'], label=config_json['methods'][m]['title'])
        plt_utils.add_limit_2(axis[ax], plot_config['limits'][0], [data[m][3] for m in methods])
    axis[3].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    if plot_config["filename"] != '':
        plt.savefig(plot_config["filename"], dpi=200, bbox_inches='tight')
    plt.show()


def overview(config_json, histogram_folder):
    methods_list = get_methods(histogram_folder, config_json['overview']['methods'])
    plt_utils.data_size = len_gt
    kwargs = dict(histtype='step', alpha=0.9)
    bins = config_json['overview']['bins']
    fig = plt.figure(figsize=(13, 3))
    ax2, ax3, ax4 = plt_utils.add_plots(fig, 3)
    axis = [ax4, ax3, ax2]
    ax2.set_ylabel('Fraction of Frames')
    data = {}
    for method in methods_list:
        data[method] = [[],[],[]] # translation, rotation and DCRE
        f_file = open(os.path.join(histogram_folder, method + '.txt'), 'r')
        for line in f_file:
            values = line.rstrip().split()
            is_val = values[0].split('/')[0] in metadata['val']
            is_test = not is_val
            if is_test:
                continue
            trans_error = float(values[1])
            if (trans_error != -1):
                data[method][0].append(trans_error)
            rot_error = float(values[2])
            if (rot_error != -1):
                data[method][1].append(rot_error)
            flow_error = float(values[3])
            if (flow_error != -1):
                data[method][2].append(flow_error)
        f_file.close()

        color = config_json['methods'][method]['color']
        for ax in range(len(axis)):
            axis[ax].hist(data[method][ax], cumulative=True, range=config_json['overview']['x_range'][ax], color=color, bins = bins, **kwargs, label=config_json['methods'][method]['title'], linewidth=0.75)
            for limit in config_json['overview']['x_limit'][ax]:
                al1 = len(([1 for i in data[method][ax] if i < limit]))
                if (al1/len_gt > config_json['overview']['y_limit_min']):
                    axis[ax].scatter(limit, al1, marker='.', s = 30, color=color, zorder=3)

    for ax in range(len(axis)):
        formatter = FuncFormatter(plt_utils.div_10)
        for limit in config_json['overview']['x_limit'][ax]:
            plt_utils.add_limit(axis[ax], len_gt, limit, data, ax, config_json['overview']['y_limit_min'], config_json['overview']['y_max'])
        axis[ax].set_xlabel(config_json['overview']['x_label'][ax])
        axis[ax].set_ylim([0, len_gt*config_json['overview']['y_max']])
        plt_utils.fix_hist_step_vertical_line_at_end(axis[ax])
        axis[ax].yaxis.set_major_formatter(formatter)

    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    if config_json["overview"]["filename"] != '':
        plt.savefig(config_json["overview"]["filename"], dpi=200, bbox_inches='tight')
    plt.show()


def load_config(filename):
    with open(filename, 'r') as f:
        config_json = json.load(f)
    return config_json


def get_methods(prediction_path, methods_list):
    methods = methods_list
    if (len(methods) == 0):
        methods = []
        for file in os.listdir(prediction_path):
            method_name = os.path.basename(file).split('.')[0]
            methods.append(method_name)
    return methods


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config file', default='./config.json')
    parser.add_argument('--data_path', type=str, help='data path of the depth maps', default='data')
    parser.add_argument('--type', type=int, help='[1 = overview, 2 = latex-table, 3 = change correlation, 4 = all]', default=1)

    args = parser.parse_args()
    config_json = load_config(args.config)
    prediction_path = os.path.join(args.data_path, 'errors')
    if (args.type == 1):
        overview(config_json, prediction_path)
    elif (args.type == 2):
        print_table(config_json, prediction_path)
    elif (args.type == 3):
        change_correlation(config_json, prediction_path)
