#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot7.py

This script reads the phase 2 experiment results data from the LQR with state estimation experiments,
then:
  1. Draws a trajectory plot for each state (4 plots total), showing the evolution over time for
     each filter. For robust filters the optimal robust parameter is used (the legend will show
     “(optimal, $\theta=...$)”). Additionally, for the position states (x and y), the desired
     trajectory is plotted.
  2. For the three filters with the best LQR cost (using the optimal robust parameter), draws a combined
     histogram for LQR cost and a combined histogram for averaged MSE. In each histogram, all three
     filters are plotted in the same figure with different pastel colors (blue, red, green—DRKF always green),
     using alpha transparency. A vertical dashed line in the same color marks the mean value.
  3. Draws a box plot and a violin plot for all five filters—one for LQR cost and one for averaged MSE—
     in which the filters are ordered professionally (with DRKF [ours] shown last).

Titles are removed from the plots, and the distribution name (e.g. `_normal`, `_quadratic`) is appended
to each figure file name.

Usage:
    python plot7.py --dist normal --noise_dist normal --time 10 --trajectory curvy
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_data(results_path, dist, noise_dist):
    phase2_file = os.path.join(results_path, f'phase2_data_{dist}_{noise_dist}.pkl')
    with open(phase2_file, 'rb') as f:
        phase2_data = pickle.load(f)
    return phase2_data

def generate_desired_trajectory(T_total, trajectory):
    Amp = 5.0
    slope = 1.0
    radius = 5.0
    omega = 0.5
    dt = 0.2
    time_steps = int(T_total / dt) + 1
    time = np.linspace(0, T_total, time_steps)
    if trajectory == 'curvy':
        x_d = Amp * np.sin(omega * time)
        vx_d = Amp * omega * np.cos(omega * time)
        y_d = slope * time
        vy_d = slope * np.ones(time_steps)
    elif trajectory == 'circular':
        x_d = radius * np.cos(omega * time)
        vx_d = -radius * omega * np.sin(omega * time)
        y_d = radius * np.sin(omega * time)
        vy_d = radius * omega * np.cos(omega * time)
    else:
        raise ValueError("Unsupported trajectory type.")
    return np.vstack((x_d, vx_d, y_d, vy_d)), time

def plot_state_trajectories(phase2_data, desired_traj, time, dist_suffix):
    filter_names = {
        'finite': "Time-varying KF",
        'inf': "Time-invariant KF",
        'drkf_inf': "DRKF (ours)",
        'bcot': "BCOT",
        'risk': "Risk-Sensitive"
    }
    filters_order = ['finite', 'inf', 'bcot', 'risk', 'drkf_inf']
    color_mapping = {
        'finite': 'tab:blue',
        'inf': 'tab:red',
        'bcot': 'tab:orange',
        'risk': 'tab:gray',
        'drkf_inf': 'tab:green'
    }
    rep_state_trajs = {
        filt: phase2_data[filt]['overall_results'][0][f"{filt}_state"]
        for filt in filters_order
    }
    
    state_names = ['x position', 'x velocity', 'y position', 'y velocity']
    
    for state_idx in range(4):
        plt.figure(figsize=(8,6))
        if state_idx == 0:
            plt.plot(time, desired_traj[0, :], 'k--', linewidth=2, label='Desired x')
        elif state_idx == 2:
            plt.plot(time, desired_traj[2, :], 'k--', linewidth=2, label='Desired y')
        for filt in filters_order:
            traj = rep_state_trajs[filt]
            state_values = traj[:, state_idx, 0]
            if filt in ['drkf_inf', 'bcot', 'risk']:
                optimal_param = phase2_data[filt].get('optimal_param', None)
                if optimal_param is not None:
                    label = rf"{filter_names[filt]} ($\theta={optimal_param:.1f}$)"
                else:
                    label = f"{filter_names[filt]} (optimal)"
            else:
                label = filter_names[filt]
            plt.plot(time, state_values, marker='o', linestyle='-', color=color_mapping[filt], label=label)
        plt.xlabel('Time (s)', fontsize=16)
        plt.ylabel(state_names[state_idx], fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        save_name = f"state_{state_idx}_trajectory{dist_suffix}.png"
        save_path = os.path.join("results", "estimator7", save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def plot_histograms(phase2_data, dist_suffix):
    filter_names = {
        'finite': "Time-varying KF",
        'inf': "Time-invariant KF",
        'drkf_inf': "DRKF (ours)",
        'bcot': "BCOT",
        'risk': "Risk-Sensitive"
    }
    filters = ['finite', 'inf', 'drkf_inf', 'bcot', 'risk']
    
    cost_dict = {}
    for filt in filters:
        exp_costs = [exp['cost'][filt] for exp in phase2_data[filt]['overall_results']]
        cost_dict[filt] = np.mean(exp_costs)
    sorted_filters = sorted(cost_dict.items(), key=lambda item: item[1])
    top3_filters = [item[0] for item in sorted_filters[:3]]
    if 'drkf_inf' in top3_filters:
        top3_filters = [f for f in top3_filters if f != 'drkf_inf'] + ['drkf_inf']
    
    # Print info
    for filt in top3_filters:
        exp_costs = [exp['cost'][filt] for exp in phase2_data[filt]['overall_results']]
        exp_mses = [exp[filt] for exp in phase2_data[filt]['overall_results']]
        mean_cost = np.mean(exp_costs)
        mean_mse = np.mean(exp_mses)
        optimal_param = phase2_data[filt].get('optimal_param', None)
        if optimal_param is not None:
            print(f"{filter_names[filt]}: Optimal parameter = {optimal_param:.1f}, "
                  f"Mean LQR cost = {mean_cost:.2f}, Mean averaged MSE = {mean_mse:.4f}")
        else:
            print(f"{filter_names[filt]}: Mean LQR cost = {mean_cost:.2f}, Mean averaged MSE = {mean_mse:.4f}")
    
    cost_data = {}
    mse_data = {}
    for filt in top3_filters:
        experiments = phase2_data[filt]['raw_data']
        costs = []
        mses = []
        for exp in experiments:
            for run in exp:
                costs.append(run[1])
                mses.append(np.mean(run[0]['mse']))
        cost_data[filt] = np.array(costs)
        mse_data[filt] = np.array(mses)
    
    all_costs = np.concatenate([cost_data[f] for f in top3_filters])
    bins_cost = np.linspace(np.min(all_costs), np.max(all_costs), 81)
    
    color_map = {}
    for i, filt in enumerate(top3_filters):
        if filt == 'drkf_inf':
            color_map[filt] = 'tab:green'
        elif i == 0:
            color_map[filt] = 'tab:blue'
        else:
            color_map[filt] = 'tab:red'
    
    # Hist for LQR cost
    plt.figure(figsize=(8,6))
    for filt in top3_filters:
        if filt in ['drkf_inf', 'bcot', 'risk']:
            optimal_param = phase2_data[filt].get('optimal_param', None)
            if optimal_param is not None:
                label = rf"{filter_names[filt]} ($\theta={optimal_param:.1f}$)"
            else:
                label = f"{filter_names[filt]} (optimal)"
        else:
            label = filter_names[filt]
        plt.hist(cost_data[filt], bins=bins_cost, alpha=0.5,
                 color=color_map[filt], label=label, linewidth=0)
        avg_cost = np.mean(cost_data[filt])
        plt.axvline(avg_cost, color=color_map[filt], linestyle='dashed', linewidth=1.5)
    plt.xlabel('LQR Cost', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    save_name = f"combined_histogram_lqr_cost{dist_suffix}.png"
    plt.savefig(os.path.join("results", "estimator7", save_name), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Hist for MSE
    all_mses = np.concatenate([mse_data[f] for f in top3_filters])
    bins_mse = np.linspace(np.min(all_mses), np.max(all_mses), 81)
    
    plt.figure(figsize=(8,6))
    for filt in top3_filters:
        if filt in ['drkf_inf', 'bcot', 'risk']:
            optimal_param = phase2_data[filt].get('optimal_param', None)
            if optimal_param is not None:
                label = rf"{filter_names[filt]} ($\theta={optimal_param:.1f}$)"
            else:
                label = f"{filter_names[filt]} (optimal)"
        else:
            label = filter_names[filt]
        plt.hist(mse_data[filt], bins=bins_mse, alpha=0.5,
                 color=color_map[filt], label=label, linewidth=0)
        avg_mse = np.mean(mse_data[filt])
        plt.axvline(avg_mse, color=color_map[filt], linestyle='dashed', linewidth=1.5)
    plt.xlabel('Averaged MSE', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    save_name = f"combined_histogram_mse{dist_suffix}.png"
    plt.savefig(os.path.join("results", "estimator7", save_name), dpi=300, bbox_inches='tight')
    plt.show()

def plot_boxplots(phase2_data, dist_suffix):
    filters = ['finite', 'inf', 'bcot', 'risk', 'drkf_inf']
    filter_names = {
        'finite': "Time-varying KF",
        'inf': "Time-invariant KF",
        'bcot': "BCOT",
        'risk': "Risk-Sensitive",
        'drkf_inf': "DRKF (ours)"
    }
    cost_data_all = []
    mse_data_all = []
    for filt in filters:
        experiments = phase2_data[filt]['raw_data']
        costs = []
        mses = []
        for exp in experiments:
            for run in exp:
                costs.append(run[1])
                mses.append(np.mean(run[0]['mse']))
        cost_data_all.append(costs)
        mse_data_all.append(mses)
    
    # Box plot for LQR cost
    plt.figure(figsize=(10,6))
    bp1 = plt.boxplot(cost_data_all, patch_artist=True,
                      medianprops={'linewidth':2}, boxprops={'linewidth':2})
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:gray', 'tab:green']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    plt.xticks(range(1, len(filters)+1), [filter_names[f] for f in filters], fontsize=14)
    plt.xlabel('Filter', fontsize=16)
    plt.ylabel('LQR Cost', fontsize=16)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_name = f"boxplot_lqr_cost{dist_suffix}.png"
    plt.savefig(os.path.join("results", "estimator7", save_name), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Box plot for MSE
    plt.figure(figsize=(10,6))
    bp2 = plt.boxplot(mse_data_all, patch_artist=True,
                      medianprops={'linewidth':2}, boxprops={'linewidth':2})
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    plt.xticks(range(1, len(filters)+1), [filter_names[f] for f in filters], fontsize=14)
    plt.xlabel('Filter', fontsize=16)
    plt.ylabel('Averaged MSE', fontsize=16)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_name = f"boxplot_mse{dist_suffix}.png"
    plt.savefig(os.path.join("results", "estimator7", save_name), dpi=300, bbox_inches='tight')
    plt.show()

def plot_violinplots(phase2_data, dist_suffix):
    filters = ['finite', 'inf', 'bcot', 'risk', 'drkf_inf']
    filter_names = {
        'finite': "Time-varying KF",
        'inf': "Time-invariant KF",
        'bcot': "BCOT",
        'risk': "Risk-Sensitive",
        'drkf_inf': "DRKF (ours)"
    }
    cost_data_all = []
    mse_data_all = []
    for filt in filters:
        experiments = phase2_data[filt]['raw_data']
        costs = []
        mses = []
        for exp in experiments:
            for run in exp:
                costs.append(run[1])
                mses.append(np.mean(run[0]['mse']))
        cost_data_all.append(costs)
        mse_data_all.append(mses)
    
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:gray', 'tab:green']
    
    # Violin plot for LQR cost
    plt.figure(figsize=(10,6))
    vp1 = plt.violinplot(cost_data_all, showmeans=True, showmedians=True, showextrema=True)
    for i, pc in enumerate(vp1['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    plt.xticks(range(1, len(filters)+1), [filter_names[f] for f in filters], fontsize=14)
    plt.xlabel('Filter', fontsize=18)
    plt.ylabel('LQR Cost', fontsize=18)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_name = f"violinplot_lqr_cost{dist_suffix}.png"
    plt.savefig(os.path.join("results", "estimator7", save_name), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Violin plot for MSE
    plt.figure(figsize=(10,6))
    vp2 = plt.violinplot(mse_data_all, showmeans=True, showmedians=True, showextrema=True)
    for i, pc in enumerate(vp2['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    plt.xticks(range(1, len(filters)+1), [filter_names[f] for f in filters], fontsize=14)
    plt.xlabel('Filter', fontsize=18)
    plt.ylabel('Average MSE', fontsize=18)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_name = f"violinplot_mse{dist_suffix}.png"
    plt.savefig(os.path.join("results", "estimator7", save_name), dpi=300, bbox_inches='tight')
    plt.show()

def main(args):
    results_path = os.path.join(".", "results", "estimator7")
    phase2_data = load_data(results_path, args.dist, args.noise_dist)
    # We'll use a suffix like "_normal" or "_quadratic" for file names
    dist_suffix = f"_{args.dist}"
    
    desired_traj, time = generate_desired_trajectory(args.time, args.trajectory)
    
    plot_state_trajectories(phase2_data, desired_traj, time, dist_suffix)
    plot_histograms(phase2_data, dist_suffix)
    plot_boxplots(phase2_data, dist_suffix)
    plot_violinplots(phase2_data, dist_suffix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', default="normal", type=str,
                        help="Disturbance distribution (normal or quadratic)")
    parser.add_argument('--noise_dist', default="normal", type=str,
                        help="Measurement noise distribution (normal or quadratic)")
    parser.add_argument('--time', default=10, type=int,
                        help="Total simulation time")
    parser.add_argument('--trajectory', default="curvy", type=str,
                        help="Desired trajectory type (curvy or circular)")
    args = parser.parse_args()
    main(args)
