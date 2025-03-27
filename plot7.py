#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot7.py

This script reads the experiment results data from the LQR with state estimation experiments,
then:
  1. Draws a trajectory plot for each state (4 plots total), showing the evolution over time for
     each filter. For robust filters the optimal robust parameter is used. Additionally, for the
     position states (x and y), the desired trajectory is plotted.
  2. For the three filters with the best LQR cost (using optimal robust parameter), draws a combined
     histogram for LQR cost and a combined histogram for averaged MSE. In each histogram, all three
     filters are plotted in the same figure with different pastel colors (blue, red, and green, with our
     DRKF filter always in green), using alpha transparency. A vertical dotted line in the same color marks
     the average value for each filter, and a grid is added.

Usage:
    python plot7.py --dist normal --noise_dist normal --time 10 --trajectory curvy
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_data(results_path, dist, noise_dist):
    overall_file = os.path.join(results_path, f'overall_results_{dist}_{noise_dist}.pkl')
    optimal_file = os.path.join(results_path, f'optimal_results_{dist}_{noise_dist}.pkl')
    raw_file = os.path.join(results_path, f'raw_experiments_{dist}_{noise_dist}.pkl')
    
    with open(overall_file, 'rb') as f:
        overall_results = pickle.load(f)
    with open(optimal_file, 'rb') as f:
        optimal_results = pickle.load(f)
    with open(raw_file, 'rb') as f:
        raw_experiments = pickle.load(f)
        
    return overall_results, optimal_results, raw_experiments

def generate_desired_trajectory(T_total, trajectory):
    Amp = 5.0      # Amplitude for curvy (sine) trajectory.
    slope = 1.0    # Slope for linear trajectory.
    radius = 5.0   # Radius for circular trajectory.
    omega = 0.5    # Angular frequency.
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

def plot_state_trajectories(overall_results, optimal_results, robust_vals, desired_traj, time):
    # Official filter names.
    filter_names = {
        'finite': "Standard KF (finite)",
        'inf': "Standard KF (infinite)",
        'drkf_inf': "DRKF (ours)",
        'bcot': "BCOT",
        'risk': "Risk-Sensitive"
    }
    # We want our DRKF to be plotted last.
    filters_order = ['finite', 'inf', 'bcot', 'risk', 'drkf_inf']
    # Define a color mapping for state trajectories.
    color_mapping = {
        'finite': 'tab:blue',
        'inf': 'tab:red',
        'bcot': 'tab:orange',
        'risk': 'tab:gray',
        'drkf_inf': 'tab:green'
    }
    
    rep_state_trajs = {}
    for filt in filters_order:
        if filt in ['finite', 'inf']:
            chosen_val = robust_vals[0]
        else:
            chosen_val = optimal_results[filt]['robust_val']
        rep_state_trajs[filt] = overall_results[chosen_val]['state'][filt]
    
    state_names = ['x position', 'x velocity', 'y position', 'y velocity']
    
    for state_idx in range(4):
        plt.figure(figsize=(8,6))
        # For x and y positions, plot the desired trajectory.
        if state_idx == 0:
            plt.plot(time, desired_traj[0, :], 'k--', linewidth=2, label='Desired x')
        elif state_idx == 2:
            plt.plot(time, desired_traj[2, :], 'k--', linewidth=2, label='Desired y')
        for filt in filters_order:
            traj = rep_state_trajs[filt]
            state_values = traj[:, state_idx, 0]
            if filt in ['drkf_inf', 'bcot', 'risk']:
                robust_val = optimal_results[filt]['robust_val']
                label = f"{filter_names[filt]}, $\\theta = {robust_val:.1f}$"
            else:
                label = filter_names[filt]
            plt.plot(time, state_values, marker='o', linestyle='-', color=color_mapping[filt], label=label)
        plt.xlabel('Time (s)', fontsize=16)
        plt.ylabel(state_names[state_idx], fontsize=16)
        plt.title(f'Trajectory for {state_names[state_idx]}', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join("results", "estimator7", f'state_{state_idx}_trajectory.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def plot_histograms(raw_experiments, optimal_results):
    # Official filter names.
    filter_names = {
        'finite': "Standard KF (finite)",
        'inf': "Standard KF (infinite)",
        'drkf_inf': "DRKF (ours)",
        'bcot': "BCOT",
        'risk': "Risk-Sensitive"
    }
    filters = ['finite', 'inf', 'drkf_inf', 'bcot', 'risk']
    sorted_filters = sorted(optimal_results.items(), key=lambda item: item[1]['cost'])
    top3_filters = [item[0] for item in sorted_filters[:3]]
    # Ensure our DRKF (ours) appears last in the legend.
    if 'drkf_inf' in top3_filters:
        top3_filters = [filt for filt in top3_filters if filt != 'drkf_inf'] + ['drkf_inf']
    print("Top three filters based on LQR cost (legend order):", top3_filters)
    
    cost_data = {}
    mse_data = {}
    for filt in top3_filters:
        if filt in ['finite', 'inf']:
            robust_key = 0.2
        else:
            robust_key = optimal_results[filt]['robust_val']
        experiments = raw_experiments[robust_key]
        costs = []
        mses = []
        for exp in experiments:
            for run in exp[filt]:
                costs.append(run[1])
                mses.append(np.mean(run[0]['mse']))
        cost_data[filt] = np.array(costs)
        mse_data[filt] = np.array(mses)
    
    # Use 50 bins.
    all_costs = np.concatenate([cost_data[filt] for filt in top3_filters])
    bins_cost = np.linspace(np.min(all_costs), np.max(all_costs), 51)
    
    # Define colors: DRKF always green; assign blue and red to the others.
    color_map = {}
    for filt in top3_filters:
        if filt == 'drkf_inf':
            color_map[filt] = 'tab:green'
        elif top3_filters.index(filt) == 0:
            color_map[filt] = 'tab:blue'
        else:
            color_map[filt] = 'tab:red'
    
    fig1 = plt.figure(figsize=(8,6))
    ax1 = fig1.gca()
    for filt in top3_filters:
        label = filter_names[filt]
        if filt in ['drkf_inf', 'bcot', 'risk']:
            robust_val = optimal_results[filt]['robust_val']
            label += f", $\\theta = {robust_val:.1f}$"
        ax1.hist(cost_data[filt], bins=bins_cost, alpha=0.5, color=color_map[filt],
                 label=label, linewidth=0)
        avg_cost = np.mean(cost_data[filt])
        ax1.axvline(avg_cost, color=color_map[filt], linestyle='dashed', linewidth=1.5)
    ax1.set_xlabel('LQR Cost', fontsize=16)
    ax1.set_ylabel('Frequency', fontsize=16)
    ax1.set_title('Histogram of LQR Cost for Top 3 Filters', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, fontsize=14)
    ax1.grid(True)
    ax1.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "estimator7", 'combined_histogram_lqr_cost.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    all_mses = np.concatenate([mse_data[filt] for filt in top3_filters])
    bins_mse = np.linspace(np.min(all_mses), np.max(all_mses), 51)
    
    fig2 = plt.figure(figsize=(8,6))
    ax2 = fig2.gca()
    for filt in top3_filters:
        label = filter_names[filt]
        if filt in ['drkf_inf', 'bcot', 'risk']:
            robust_val = optimal_results[filt]['robust_val']
            label += f", $\\theta = {robust_val:.1f}$"
        ax2.hist(mse_data[filt], bins=bins_mse, alpha=0.5, color=color_map[filt],
                 label=label, linewidth=0)
        avg_mse = np.mean(mse_data[filt])
        ax2.axvline(avg_mse, color=color_map[filt], linestyle='dashed', linewidth=1.5)
    ax2.set_xlabel('Averaged MSE', fontsize=16)
    ax2.set_ylabel('Frequency', fontsize=16)
    ax2.set_title('Histogram of Averaged MSE for Top 3 Filters', fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles2, labels2, fontsize=14)
    ax2.grid(True)
    ax2.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "estimator7", 'combined_histogram_mse.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main(args):
    results_path = os.path.join(".", "results", "estimator7")
    overall_results, optimal_results, raw_experiments = load_data(results_path, args.dist, args.noise_dist)
    
    robust_vals = [0.2, 0.3, 0.4, 0.5, 1.0, 2.0]
    desired_traj, time = generate_desired_trajectory(args.time, args.trajectory)
    
    plot_state_trajectories(overall_results, optimal_results, robust_vals, desired_traj, time)
    plot_histograms(raw_experiments, optimal_results)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', default="normal", type=str, help="Disturbance distribution (normal or quadratic)")
    parser.add_argument('--noise_dist', default="normal", type=str, help="Measurement noise distribution (normal or quadratic)")
    parser.add_argument('--time', default=10, type=int, help="Total simulation time")
    parser.add_argument('--trajectory', default="curvy", type=str, help="Desired trajectory type (curvy or circular)")
    args = parser.parse_args()
    main(args)
