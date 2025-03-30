#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised plot7.py

This script reads phase 2 experiment results data produced by main7.py.
It produces two types of plots for each filter:
  1. State trajectories for x position (state 0) and y position (state 2).
     For each, the mean over independent experiments is plotted with a shaded area representing
     the standard deviation at each time step, and the desired trajectory is overlaid.
  2. Control input trajectories.
     Using the LQR gain, the input at each time step is computed as
         u[t] = -K_lqr @ (est_state_traj[t] - desired_traj[:, t])
     for each experiment. Then, the mean and standard deviation (per time step) are computed
     and plotted for each input dimension.

Usage:
    python plot7.py --dist normal --noise_dist normal --time 10 --trajectory curvy
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

# --- LQR Controller Computation ---
def compute_lqr_gain(A, B, Q_lqr, R_lqr):
    P = solve_discrete_are(A, B, Q_lqr, R_lqr)
    K = np.linalg.inv(B.T @ P @ B + R_lqr) @ (B.T @ P @ A)
    return K

def load_data(results_path, dist, noise_dist):
    data_file = os.path.join(results_path, f'phase2_data_{dist}_{noise_dist}.pkl')
    with open(data_file, 'rb') as f:
        phase2_data = pickle.load(f)
    return phase2_data

def generate_desired_trajectory(T_total, trajectory):
    # Same parameters as in main7.py.
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
    # Return a (4 x time_steps) array and the time vector.
    return np.vstack((x_d, vx_d, y_d, vy_d)), time

def debug_print_data(phase2_data, filters_order):
    # For each filter, print the keys (and if the value is a numpy array, its shape)
    for filt in filters_order:
        print(f"\n[DEBUG] Data stored for filter: '{filt}'")
        # Use the first experiment's result to check the keys and shapes.
        first_result = phase2_data[filt]['overall_results'][0]
        for key, value in first_result.items():
            if isinstance(value, np.ndarray):
                print(f"  Key '{key}': shape = {value.shape}")
            else:
                print(f"  Key '{key}': type = {type(value)}")

def plot_filter_trajectories(phase2_data, desired_traj, time, dist_suffix):
    # Mapping filter keys to display names.
    filter_names = {
        'finite': "Time-varying KF",
        'inf': "Time-invariant KF",
        'drkf_inf': "DRKF (ours)",
        'bcot': "BCOT",
        'risk': "Risk-Sensitive"
    }
    filters_order = ['finite', 'inf', 'drkf_inf', 'bcot', 'risk']
    # We only plot state indices 0 (x position) and 2 (y position).
    state_indices = [0, 2]
    state_labels = {0: 'x position', 2: 'y position'}

    for filt in filters_order:
        runs = phase2_data[filt]['overall_results']
        num_experiments = len(runs)
        print(f"[DEBUG] Filter '{filt}' has {num_experiments} experiments (state trajectories).")
        traj_list = []
        for exp_result in runs:
            # Each experiment's state trajectory is stored under f"{filt}_state"
            traj = exp_result[f"{filt}_state"]
            # Expected shape: (time_steps, state_dim, 1); squeeze to (time_steps, state_dim)
            traj = np.squeeze(traj, axis=2)
            traj_list.append(traj)
        traj_stack = np.stack(traj_list, axis=0)  # shape: (num_exp, time_steps, state_dim)
        mean_traj = np.mean(traj_stack, axis=0)     # (time_steps, state_dim)
        std_traj  = np.std(traj_stack, axis=0)       # (time_steps, state_dim)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        for i, state_idx in enumerate(state_indices):
            ax = axs[i]
            if state_idx == 0:
                ax.plot(time, desired_traj[0, :], 'k--', linewidth=2, label='Desired x')
            elif state_idx == 2:
                ax.plot(time, desired_traj[2, :], 'k--', linewidth=2, label='Desired y')
            m = mean_traj[:, state_idx]
            s = std_traj[:, state_idx]
            ax.plot(time, m, marker='o', linestyle='-', color='tab:blue', label='Mean')
            ax.fill_between(time, m - s, m + s, color='tab:blue', alpha=0.3, label='Std dev')
            ax.set_xlabel('Time (s)', fontsize=14)
            ax.set_ylabel(state_labels[state_idx], fontsize=14)
            ax.grid(True)
            ax.legend(fontsize=12)
        if filt in ['drkf_inf', 'bcot', 'risk']:
            optimal_param = phase2_data[filt].get('optimal_param', None)
            if optimal_param is not None:
                title = f"{filter_names[filt]} (θ={optimal_param:.1f})"
            else:
                title = f"{filter_names[filt]} (optimal)"
        else:
            title = filter_names[filt]
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_folder = os.path.join("results", "estimator7")
        os.makedirs(save_folder, exist_ok=True)
        filename = f"state_traj_{filt}{dist_suffix}.png"
        filepath = os.path.join(save_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()

def plot_filter_inputs(phase2_data, desired_traj, time, dist_suffix):
    """
    For each filter, compute the control input at each time step using:
       u[t] = -K_lqr @ (est_state_traj[t] - desired_traj[:, t])
    Then, compute the mean and std (across experiments) for each input dimension and plot.
    """
    # First, define the system matrices and compute K_lqr.
    dt = 0.2
    A = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])
    B = np.array([[0.5*dt**2, 0],
                  [dt, 0],
                  [0, 0.5*dt**2],
                  [0, dt]])
    Q_lqr = np.diag([10, 1, 10, 1])
    R_lqr = 0.1 * np.eye(2)
    K_lqr = compute_lqr_gain(A, B, Q_lqr, R_lqr)

    filters_order = ['finite', 'inf', 'drkf_inf', 'bcot', 'risk']
    # For input, we have 2 dimensions.
    input_labels = {0: 'input 1', 1: 'input 2'}

    for filt in filters_order:
        runs = phase2_data[filt]['overall_results']
        num_experiments = len(runs)
        print(f"[DEBUG] Filter '{filt}' has {num_experiments} experiments (input trajectories).")
        input_list = []
        for exp_result in runs:
            # Each experiment should have an estimated state trajectory stored under 'est_state_traj'
            X_est = exp_result['est_state_traj']
            # Expected shape: (time_steps, state_dim, 1); squeeze to (time_steps, state_dim)
            X_est = np.squeeze(X_est, axis=2)
            # Compute control input at each time step:
            # For each time step t, u[t] = -K_lqr @ (X_est[t] - desired_traj[:, t].reshape(-1,1))
            # Vectorize this computation.
            diff = X_est - desired_traj.T  # (time_steps, state_dim)
            u = - (K_lqr @ diff.T)  # shape: (2, time_steps)
            u = u.T  # shape: (time_steps, 2)
            input_list.append(u)
        input_stack = np.stack(input_list, axis=0)  # shape: (num_exp, time_steps, 2)
        mean_u = np.mean(input_stack, axis=0)         # (time_steps, 2)
        std_u  = np.std(input_stack, axis=0)           # (time_steps, 2)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        for i in range(2):  # two input dimensions
            ax = axs[i]
            m = mean_u[:, i]
            s = std_u[:, i]
            ax.plot(time, m, marker='o', linestyle='-', color='tab:orange', label='Mean')
            ax.fill_between(time, m - s, m + s, color='tab:orange', alpha=0.3, label='Std dev')
            ax.set_xlabel('Time (s)', fontsize=14)
            ax.set_ylabel(input_labels[i], fontsize=14)
            ax.grid(True)
            ax.legend(fontsize=12)
        title = f"Control Input Trajectory for {filt}"
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_folder = os.path.join("results", "estimator7")
        os.makedirs(save_folder, exist_ok=True)
        filename = f"input_traj_{filt}{dist_suffix}.png"
        filepath = os.path.join(save_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()

def main(args):
    results_path = os.path.join(".", "results", "estimator7")
    phase2_data = load_data(results_path, args.dist, args.noise_dist)
    # --- Debug: Print stored data keys and shapes for the first experiment of each filter ---
    filters_order = ['finite', 'inf', 'drkf_inf', 'bcot', 'risk']
    debug_print_data(phase2_data, filters_order)
    dist_suffix = f"_{args.dist}"
    desired_traj, time = generate_desired_trajectory(args.time, args.trajectory)
    plot_filter_trajectories(phase2_data, desired_traj, time, dist_suffix)
    plot_filter_inputs(phase2_data, desired_traj, time, dist_suffix)

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
