#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot3.py

This script loads the overall results from the LQR with state estimation experiments
(which were run for a range of robust parameters, θ), and then produces two plots:
  (1) Terminal MSE versus θ for each estimator.
  (2) LQR cost versus θ for each estimator.

The robust parameter (θ) is plotted on the x-axis.

Usage:
    python plot3.py --dist normal --noise_dist normal --results_dir ./results/estimator2/
"""

import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', default='normal', type=str,
                        help="Disturbance distribution (normal or quadratic)")
    parser.add_argument('--noise_dist', default='normal', type=str,
                        help="Measurement noise distribution (normal or quadratic)")
    parser.add_argument('--results_dir', default='./results/estimator2/', type=str,
                        help="Directory where result pickle files are stored")
    args = parser.parse_args()
    
    dist = args.dist
    noise_dist = args.noise_dist
    results_dir = args.results_dir

    # Load overall results.
    overall_results_file = os.path.join(results_dir, f'overall_results_{dist}_{noise_dist}.pkl')
    overall_results = load_pickle(overall_results_file)
    
    # Get sorted robust parameter values.
    robust_vals = sorted(overall_results.keys(), key=lambda x: float(x))
    
    # Define estimators and corresponding legend labels.
    estimator_labels = {
        'finite': "Kalman filter (finite)",
        'inf': "Kalman filter (infinite)",
        'drkf_inf': "DRKF (infinite)",
        'bcot': "BCOT",
        'kl': "KL",
        'risk': "Risk-sensitive"
    }
    estimators = list(estimator_labels.keys())
    
    # Prepare arrays for terminal MSE and LQR cost per estimator.
    mse_terminal = {est: [] for est in estimators}
    lqr_cost = {est: [] for est in estimators}
    
    for robust in robust_vals:
        res = overall_results[robust]
        mse_data = res['mse']
        cost_data = res['cost']
        # For each estimator, extract terminal MSE (last value of the mse trajectory)
        for est in estimators:
            mse_terminal[est].append(mse_data[est][-1])
            lqr_cost[est].append(cost_data[est])
    
    robust_array = np.array(robust_vals, dtype=float)
    
    # Plot terminal MSE vs robust parameter.
    plt.figure(figsize=(8,6))
    for est in estimators:
        plt.plot(robust_array, mse_terminal[est], marker='o', label=estimator_labels[est])
    plt.xlabel('Robust Parameter (θ)')
    plt.ylabel('Terminal MSE')
    plt.title(f'Terminal MSE vs Robust Parameter\n(dist: {dist}, noise_dist: {noise_dist})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    mse_plot_file = os.path.join(results_dir, f'mse_vs_theta_{dist}_{noise_dist}.png')
    plt.savefig(mse_plot_file)
    plt.show()
    
    # Plot LQR cost vs robust parameter.
    plt.figure(figsize=(8,6))
    for est in estimators:
        plt.plot(robust_array, lqr_cost[est], marker='s', label=estimator_labels[est])
    plt.xlabel('Robust Parameter (θ)')
    plt.ylabel('LQR Cost')
    plt.title(f'LQR Cost vs Robust Parameter\n(dist: {dist}, noise_dist: {noise_dist})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    cost_plot_file = os.path.join(results_dir, f'cost_vs_theta_{dist}_{noise_dist}.png')
    plt.savefig(cost_plot_file)
    plt.show()

if __name__ == '__main__':
    main()
