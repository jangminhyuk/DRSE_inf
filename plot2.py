#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot.py

This script loads the saved MSE mean and standard deviation data (from the
finite–horizon KF, steady–state KF, steady–state DRKF, steady–state BCOT, 
steady–state KL robust filter, and risk–sensitive filter experiments) and plots
the evolution of the MSE over time. Each estimator’s MSE mean is plotted with a 
shaded region indicating ±0.2× the standard deviation.

Usage:
    python plot.py --dist normal --noise_dist normal --results_dir ./results/estimator2/
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

    # Construct file names for each estimator:
    file_kf_mean       = os.path.join(results_dir, f'kf_mse_mean_{dist}_{noise_dist}.pkl')
    file_kf_std        = os.path.join(results_dir, f'kf_mse_std_{dist}_{noise_dist}.pkl')
    file_kf_inf_mean   = os.path.join(results_dir, f'kf_inf_mse_mean_{dist}_{noise_dist}.pkl')
    file_kf_inf_std    = os.path.join(results_dir, f'kf_inf_mse_std_{dist}_{noise_dist}.pkl')
    file_drkf_inf_mean = os.path.join(results_dir, f'kf_drkf_inf_mse_mean_{dist}_{noise_dist}.pkl')
    file_drkf_inf_std  = os.path.join(results_dir, f'kf_drkf_inf_mse_std_{dist}_{noise_dist}.pkl')
    file_bcot_mean     = os.path.join(results_dir, f'bcot_mse_mean_{dist}_{noise_dist}.pkl')
    file_bcot_std      = os.path.join(results_dir, f'bcot_mse_std_{dist}_{noise_dist}.pkl')
    file_kl_mean       = os.path.join(results_dir, f'kl_mse_mean_{dist}_{noise_dist}.pkl')
    file_kl_std        = os.path.join(results_dir, f'kl_mse_std_{dist}_{noise_dist}.pkl')
    file_risk_mean     = os.path.join(results_dir, f'risk_sensitive_mse_mean_{dist}_{noise_dist}.pkl')
    file_risk_std      = os.path.join(results_dir, f'risk_sensitive_mse_std_{dist}_{noise_dist}.pkl')
    
    # Load data.
    mse_kf_mean       = load_pickle(file_kf_mean)
    mse_kf_std        = load_pickle(file_kf_std)
    mse_kf_inf_mean   = load_pickle(file_kf_inf_mean)
    mse_kf_inf_std    = load_pickle(file_kf_inf_std)
    mse_drkf_inf_mean = load_pickle(file_drkf_inf_mean)
    mse_drkf_inf_std  = load_pickle(file_drkf_inf_std)
    mse_bcot_mean     = load_pickle(file_bcot_mean)
    mse_bcot_std      = load_pickle(file_bcot_std)
    mse_kl_mean       = load_pickle(file_kl_mean)
    mse_kl_std        = load_pickle(file_kl_std)
    mse_risk_mean     = load_pickle(file_risk_mean)
    mse_risk_std      = load_pickle(file_risk_std)
    
    # Time axis (assumes that the saved MSE arrays have T+1 entries).
    T = len(mse_kf_mean) - 1
    time_steps = np.arange(T + 1)
    
    plt.figure(figsize=(10, 6))
    
    # Finite-horizon KF.
    plt.plot(time_steps, mse_kf_mean, 'b-', label='Finite-horizon KF')
    plt.fill_between(time_steps,
                     mse_kf_mean - 0.2 * mse_kf_std,
                     mse_kf_mean + 0.2 * mse_kf_std,
                     color='b', alpha=0.2)
    
    # Steady-state KF.
    plt.plot(time_steps, mse_kf_inf_mean, 'r-', label='Steady-state KF')
    plt.fill_between(time_steps,
                     mse_kf_inf_mean - 0.2 * mse_kf_inf_std,
                     mse_kf_inf_mean + 0.2 * mse_kf_inf_std,
                     color='r', alpha=0.2)
    
    # Steady-state DRKF.
    plt.plot(time_steps, mse_drkf_inf_mean, 'g-', label='Steady-state DRKF')
    plt.fill_between(time_steps,
                     mse_drkf_inf_mean - 0.2 * mse_drkf_inf_std,
                     mse_drkf_inf_mean + 0.2 * mse_drkf_inf_std,
                     color='g', alpha=0.2)
    
    # Steady-state BCOT.
    plt.plot(time_steps, mse_bcot_mean, 'm-', label='BCOT')
    plt.fill_between(time_steps,
                     mse_bcot_mean - 0.2 * mse_bcot_std,
                     mse_bcot_mean + 0.2 * mse_bcot_std,
                     color='m', alpha=0.2)
    
    # Steady-state KL robust filter.
    plt.plot(time_steps, mse_kl_mean, 'c-', label='KL robust filter')
    plt.fill_between(time_steps,
                     mse_kl_mean - 0.2 * mse_kl_std,
                     mse_kl_mean + 0.2 * mse_kl_std,
                     color='c', alpha=0.2)
    
    # Risk-sensitive filter.
    plt.plot(time_steps, mse_risk_mean, color='orange', linestyle='-', label='Risk-sensitive filter')
    plt.fill_between(time_steps,
                     mse_risk_mean - 0.2 * mse_risk_std,
                     mse_risk_mean + 0.2 * mse_risk_std,
                     color='orange', alpha=0.2)
    
    plt.xlabel('Time Step')
    plt.ylabel('MSE')
    plt.title(f'MSE Mean vs. Time (dist: {dist}, noise_dist: {noise_dist})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_filename = os.path.join(results_dir, f'mse_plot_{dist}_{noise_dist}.png')
    plt.savefig(plot_filename)
    plt.show()

if __name__ == '__main__':
    main()
