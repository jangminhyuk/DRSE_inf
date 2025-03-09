#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot.py

This script loads the saved MSE mean and standard deviation data (from the
finite–horizon KF, steady–state KF, steady–state DRKF, and H∞ filter experiments)
and plots the evolution of the MSE over time. Each estimator’s MSE mean is plotted
with a shaded region indicating ±0.2× the standard deviation.

Usage:
    python plot.py --dist normal --noise_dist normal --results_dir ./results/estimator/
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
    parser.add_argument('--results_dir', default='./results/estimator/', type=str,
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
    file_hinf_mean     = os.path.join(results_dir, f'hinf_mse_mean_{dist}_{noise_dist}.pkl')
    file_hinf_std      = os.path.join(results_dir, f'hinf_mse_std_{dist}_{noise_dist}.pkl')
    
    # Load data.
    mse_kf_mean       = load_pickle(file_kf_mean)
    mse_kf_std        = load_pickle(file_kf_std)
    mse_kf_inf_mean   = load_pickle(file_kf_inf_mean)
    mse_kf_inf_std    = load_pickle(file_kf_inf_std)
    mse_drkf_inf_mean = load_pickle(file_drkf_inf_mean)
    mse_drkf_inf_std  = load_pickle(file_drkf_inf_std)
    mse_hinf_mean     = load_pickle(file_hinf_mean)
    mse_hinf_std      = load_pickle(file_hinf_std)
    
    # Time axis (assumes that the saved MSE arrays have T+1 entries).
    T = len(mse_kf_mean) - 1
    time_steps = np.arange(T + 1)
    
    plt.figure(figsize=(10, 6))
    
    # Finite-horizon KF.
    plt.plot(time_steps, mse_kf_mean, 'b-', label='Finite-horizon KF')
    # plt.fill_between(time_steps,
    #                  mse_kf_mean - 0.2 * mse_kf_std,
    #                  mse_kf_mean + 0.2 * mse_kf_std,
    #                  color='b', alpha=0.2)
    
    # Steady-state KF.
    plt.plot(time_steps, mse_kf_inf_mean, 'r-', label='Steady-state KF')
    # plt.fill_between(time_steps,
    #                  mse_kf_inf_mean - 0.2 * mse_kf_inf_std,
    #                  mse_kf_inf_mean + 0.2 * mse_kf_inf_std,
    #                  color='g', alpha=0.2)
    
    # Steady-state DRKF.
    plt.plot(time_steps, mse_drkf_inf_mean, 'g-', label='Steady-state DRKF')
    # plt.fill_between(time_steps,
    #                  mse_drkf_inf_mean - 0.2 * mse_drkf_inf_std,
    #                  mse_drkf_inf_mean + 0.2 * mse_drkf_inf_std,
    #                  color='r', alpha=0.2)
    
    # Steady-state H∞ filter.
    plt.plot(time_steps, mse_hinf_mean, 'm-', label='H∞')
    # plt.fill_between(time_steps,
    #                  mse_hinf_mean - 0.2 * mse_hinf_std,
    #                  mse_hinf_mean + 0.2 * mse_hinf_std,
    #                  color='m', alpha=0.2)
    
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
