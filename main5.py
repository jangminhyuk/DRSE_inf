#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main5.py

This experiment runs a closed–loop simulation where an LQR controller is applied to the system:
    xₜ₊₁ = A xₜ + B uₜ + wₜ,    yₜ = C xₜ + vₜ,
with B defined as:
    [[0, 0],
     [0, 0],
     [1, 0],
     [0, 1]].
A steady–state LQR gain is computed offline using specified cost matrices Q_lqr and R_lqr.
The nominal parameters are obtained via EM (covariances only) while the known mean
vectors (x0_mean, mu_w, mu_v) are used for the means.
In this experiment the robust parameter (θ) is fixed to 1.0.
We study the computation time scalability of each filter by varying the time horizon T over:
    2, 5, 10, …, 50.
For each T, 10 independent experiments are run.
For each experiment, the computation time (per filter) is measured.
The mean and standard deviation of the computation time (for each filter) are saved in the folder "estimator5".

Usage example:
    python main5.py --dist normal --noise_dist normal --num_sim 1 --num_exp 10
"""

import numpy as np
import argparse
import os
import pickle
import time
from joblib import Parallel, delayed
from pykalman import KalmanFilter
from scipy.linalg import solve_discrete_are, expm

# Import filter implementations from the LQR_with_estimator folder.
from LQR_with_estimator.KF import KF
from LQR_with_estimator.KF_inf import KF_inf
from LQR_with_estimator.DRKF_ours_inf import DRKF_ours_inf
from LQR_with_estimator.BCOT import BCOT 
from LQR_with_estimator.KL import KL
from LQR_with_estimator.risk_sensitive import RiskSensitive

# --- Distribution Sampling Functions ---
def normal(mu, Sigma, N=1):
    return np.random.multivariate_normal(mu[:, 0], Sigma, size=N).T

def uniform(a, b, N=1):
    n = a.shape[0]
    x = a + (b - a) * np.random.rand(N, n)
    return x.T

def quad_inverse(x, b, a):
    row, col = x.shape
    for i in range(row):
        for j in range(col):
            beta = (a[j] + b[j]) / 2.0
            alpha = 12.0 / ((b[j] - a[j]) ** 3)
            tmp = 3 * x[i][j] / alpha - (beta - a[j]) ** 3
            if tmp >= 0:
                x[i][j] = beta + tmp ** (1.0/3.0)
            else:
                x[i][j] = beta - (-tmp) ** (1.0/3.0)
    return x

def quadratic(w_max, w_min, N=1):
    n = w_min.shape[0]
    x = np.random.rand(N, n)
    x = quad_inverse(x, w_max, w_min)
    return x.T

# --- Data Generation ---
def generate_data(T, nx, ny, A, C, mu_w, Sigma_w, mu_v, M,
                  x0_mean, x0_cov, x0_max, x0_min, w_max, w_min, v_max, v_min, dist):
    x_true_all = np.zeros((T + 1, nx, 1))
    y_all = np.zeros((T, ny, 1))
    if dist == "normal":
        x_true = normal(x0_mean, x0_cov)
    elif dist == "quadratic":
        x_true = quadratic(x0_max, x0_min)
    x_true_all[0] = x_true
    for t in range(T):
        if dist == "normal":
            true_w = normal(mu_w, Sigma_w)
            true_v = normal(mu_v, M)
        elif dist == "quadratic":
            true_w = quadratic(w_max, w_min)
            true_v = quadratic(v_max, v_min)
        y_t = C @ x_true + true_v
        y_all[t] = y_t
        # Closed-loop: control u will be applied externally.
        x_true = A @ x_true + true_w
        x_true_all[t+1] = x_true
    return x_true_all, y_all

def save_data(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def is_stabilizable(A, B, tol=1e-9):
    n = A.shape[0]
    eigenvals, _ = np.linalg.eig(A)
    for eig in eigenvals:
        if np.abs(eig) >= 1 - tol:
            M_mat = np.hstack([eig * np.eye(n) - A, B])
            if np.linalg.matrix_rank(M_mat, tol) < n:
                return False
    return True

def is_detectable(A, C, tol=1e-9):
    n = A.shape[0]
    eigenvals, _ = np.linalg.eig(A)
    for eig in eigenvals:
        if np.abs(eig) >= 1 - tol:
            M_mat = np.vstack([eig * np.eye(n) - A, C])
            if np.linalg.matrix_rank(M_mat, tol) < n:
                return False
    return True

def is_positive_definite(M, tol=1e-9):
    if not np.allclose(M, M.T, atol=tol):
        return False
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False

def enforce_positive_definiteness(Sigma, epsilon=1e-4):
    Sigma = (Sigma + Sigma.T) / 2
    eigvals = np.linalg.eigvalsh(Sigma)
    min_eig = np.min(eigvals)
    if min_eig < epsilon:
        Sigma += (epsilon - min_eig) * np.eye(Sigma.shape[0])
    return Sigma

# --- LQR Controller & Cost Computation ---
def compute_lqr_gain(A, B, Q_lqr, R_lqr):
    P = solve_discrete_are(A, B, Q_lqr, R_lqr)
    K = np.linalg.inv(B.T @ P @ B + R_lqr) @ (B.T @ P @ A)
    return K

def compute_lqr_cost(result, Q_lqr, R_lqr, K_lqr):
    x = result['state_traj']
    x_est = result['est_state_traj']
    T_sim = x.shape[0] - 1
    cost = 0.0
    for t in range(T_sim):
        u = -K_lqr @ x_est[t]
        cost += (x[t].T @ Q_lqr @ x[t])[0,0] + (u.T @ R_lqr @ u)[0,0]
    cost += (x[T_sim].T @ Q_lqr @ x[T_sim])[0,0]
    return cost

# --- Experiment Function ---
def run_experiment(exp_idx, dist, noise_dist, num_sim, T, seed_base, robust_val, _unused):
    np.random.seed(seed_base + exp_idx)
    
    # System dimensions.
    nx = 4; nw = 4; ny = 2; dt = 0.5
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    system_data = (A, C)
    
    # B matrix for control.
    B = np.array([[0, 0],
                  [0, 0],
                  [1, 0],
                  [0, 1]])
    
    # True distribution parameters.
    if dist == "normal":
        mu_w = 0.0 * np.ones((nw, 1))
        Sigma_w = 0.05 * np.eye(nw)
        x0_mean = 0.0 * np.ones((nx, 1))
        x0_cov = 0.05 * np.eye(nx)
        w_max = None; w_min = None; x0_max = None; x0_min = None
    elif dist == "quadratic":
        w_max = 1.0 * np.ones(nx)
        w_min = -2.0 * np.ones(nx)
        mu_w = (0.5 * (w_max + w_min))[:, None]
        Sigma_w = 3.0/20.0 * np.diag((w_max - w_min)**2)
        x0_max = 0.21 * np.ones(nx)
        x0_min = 0.19 * np.ones(nx)
        x0_mean = (0.5 * (x0_max + x0_min))[:, None]
        x0_cov = 3.0/20.0 * np.diag((x0_max - x0_min)**2)
    else:
        raise ValueError("Unsupported process noise distribution.")
    
    if noise_dist == "normal":
        mu_v = 0.0 * np.ones((ny, 1))
        M = 0.05 * np.eye(ny)
        v_max = None; v_min = None
    elif noise_dist == "quadratic":
        v_min = -1.0 * np.ones(ny)
        v_max = 2.0 * np.ones(ny)
        mu_v = (0.5 * (v_max + v_min))[:, None]
        M = 3.0/20.0 * np.diag((v_max - v_min)**2)
    else:
        raise ValueError("Unsupported measurement noise distribution.")
    
    # Generate data for EM.
    N_data = 20
    _, y_all_em = generate_data(N_data, nx, ny, A, C,
                                mu_w, Sigma_w, mu_v, M,
                                x0_mean, x0_cov, x0_max, x0_min,
                                w_max, w_min, v_max, v_min, dist)
    y_all_em = y_all_em.squeeze()
    
    # EM Estimation.
    mu_w_hat = np.zeros((nx, 1))
    mu_v_hat = np.zeros((ny, 1))
    mu_x0_hat = x0_mean.copy()
    Sigma_w_hat = np.eye(nx)
    Sigma_v_hat = np.eye(ny)
    Sigma_x0_hat = x0_cov.copy()
    
    kf_em = KalmanFilter(transition_matrices=A,
                           observation_matrices=C,
                           transition_covariance=Sigma_w_hat,
                           observation_covariance=Sigma_v_hat,
                           transition_offsets=mu_w_hat.squeeze(),
                           observation_offsets=mu_v_hat.squeeze(),
                           initial_state_mean=mu_x0_hat.squeeze(),
                           initial_state_covariance=Sigma_x0_hat,
                           em_vars=['transition_covariance', 'observation_covariance',
                                    'transition_offsets', 'observation_offsets'])
    max_iter = 100
    eps_log = 1e-4
    loglikelihoods = np.zeros(max_iter)
    for i in range(max_iter):
        kf_em = kf_em.em(X=y_all_em, n_iter=1)
        loglikelihoods[i] = kf_em.loglikelihood(y_all_em)
        Sigma_w_hat = kf_em.transition_covariance
        Sigma_v_hat = kf_em.observation_covariance
        mu_x0_hat = kf_em.initial_state_mean
        Sigma_x0_hat = kf_em.initial_state_covariance
        if i > 0 and (loglikelihoods[i] - loglikelihoods[i-1] <= eps_log):
            break
    Sigma_w_hat = enforce_positive_definiteness(Sigma_w_hat)
    Sigma_v_hat = enforce_positive_definiteness(Sigma_v_hat)
    Sigma_x0_hat = enforce_positive_definiteness(Sigma_x0_hat)
    
    # Nominal parameters.
    nominal_mu_w    = mu_w
    nominal_Sigma_w = Sigma_w_hat.copy()
    nominal_mu_v    = mu_v
    nominal_M       = Sigma_v_hat.copy()
    nominal_x0_mean = x0_mean
    nominal_x0_cov  = Sigma_x0_hat.copy()
    
    # Compute LQR gain.
    Q_lqr = np.eye(nx)
    R_lqr = np.eye(B.shape[1])
    K_lqr = compute_lqr_gain(A, B, Q_lqr, R_lqr)
    if not is_stabilizable(A, B):
        print("Warning: (A, B) is not stabilizable!")
        exit()
    if not is_detectable(A, C):
        print("Warning: (A, C) is not detectable!")
        exit()
    if not is_positive_definite(nominal_Sigma_w):
        print("Warning: nominal_Sigma_w is not positive definite!")
        exit()
    if not is_positive_definite(nominal_M):
        print("Warning: nominal_M (noise covariance) is not positive definite!")
        exit()
    
    # --- Simulation Functions for Each Filter ---
    def run_simulation_finite(sim_idx_local):
        estimator = KF(
            T=T, dist=dist, noise_dist=noise_dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_M=M,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_M=nominal_M,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min)
        estimator.K_lqr = K_lqr
        res = estimator.forward()
        cost = compute_lqr_cost(res, Q_lqr, R_lqr, K_lqr)
        return res['mse'], cost

    def run_simulation_inf_kf(sim_idx_local):
        estimator = KF_inf(
            T=T, dist=dist, noise_dist=noise_dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_M=M,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_M=nominal_M,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min)
        estimator.K_lqr = K_lqr
        res = estimator.forward()
        cost = compute_lqr_cost(res, Q_lqr, R_lqr, K_lqr)
        return res['mse'], cost

    def run_simulation_inf_drkf(sim_idx_local):
        estimator = DRKF_ours_inf(
            T=T, dist=dist, noise_dist=noise_dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=M,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_M,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            theta_x=robust_val, theta_v=robust_val)
        estimator.K_lqr = K_lqr
        res = estimator.forward()
        cost = compute_lqr_cost(res, Q_lqr, R_lqr, K_lqr)
        return res['mse'], cost

    def run_simulation_bcot(sim_idx_local):
        estimator = BCOT(
            T=T, dist=dist, noise_dist=noise_dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_M=M,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_M=nominal_M,
            radius=robust_val, maxit=20,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min)
        estimator.K_lqr = K_lqr
        res = estimator.forward()
        cost = compute_lqr_cost(res, Q_lqr, R_lqr, K_lqr)
        return res['mse'], cost

    def run_simulation_kl(sim_idx_local):
        kl_maxit = 2
        estimator = KL(
            T=T, dist=dist, noise_dist=noise_dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_M=M,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_M=nominal_M,
            radius=robust_val, maxit=kl_maxit,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min)
        estimator.K_lqr = K_lqr
        res = estimator.forward()
        cost = compute_lqr_cost(res, Q_lqr, R_lqr, K_lqr)
        return res['mse'], cost

    def run_simulation_risk(sim_idx_local):
        estimator = RiskSensitive(
            T=T, dist=dist, noise_dist=noise_dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_M=M,
            nominal_x0_mean=x0_mean,  # known initial state mean
            nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=mu_w,        # known process noise mean
            nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=mu_v,        # known measurement noise mean
            nominal_M=nominal_M,
            theta_rs=robust_val,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min)
        estimator.K_lqr = K_lqr
        res = estimator.forward()
        cost = compute_lqr_cost(res, Q_lqr, R_lqr, K_lqr)
        return res['mse'], cost

    # --- Measure computation time for each filter ---
    start = time.perf_counter()
    results_finite = [run_simulation_finite(i) for i in range(num_sim)]
    time_finite = time.perf_counter() - start

    start = time.perf_counter()
    results_inf = [run_simulation_inf_kf(i) for i in range(num_sim)]
    time_inf = time.perf_counter() - start

    start = time.perf_counter()
    results_drkf = [run_simulation_inf_drkf(i) for i in range(num_sim)]
    time_drkf = time.perf_counter() - start

    start = time.perf_counter()
    results_bcot = [run_simulation_bcot(i) for i in range(num_sim)]
    time_bcot = time.perf_counter() - start

    start = time.perf_counter()
    results_kl = [run_simulation_kl(i) for i in range(num_sim)]
    time_kl = time.perf_counter() - start

    start = time.perf_counter()
    results_risk = [run_simulation_risk(i) for i in range(num_sim)]
    time_risk = time.perf_counter() - start

    # (Optional) Also compute simulation results.
    mse_finite_all, cost_finite_all = zip(*results_finite)
    mse_inf_all, cost_inf_all = zip(*results_inf)
    mse_drkf_all, cost_drkf_all = zip(*results_drkf)
    mse_bcot_all, cost_bcot_all = zip(*results_bcot)
    mse_kl_all, cost_kl_all = zip(*results_kl)
    mse_risk_all, cost_risk_all = zip(*results_risk)
    
    return {
        'finite': np.mean(np.array(mse_finite_all), axis=0),
        'inf': np.mean(np.array(mse_inf_all), axis=0),
        'drkf_inf': np.mean(np.array(mse_drkf_all), axis=0),
        'bcot': np.mean(np.array(mse_bcot_all), axis=0),
        'kl': np.mean(np.array(mse_kl_all), axis=0),
        'risk': np.mean(np.array(mse_risk_all), axis=0),
        'cost': {
            'finite': np.mean(np.array(cost_finite_all)),
            'inf': np.mean(np.array(cost_inf_all)),
            'drkf_inf': np.mean(np.array(cost_drkf_all)),
            'bcot': np.mean(np.array(cost_bcot_all)),
            'kl': np.mean(np.array(cost_kl_all)),
            'risk': np.mean(np.array(cost_risk_all))
        },
        'time': {
            'finite': time_finite,
            'inf': time_inf,
            'drkf_inf': time_drkf,
            'bcot': time_bcot,
            'kl': time_kl,
            'risk': time_risk
        }
    }

# --- Main Routine for Scalability Experiment ---
def main(dist, noise_dist, num_sim, num_exp):
    seed_base = 2024
    robust_val = 1.0  # Fixed robust parameter.
    # Define time horizon values.
    T_values = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    scalability_results = {}
    
    for T in T_values:
        print(f"\nRunning experiments for Time horizon T = {T}")
        experiments = Parallel(n_jobs=-1)(
            delayed(run_experiment)(exp_idx, dist, noise_dist, num_sim, T, seed_base, robust_val, 0)
            for exp_idx in range(num_exp)
        )
        # Aggregate computation times for each filter.
        filter_keys = ['finite', 'inf', 'drkf_inf', 'bcot', 'kl', 'risk']
        times = {key: [] for key in filter_keys}
        for exp in experiments:
            for key in filter_keys:
                times[key].append(exp['time'][key])
        # Compute mean and std for each filter.
        filter_times = {}
        for key in filter_keys:
            filter_times[key] = {'mean_time': np.mean(times[key]),
                                 'std_time': np.std(times[key])}
            print(f"T = {T}, Filter {key}: Mean time = {filter_times[key]['mean_time']:.4f} sec, Std = {filter_times[key]['std_time']:.4f} sec")
        scalability_results[T] = filter_times
    
    # Save scalability results in the estimator5 folder.
    results_path = "./results/estimator5/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    save_data(os.path.join(results_path, f'scalability_results_{dist}_{noise_dist}.pkl'), scalability_results)
    print("\nScalability experiments completed. Results saved in estimator5 folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', default="normal", type=str,
                        help="Disturbance distribution (normal or quadratic)")
    parser.add_argument('--noise_dist', default="normal", type=str,
                        help="Measurement noise distribution (normal or quadratic)")
    parser.add_argument('--num_sim', default=1, type=int,
                        help="Number of simulation runs per experiment")
    parser.add_argument('--num_exp', default=10, type=int,
                        help="Number of independent experiments per time horizon")
    args = parser.parse_args()
    main(args.dist, args.noise_dist, args.num_sim, args.num_exp)
