#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main4.py

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
Then, six filters (KF, KF_inf, DRKF, BCOT, KL, and risk–sensitive filter)
(from the folder LQR_with_estimator) are run in closed–loop with exactly the same controller
for each candidate robust parameter value.
For each simulation run, the mean squared error (MSE) trajectory and the LQR cost
computed as:
    J = Σₜ (x[t]ᵀ Q_lqr x[t] + u[t]ᵀ R_lqr u[t]) + x[T]ᵀ Q_lqr x[T]
are returned. For each filter the candidate robust parameter with the lowest average
LQR cost is chosen as “optimal.” Finally, the performance (terminal MSE and cost)
of each filter using its optimal robust parameter is saved for comparison.

Usage example:
    python main4.py --dist normal --noise_dist normal --num_sim 500 --horizon 20 --num_exp 10
"""

import numpy as np
import argparse
import os
import pickle
from joblib import Parallel, delayed
from pykalman import KalmanFilter
from scipy.linalg import solve_discrete_are

# Import filter implementations from the LQR_with_estimator folder.
from LQR_with_estimator.KF import KF
from LQR_with_estimator.KF_inf import KF_inf
from LQR_with_estimator.DRKF_ours_inf import DRKF_ours_inf
from LQR_with_estimator.BCOT import BCOT 
from LQR_with_estimator.KL import KL
from LQR_with_estimator.risk_sensitive import RiskSensitive

# --- Distribution Sampling Functions (for true data generation) ---
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

# Function to generate the true states and measurements.
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

def enforce_positive_definiteness(Sigma, epsilon=1e-6):
    Sigma = (Sigma + Sigma.T) / 2
    eigvals = np.linalg.eigvalsh(Sigma)
    min_eig = np.min(eigvals)
    if min_eig < epsilon:
        Sigma += (epsilon - min_eig) * np.eye(Sigma.shape[0])
    return Sigma

# --- LQR Controller Computation ---
def compute_lqr_gain(A, B, Q_lqr, R_lqr):
    P = solve_discrete_are(A, B, Q_lqr, R_lqr)
    K = np.linalg.inv(B.T @ P @ B + R_lqr) @ (B.T @ P @ A)
    return K

# --- LQR Cost Computation ---
def compute_lqr_cost(result, Q_lqr, R_lqr, K_lqr):
    x = result['state_traj']
    x_est = result['est_state_traj']
    T = x.shape[0] - 1
    cost = 0.0
    for t in range(T):
        u = -K_lqr @ x_est[t]
        cost += (x[t].T @ Q_lqr @ x[t])[0,0] + (u.T @ R_lqr @ u)[0,0]
    cost += (x[T].T @ Q_lqr @ x[T])[0,0]
    return cost

# --- Experiment Function (one independent experiment) ---
def run_experiment(exp_idx, dist, noise_dist, num_sim, T, seed_base, robust_val, _unused):
    np.random.seed(seed_base + exp_idx)
    
    # System dimensions.
    nx = 4; nw = 4; ny = 2; dt = 0.2
    # Correct A matrix for tracking [x1, x2, x1dot, x2dot]:
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
    
    # --- True Distribution Parameters ---
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
    
    # --- Generate Data for EM ---
    N_data = 30
    _, y_all_em = generate_data(N_data, nx, ny, A, C,
                                mu_w, Sigma_w, mu_v, M,
                                x0_mean, x0_cov, x0_max, x0_min,
                                w_max, w_min, v_max, v_min, dist)
    y_all_em = y_all_em.squeeze()
    
    # --- EM Estimation of Nominal Covariances ---
    mu_w_hat = np.zeros((nx, 1))
    mu_v_hat = np.zeros((ny, 1))
    mu_x0_hat = x0_mean.copy()
    Sigma_w_hat = np.eye(nx)
    Sigma_v_hat = np.eye(ny)
    Sigma_x0_hat = x0_cov.copy()
    
    from pykalman import KalmanFilter
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
    
    # Nominal parameters: use known mean values and EM–estimated covariances.
    nominal_mu_w    = mu_w
    nominal_Sigma_w = Sigma_w_hat.copy()
    nominal_mu_v    = mu_v
    nominal_M       = Sigma_v_hat.copy()
    nominal_x0_mean = x0_mean
    nominal_x0_cov  = Sigma_x0_hat.copy()
    
    # --- Compute LQR Gain ---
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
    
    results_finite = [run_simulation_finite(i) for i in range(num_sim)]
    mse_finite_all, cost_finite_all = zip(*results_finite)
    mse_mean_finite = np.mean(np.array(mse_finite_all), axis=0)
    cost_mean_finite = np.mean(np.array(cost_finite_all))
    
    results_inf = [run_simulation_inf_kf(i) for i in range(num_sim)]
    mse_inf_all, cost_inf_all = zip(*results_inf)
    mse_mean_inf = np.mean(np.array(mse_inf_all), axis=0)
    cost_mean_inf = np.mean(np.array(cost_inf_all))
    
    results_drkf = [run_simulation_inf_drkf(i) for i in range(num_sim)]
    mse_drkf_all, cost_drkf_all = zip(*results_drkf)
    mse_mean_drkf = np.mean(np.array(mse_drkf_all), axis=0)
    cost_mean_drkf = np.mean(np.array(cost_drkf_all))
    
    results_bcot = [run_simulation_bcot(i) for i in range(num_sim)]
    mse_bcot_all, cost_bcot_all = zip(*results_bcot)
    mse_mean_bcot = np.mean(np.array(mse_bcot_all), axis=0)
    cost_mean_bcot = np.mean(np.array(cost_bcot_all))
    
    results_kl = [run_simulation_kl(i) for i in range(num_sim)]
    mse_kl_all, cost_kl_all = zip(*results_kl)
    mse_mean_kl = np.mean(np.array(mse_kl_all), axis=0)
    cost_mean_kl = np.mean(np.array(cost_kl_all))
    
    results_risk = [run_simulation_risk(i) for i in range(num_sim)]
    mse_risk_all, cost_risk_all = zip(*results_risk)
    mse_mean_risk = np.mean(np.array(mse_risk_all), axis=0)
    cost_mean_risk = np.mean(np.array(cost_risk_all))
    
    return {
        'finite': mse_mean_finite,
        'inf': mse_mean_inf,
        'drkf_inf': mse_mean_drkf,
        'bcot': mse_mean_bcot,
        'kl': mse_mean_kl,
        'risk': mse_mean_risk,
        'cost': {
            'finite': cost_mean_finite,
            'inf': cost_mean_inf,
            'drkf_inf': cost_mean_drkf,
            'bcot': cost_mean_bcot,
            'kl': cost_mean_kl,
            'risk': cost_mean_risk
        }
    }

# --- Main Routine ---
def main(dist, noise_dist, num_sim, T, num_exp):
    seed_base = 2024
    # Define candidate robust parameter values.
    robust_vals = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    all_results = {}
    for robust_val in robust_vals:
        print(f"Running experiments for robust parameter = {robust_val}")
        experiments = Parallel(n_jobs=-1)(
            delayed(run_experiment)(exp_idx, dist, noise_dist, num_sim, T, seed_base, robust_val, 0)
            for exp_idx in range(num_exp)
        )
        all_finite = np.array([exp['finite'] for exp in experiments])
        all_inf = np.array([exp['inf'] for exp in experiments])
        all_drkf = np.array([exp['drkf_inf'] for exp in experiments])
        all_bcot = np.array([exp['bcot'] for exp in experiments])
        all_kl = np.array([exp['kl'] for exp in experiments])
        all_risk = np.array([exp['risk'] for exp in experiments])
        all_cost = [exp['cost'] for exp in experiments]
        
        final_mean_finite = np.mean(all_finite, axis=0)
        final_std_finite = np.std(all_finite, axis=0)
        final_mean_inf = np.mean(all_inf, axis=0)
        final_std_inf = np.std(all_inf, axis=0)
        final_mean_drkf = np.mean(all_drkf, axis=0)
        final_std_drkf = np.std(all_drkf, axis=0)
        final_mean_bcot = np.mean(all_bcot, axis=0)
        final_std_bcot = np.std(all_bcot, axis=0)
        final_mean_kl = np.mean(all_kl, axis=0)
        final_std_kl = np.std(all_kl, axis=0)
        final_mean_risk = np.mean(all_risk, axis=0)
        final_std_risk = np.std(all_risk, axis=0)
        
        cost_keys = ['finite', 'inf', 'drkf_inf', 'bcot', 'kl', 'risk']
        final_cost = {}
        for key in cost_keys:
            final_cost[key] = np.mean([exp_cost[key] for exp_cost in all_cost])
        
        results_path = "./results/estimator2/"
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        suffix = f"{dist}_{noise_dist}_robust_{robust_val}"
        save_data(os.path.join(results_path, f'kf_mse_mean_{suffix}.pkl'), final_mean_finite)
        save_data(os.path.join(results_path, f'kf_mse_std_{suffix}.pkl'), final_std_finite)
        save_data(os.path.join(results_path, f'kf_inf_mse_mean_{suffix}.pkl'), final_mean_inf)
        save_data(os.path.join(results_path, f'kf_inf_mse_std_{suffix}.pkl'), final_std_inf)
        save_data(os.path.join(results_path, f'kf_drkf_inf_mse_mean_{suffix}.pkl'), final_mean_drkf)
        save_data(os.path.join(results_path, f'kf_drkf_inf_mse_std_{suffix}.pkl'), final_std_drkf)
        save_data(os.path.join(results_path, f'bcot_mse_mean_{suffix}.pkl'), final_mean_bcot)
        save_data(os.path.join(results_path, f'bcot_mse_std_{suffix}.pkl'), final_std_bcot)
        save_data(os.path.join(results_path, f'kl_mse_mean_{suffix}.pkl'), final_mean_kl)
        save_data(os.path.join(results_path, f'kl_mse_std_{suffix}.pkl'), final_std_kl)
        save_data(os.path.join(results_path, f'risk_sensitive_mse_mean_{suffix}.pkl'), final_mean_risk)
        save_data(os.path.join(results_path, f'risk_sensitive_mse_std_{suffix}.pkl'), final_std_risk)
        save_data(os.path.join(results_path, f'lqr_cost_{suffix}.pkl'), final_cost)
        
        all_results[robust_val] = {
            'mse': {
                'finite': final_mean_finite,
                'inf': final_mean_inf,
                'drkf_inf': final_mean_drkf,
                'bcot': final_mean_bcot,
                'kl': final_mean_kl,
                'risk': final_mean_risk
            },
            'cost': final_cost
        }
        print(f"Completed robust parameter = {robust_val}\n")
    
    # Now, for each filter, choose the robust parameter that minimizes the average LQR cost.
    # We assume that all_results is a dictionary keyed by robust value, each containing a 'cost' dict.
    filters = ['finite', 'inf', 'drkf_inf', 'bcot', 'kl', 'risk']
    optimal_results = {}
    for f in filters:
        best_val = None
        best_cost = np.inf
        for robust_val, res in all_results.items():
            current_cost = res['cost'][f]
            if current_cost < best_cost:
                best_cost = current_cost
                best_val = robust_val
        optimal_results[f] = {
            'robust_val': best_val,
            'cost': best_cost,
            'mse': all_results[best_val]['mse'][f]
        }
        print(f"Optimal robust parameter for {f}: {best_val} with cost {best_cost}")
    
    results_path = "./results/estimator2/"
    save_data(os.path.join(results_path, f'optimal_results_{dist}_{noise_dist}.pkl'), optimal_results)
    save_data(os.path.join(results_path, f'overall_results_{dist}_{noise_dist}.pkl'), all_results)
    print("LQR with state estimation experiments completed for all robust parameters.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', default="normal", type=str,
                        help="Disturbance distribution (normal or quadratic)")
    parser.add_argument('--noise_dist', default="normal", type=str,
                        help="Measurement noise distribution (normal or quadratic)")
    parser.add_argument('--num_sim', default=500, type=int,
                        help="Number of simulation runs per experiment")
    parser.add_argument('--horizon', default=20, type=int,
                        help="Time horizon T")
    parser.add_argument('--num_exp', default=10, type=int,
                        help="Number of independent experiments")
    args = parser.parse_args()
    main(args.dist, args.noise_dist, args.num_sim, args.horizon, args.num_exp)
