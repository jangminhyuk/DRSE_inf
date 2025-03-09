#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This main.py file performs the following steps for num_exp independent experiments:
 1. Generates true state and measurement trajectories from a linear system
    xₜ₊₁ = A xₜ + wₜ   and   yₜ = C xₜ + vₜ,
    using specified (true) noise distributions.
 2. Uses an EM method (via pykalman) to learn nominal distribution parameters from
    a short data batch (this is repeated for each experiment).
 3. Runs six state–estimation experiments:
     (a) Finite–horizon KF (KF.py)
     (b) Steady–state (infinite–horizon) KF (KF_inf.py)
     (c) Steady–state (infinite–horizon) DRKF (DRKF_ours_inf.py)
     (d) Steady–state (infinite–horizon) BCOT filter (BCOT.py)
     (e) Steady–state (infinite–horizon) KL robust filter (KL.py)
     (f) Steady–state (infinite–horizon) risk–sensitive filter (risk_sensitive.py)
 4. For each estimator, the MSE over time is computed by averaging num_sim simulation runs.
 5. Finally, the experiment is repeated num_exp times (in parallel) and the results are
    averaged over experiments. The final averaged MSE mean and standard deviation are saved.
    
Usage example:
    python main.py --dist normal --noise_dist normal --num_sim 500 --horizon 20 --num_exp 10
"""

import numpy as np
import argparse
import os
import pickle
from joblib import Parallel, delayed
from pykalman import KalmanFilter
from estimator.KF import KF
from estimator.KF_inf import KF_inf
from estimator.DRKF_ours_inf import DRKF_ours_inf
from estimator.BCOT import BCOT 
from estimator.KL import KL
from estimator.risk_sensitive import RiskSensitive

# --- Distribution Sampling Functions (for true data generation) ---
def normal(mu, Sigma, N=1):
    # mu shape: (n,1)
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

def quadratic(wmax, wmin, N=1):
    n = wmin.shape[0]
    x = np.random.rand(N, n)
    x = quad_inverse(x, wmax, wmin)
    return x.T

# Function to generate the true states and measurements
def generate_data(T, nx, ny, A, C, mu_w, Sigma_w, mu_v, M,
                  x0_mean, x0_cov, x0_max, x0_min, w_max, w_min, v_max, v_min, dist):
    x_true_all = np.zeros((T + 1, nx, 1))
    y_all = np.zeros((T, ny, 1))
    # Initialize the true state
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
        # Here, the system is: xₜ₊₁ = A xₜ + wₜ
        x_true = A @ x_true + true_w
        x_true_all[t+1] = x_true
    return x_true_all, y_all

def save_data(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def effective_noise_parameters(B, mu_w, Sigma_w):
    effective_mu = B @ mu_w
    effective_Sigma = B @ Sigma_w @ B.T
    return effective_mu, effective_Sigma

def is_stabilizable(A, B, tol=1e-9):
    n = A.shape[0]
    eigenvals, _ = np.linalg.eig(A)
    for eig in eigenvals:
        if np.abs(eig) >= 1 - tol:
            M = np.hstack([eig * np.eye(n) - A, B])
            if np.linalg.matrix_rank(M, tol) < n:
                return False
    return True

def is_detectable(A, C, tol=1e-9):
    n = A.shape[0]
    eigenvals, _ = np.linalg.eig(A)
    for eig in eigenvals:
        if np.abs(eig) >= 1 - tol:
            M = np.vstack([eig * np.eye(n) - A, C])
            if np.linalg.matrix_rank(M, tol) < n:
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
    
# --- Experiment Function (one independent experiment) ---
def run_experiment(exp_idx, dist, noise_dist, num_sim, T, seed_base, theta_x, theta_v, _unused):
    # Set a unique seed for this experiment.
    np.random.seed(seed_base + exp_idx)
    
    # System dimensions and matrices for 2D tracking.
    nx = 4
    nw = 4
    ny = 2
    dt = 1
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    system_data = (A, C)
    
    # --- True Distribution Parameters ---
    if dist == "normal":
        mu_w = 0.0 * np.ones((nw, 1))
        Sigma_w = 0.2 * np.eye(nw)
        x0_mean = 0.0 * np.ones((nx, 1))
        x0_cov = 0.2 * np.eye(nx)
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
        M = 0.2 * np.eye(ny)
        v_max = None; v_min = None
    elif noise_dist == "quadratic":
        v_min = -1.0 * np.ones(ny)
        v_max = 2.0 * np.ones(ny)
        mu_v = (0.5 * (v_max + v_min))[:, None]
        M = 3.0/20.0 * np.diag((v_max - v_min)**2)
    else:
        raise ValueError("Unsupported measurement noise distribution.")
    
    # --- Generate Data for EM ---
    N_data = 10
    _, y_all_em = generate_data(N_data, nx, ny, A, C,
                                mu_w, Sigma_w, mu_v, M,
                                x0_mean, x0_cov, x0_max, x0_min,
                                w_max, w_min, v_max, v_min, dist)
    y_all_em = y_all_em.squeeze()  # shape: (N_data, ny)
    
    # --- EM Estimation of Nominal Parameters ---
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
                           em_vars=[
                               'transition_covariance', 'observation_covariance',
                               'transition_offsets', 'observation_offsets',
                           ])
    
    max_iter = 50
    eps_log = 1e-4
    loglikelihoods = np.zeros(max_iter)
    for i in range(max_iter):
        kf_em = kf_em.em(X=y_all_em, n_iter=1)
        loglikelihoods[i] = kf_em.loglikelihood(y_all_em)
        Sigma_w_hat = kf_em.transition_covariance
        Sigma_v_hat = kf_em.observation_covariance
        mu_w_hat = kf_em.transition_offsets
        mu_v_hat = kf_em.observation_offsets
        mu_x0_hat = kf_em.initial_state_mean
        Sigma_x0_hat = kf_em.initial_state_covariance
        if i > 0 and (loglikelihoods[i] - loglikelihoods[i-1] <= eps_log):
            break

    # Nominal parameters: use the known mean vectors and the EM–estimated covariances.
    nominal_mu_w    = mu_w            # known process noise mean
    nominal_Sigma_w = Sigma_w_hat.copy() + 0.01 * np.eye(nw)  # EM covariance estimate
    nominal_mu_v    = mu_v            # known measurement noise mean
    nominal_M       = Sigma_v_hat.copy()  # EM measurement covariance estimate
    nominal_x0_mean = x0_mean         # known initial state mean
    nominal_x0_cov  = Sigma_x0_hat.copy()  # EM initial covariance estimate
    
    try:
        B_temp = np.linalg.cholesky(nominal_Sigma_w)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(nominal_Sigma_w)
        sqrt_eigvals = np.sqrt(np.maximum(eigvals, 0))
        B_temp = eigvecs @ np.diag(sqrt_eigvals)
    
    if not is_stabilizable(A, B_temp):
        print("Warning: The pair (A, sqrt(nominal_Sigma_w)) is not stabilizable!")
        exit()
    
    if not is_detectable(A, C):
        print("Warning: The pair (A, C) is not detectable!")
        exit()
        
    if not is_positive_definite(nominal_Sigma_w):
        print("Warning: nominal_Sigma_w is not positive definite!")
        exit()
    
    if not is_positive_definite(nominal_M):
        print("Warning: nominal_M (noise covariance) is not positive definite!")
        exit()
    
    # --- Simulation Functions ---
    def run_simulation_finite(sim_idx_local):
        kf_estimator = KF(T, dist, noise_dist, system_data,
                          true_x0_mean=x0_mean, true_x0_cov=x0_cov,
                          true_mu_w=mu_w, true_Sigma_w=Sigma_w,
                          true_mu_v=mu_v, true_M=M,
                          nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
                          nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
                          nominal_mu_v=nominal_mu_v, nominal_M=nominal_M,
                          x0_max=x0_max, x0_min=x0_min,
                          w_max=w_max, w_min=w_min,
                          v_max=v_max, v_min=v_min)
        result = kf_estimator.forward()
        return result['mse']
    
    def run_simulation_inf_kf(sim_idx_local):
        kf_inf_estimator = KF_inf(T, dist, noise_dist, system_data,
                                  true_x0_mean=x0_mean, true_x0_cov=x0_cov,
                                  true_mu_w=mu_w, true_Sigma_w=Sigma_w,
                                  true_mu_v=mu_v, true_M=M,
                                  nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
                                  nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
                                  nominal_mu_v=nominal_mu_v, nominal_M=nominal_M,
                                  x0_max=x0_max, x0_min=x0_min,
                                  w_max=w_max, w_min=w_min,
                                  v_max=v_max, v_min=v_min)
        result = kf_inf_estimator.forward()
        return result['mse']
    
    def run_simulation_inf_drkf(sim_idx_local):
        drkf_inf_estimator = DRKF_ours_inf(T, dist, noise_dist, system_data,
                                          true_x0_mean=x0_mean, true_x0_cov=x0_cov,
                                          true_mu_w=mu_w, true_Sigma_w=Sigma_w,
                                          true_mu_v=mu_v, true_Sigma_v=M,
                                          nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
                                          nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
                                          nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_M,
                                          x0_max=x0_max, x0_min=x0_min,
                                          w_max=w_max, w_min=w_min,
                                          v_max=v_max, v_min=v_min,
                                          theta_x=theta_x, theta_v=theta_v)
        result = drkf_inf_estimator.forward()
        return result['mse']
    
    bcot_radius = 1.0
    bcot_maxit = 20
    bcot_estimator = BCOT(T, dist, noise_dist, system_data,
                          true_x0_mean=x0_mean, true_x0_cov=x0_cov,
                          true_mu_w=mu_w, true_Sigma_w=Sigma_w,
                          true_mu_v=mu_v, true_M=M,
                          nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
                          nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
                          nominal_mu_v=nominal_mu_v, nominal_M=nominal_M,
                          radius=bcot_radius, maxit=bcot_maxit,
                          x0_max=x0_max, x0_min=x0_min,
                          w_max=w_max, w_min=w_min,
                          v_max=v_max, v_min=v_min)
    
    def run_simulation_bcot(sim_idx_local):
        result = bcot_estimator.forward()
        return result['mse']
    
    kl_radius = 1.0
    kl_maxit = 2  # use low max iterations to prevent divergence.
    kl_estimator = KL(T, dist, noise_dist, system_data,
                      true_x0_mean=x0_mean, true_x0_cov=x0_cov,
                      true_mu_w=mu_w, true_Sigma_w=Sigma_w,
                      true_mu_v=mu_v, true_M=M,
                      nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
                      nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
                      nominal_mu_v=nominal_mu_v, nominal_M=nominal_M,
                      radius=kl_radius, maxit=kl_maxit,
                      x0_max=x0_max, x0_min=x0_min,
                      w_max=w_max, w_min=w_min,
                      v_max=v_max, v_min=v_min)
    
    def run_simulation_kl(sim_idx_local):
        result = kl_estimator.forward()
        return result['mse']
    
    # New Risk-Sensitive estimator.
    # and EM-estimated covariances.
    theta_rs = 1.0  # risk sensitivity parameter; adjust as needed.
    risk_estimator = RiskSensitive(T, dist, noise_dist, system_data,
                                   true_x0_mean=x0_mean, true_x0_cov=x0_cov,
                                   true_mu_w=mu_w, true_Sigma_w=Sigma_w,
                                   true_mu_v=mu_v, true_M=M,
                                   nominal_x0_mean=nominal_x0_mean,  
                                   nominal_x0_cov=nominal_x0_cov,
                                   nominal_mu_w=nominal_mu_w,      
                                   nominal_Sigma_w=nominal_Sigma_w,
                                   nominal_mu_v=nominal_mu_v,
                                   nominal_M=nominal_M,
                                   theta_rs=theta_rs,
                                   x0_max=x0_max, x0_min=x0_min,
                                   w_max=w_max, w_min=w_min,
                                   v_max=v_max, v_min=v_min)
    
    def run_simulation_risk(sim_idx_local):
        result = risk_estimator.forward()
        return result['mse']
    
    # --- Run Simulation Experiments Sequentially for each estimator ---
    mse_finite_all = np.array([run_simulation_finite(i) for i in range(num_sim)])
    mse_mean_finite = np.mean(mse_finite_all, axis=0)
    
    mse_inf_all = np.array([run_simulation_inf_kf(i) for i in range(num_sim)])
    mse_mean_inf = np.mean(mse_inf_all, axis=0)
    
    mse_drkf_inf_all = np.array([run_simulation_inf_drkf(i) for i in range(num_sim)])
    mse_mean_drkf_inf = np.mean(mse_drkf_inf_all, axis=0)
    
    mse_bcot_all = np.array([run_simulation_bcot(i) for i in range(num_sim)])
    mse_mean_bcot = np.mean(mse_bcot_all, axis=0)
    
    mse_kl_all = np.array([run_simulation_kl(i) for i in range(num_sim)])
    mse_mean_kl = np.mean(mse_kl_all, axis=0)
    
    mse_risk_all = np.array([run_simulation_risk(i) for i in range(num_sim)])
    mse_mean_risk = np.mean(mse_risk_all, axis=0)
    
    # Return a dictionary with the average MSE curves from this experiment.
    return {
        'finite': mse_mean_finite,
        'inf': mse_mean_inf,
        'drkf_inf': mse_mean_drkf_inf,
        'bcot': mse_mean_bcot,
        'kl': mse_mean_kl,
        'risk': mse_mean_risk
    }

# --- Main Routine ---
def main(dist, noise_dist, num_sim, T, num_exp):
    seed_base = 2024
    theta_x = 1.0
    theta_v = 1.0 
    
    experiments = Parallel(n_jobs=-1)(
        delayed(run_experiment)(exp_idx, dist, noise_dist, num_sim, T, seed_base, theta_x, theta_v, 0)
        for exp_idx in range(num_exp)
    )
    
    all_finite = np.array([exp['finite'] for exp in experiments])
    all_inf = np.array([exp['inf'] for exp in experiments])
    all_drkf_inf = np.array([exp['drkf_inf'] for exp in experiments])
    all_bcot = np.array([exp['bcot'] for exp in experiments])
    all_kl = np.array([exp['kl'] for exp in experiments])
    all_risk = np.array([exp['risk'] for exp in experiments])
    
    final_mean_finite = np.mean(all_finite, axis=0)
    final_std_finite = np.std(all_finite, axis=0)
    
    final_mean_inf = np.mean(all_inf, axis=0)
    final_std_inf = np.std(all_inf, axis=0)
    
    final_mean_drkf_inf = np.mean(all_drkf_inf, axis=0)
    final_std_drkf_inf = np.std(all_drkf_inf, axis=0)
    
    final_mean_bcot = np.mean(all_bcot, axis=0)
    final_std_bcot = np.std(all_bcot, axis=0)
    
    final_mean_kl = np.mean(all_kl, axis=0)
    final_std_kl = np.std(all_kl, axis=0)
    
    final_mean_risk = np.mean(all_risk, axis=0)
    final_std_risk = np.std(all_risk, axis=0)
    
    results_path = "./results/estimator2/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    save_data(os.path.join(results_path, f'kf_mse_mean_{dist}_{noise_dist}.pkl'), final_mean_finite)
    save_data(os.path.join(results_path, f'kf_mse_std_{dist}_{noise_dist}.pkl'), final_std_finite)
    
    save_data(os.path.join(results_path, f'kf_inf_mse_mean_{dist}_{noise_dist}.pkl'), final_mean_inf)
    save_data(os.path.join(results_path, f'kf_inf_mse_std_{dist}_{noise_dist}.pkl'), final_std_inf)
    
    save_data(os.path.join(results_path, f'kf_drkf_inf_mse_mean_{dist}_{noise_dist}.pkl'), final_mean_drkf_inf)
    save_data(os.path.join(results_path, f'kf_drkf_inf_mse_std_{dist}_{noise_dist}.pkl'), final_std_drkf_inf)
    
    save_data(os.path.join(results_path, f'bcot_mse_mean_{dist}_{noise_dist}.pkl'), final_mean_bcot)
    save_data(os.path.join(results_path, f'bcot_mse_std_{dist}_{noise_dist}.pkl'), final_std_bcot)
    
    save_data(os.path.join(results_path, f'kl_mse_mean_{dist}_{noise_dist}.pkl'), final_mean_kl)
    save_data(os.path.join(results_path, f'kl_mse_std_{dist}_{noise_dist}.pkl'), final_std_kl)
    
    save_data(os.path.join(results_path, f'risk_sensitive_mse_mean_{dist}_{noise_dist}.pkl'), final_mean_risk)
    save_data(os.path.join(results_path, f'risk_sensitive_mse_std_{dist}_{noise_dist}.pkl'), final_std_risk)
    
    print("\nKF state estimation experiments completed.")
    print("Finite-horizon KF MSE Mean:\n", final_mean_finite)
    print("Finite-horizon KF MSE Std:\n", final_std_finite)
    print("Steady-state KF MSE Mean:\n", final_mean_inf)
    print("Steady-state KF MSE Std:\n", final_std_inf)
    print("Steady-state DRKF MSE Mean:\n", final_mean_drkf_inf)
    print("Steady-state DRKF MSE Std:\n", final_std_drkf_inf)
    print("BCOT MSE Mean:\n", final_mean_bcot)
    print("BCOT MSE Std:\n", final_std_bcot)
    print("KL robust filter MSE Mean:\n", final_mean_kl)
    print("KL robust filter MSE Std:\n", final_std_kl)
    print("Risk-sensitive filter MSE Mean:\n", final_mean_risk)
    print("Risk-sensitive filter MSE Std:\n", final_std_risk)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', default="normal", type=str,
                        help="Disturbance distribution (normal or quadratic)")
    parser.add_argument('--noise_dist', default="normal", type=str,
                        help="Measurement noise distribution (normal or quadratic)")
    parser.add_argument('--num_sim', default=20, type=int,
                        help="Number of simulation runs per experiment")
    parser.add_argument('--horizon', default=20, type=int,
                        help="Time horizon T")
    parser.add_argument('--num_exp', default=20, type=int,
                        help="Number of independent experiments")
    args = parser.parse_args()
    main(args.dist, args.noise_dist, args.num_sim, args.horizon, args.num_exp)
