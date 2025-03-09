#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRKF_ours.py implements a distributionally robust Kalman filter (DRKF) for state estimation.
It is based on our DRCE formulation, and before running the filter it solves (via SDP)
for a worst–case measurement noise covariance.

"""

import numpy as np
import time
import cvxpy as cp  # For SDP (to be implemented)

class DRKF_ours:
    def __init__(self, T, dist, noise_dist, system_data,
                 true_x0_mean, true_x0_cov,
                 true_mu_w, true_Sigma_w,
                 true_mu_v, true_Sigma_v,
                 nominal_x0_mean, nominal_x0_cov,
                 nominal_mu_w, nominal_Sigma_w,
                 nominal_mu_v, nominal_Sigma_v,
                 x0_max=None, x0_min=None, w_max=None, w_min=None, v_max=None, v_min=None,
                 theta_x=None, theta_v=None):
        """
        Parameters:
          T           : Horizon length.
          dist, noise_dist: Distribution types ('normal' or 'quadratic').
          system_data : Tuple (A, B, C).
          
          The following parameters are provided in two sets:
             (i) True parameters (used to simulate the system).
             (ii) Nominal parameters (obtained via EM, used in filtering).
             
        """
        self.T = T
        self.dist = dist
        self.noise_dist = noise_dist
        self.A, self.B, self.C = system_data
        self.nx = self.B.shape[0]
        self.nu = self.B.shape[1]
        self.ny = self.C.shape[0]
        
        # True parameters (for simulation)
        self.true_x0_mean = true_x0_mean
        self.true_x0_cov = true_x0_cov
        self.true_mu_w = true_mu_w
        self.true_Sigma_w = true_Sigma_w
        self.true_mu_v = true_mu_v
        self.true_Sigma_v = true_Sigma_v
        
        # Nominal parameters (for filtering)
        self.nominal_x0_mean = nominal_x0_mean
        self.nominal_x0_cov = nominal_x0_cov
        self.nominal_mu_w = nominal_mu_w
        self.nominal_Sigma_w = nominal_Sigma_w
        self.nominal_mu_v = nominal_mu_v
        self.nominal_Sigma_v = nominal_Sigma_v
        
        # Bounds for non-normal distributions
        self.x0_max = x0_max
        self.x0_min = x0_min
        self.w_max = w_max
        self.w_min = w_min
        self.v_max = v_max
        self.v_min = v_min
        
        # Additional DRKF parameters
        self.theta_x = theta_x
        self.theta_v = theta_v
        
        # Initialize the initial state sample (for simulation) using the true distribution.
        if self.dist == "normal":
            self.x0_init = self.normal(self.true_x0_mean, self.true_x0_cov)
        elif self.dist == "quadratic":
            self.x0_init = self.quadratic(self.x0_max, self.x0_min)
        else:
            raise ValueError("Unsupported distribution for initial state.")
        
        # Similarly, generate an initial measurement sample.
        if self.noise_dist == "normal":
            self.true_v_init = self.normal(self.true_mu_v, self.true_Sigma_v)
        elif self.noise_dist == "quadratic":
            self.true_v_init = self.quadratic(self.v_max, self.v_min)
        else:
            raise ValueError("Unsupported measurement noise distribution.")
        
        # Before filtering, compute the worst-case measurement noise covariance via SDP.
        worst_case_Sigma_v, worst_case_Xprior, status = self.solve_sdp()
        
        self.wc_Sigma_v = worst_case_Sigma_v
        self.wc_Xprior = worst_case_Xprior
    
    # --- Distribution Sampling Functions ---
    def normal(self, mu, Sigma, N=1):
        # mu expected shape: (n, 1)
        return np.random.multivariate_normal(mu[:, 0], Sigma, size=N).T

    def uniform(self, a, b, N=1):
        n = a.shape[0]
        return a + (b - a) * np.random.rand(n, N)

    def quad_inverse(self, x, b, a):
        row, col = x.shape
        for i in range(row):
            for j in range(col):
                beta = (a[j] + b[j]) / 2.0
                alpha = 12.0 / ((b[j] - a[j])**3)
                tmp = 3 * x[i, j] / alpha - (beta - a[j])**3
                if tmp >= 0:
                    x[i, j] = beta + tmp**(1.0/3.0)
                else:
                    x[i, j] = beta - (-tmp)**(1.0/3.0)
        return x

    def quadratic(self, max_val, min_val, N=1):
        n = min_val.shape[0]
        x = np.random.rand(n, N)
        x = self.quad_inverse(x, max_val, min_val)
        return x

    def sample_initial_state(self):
        if self.dist == "normal":
            return self.normal(self.true_x0_mean, self.true_x0_cov)
        elif self.dist == "quadratic":
            return self.quadratic(self.x0_max, self.x0_min)
        else:
            raise ValueError("Unsupported distribution for initial state.")

    def sample_process_noise(self):
        if self.dist == "normal":
            return self.normal(self.true_mu_w, self.true_Sigma_w)
        elif self.dist == "quadratic":
            return self.quadratic(self.w_max, self.w_min)
        else:
            raise ValueError("Unsupported distribution for process noise.")

    def sample_measurement_noise(self):
        if self.noise_dist == "normal":
            return self.normal(self.true_mu_v, self.true_Sigma_v)
        elif self.noise_dist == "quadratic":
            return self.quadratic(self.v_max, self.v_min)
        else:
            raise ValueError("Unsupported distribution for measurement noise.")
        
    def create_DR_sdp(self):
        #construct problem
        #Variables
        X = cp.Variable((self.nx, self.nx), symmetric=True, name='X')
        X_pred = cp.Variable((self.nx, self.nx), symmetric=True, name='X_pred')
        Sigma_v = cp.Variable((self.ny, self.ny), symmetric=True, name='Sigma_v')
        X_pred_hat = cp.Variable((self.nx, self.nx), symmetric=True, name='X_pred_hat')
        Y = cp.Variable((self.nx, self.nx), name='Y', PSD=True)
        Z = cp.Variable((self.ny, self.ny), name='Z', PSD=True)
        
        
        #Parameters
        theta_x = cp.Parameter(nonneg=True, name='theta_x')
        Sigma_v_hat = cp.Parameter((self.ny, self.ny), name='Sigma_v_hat') # nominal measurement noise covariance
        theta_v = cp.Parameter(nonneg=True, name='theta_v')
        Sigma_w_hat = cp.Parameter((self.nx, self.nx), name='Sigma_w_hat') # nominal disturbance covariance
        
        #use Schur Complements
        #obj function
        obj = cp.Maximize(cp.trace(X)) 
        
        #constraints
        constraints = [
                cp.bmat([[X_pred-X, X_pred @ self.C.T],
                        [self.C @ X_pred, self.C @ X_pred @ self.C.T + Sigma_v]
                        ]) >> 0 ,
                cp.trace(X_pred_hat + X_pred - 2*Y ) <= theta_x**2,
                cp.bmat([[X_pred_hat, Y],
                         [Y.T, X_pred]
                         ]) >> 0,
                cp.trace(Sigma_v_hat + Sigma_v - 2*Z ) <= theta_v**2,
                cp.bmat([[Sigma_v_hat, Z],
                         [Z.T, Sigma_v]
                         ]) >> 0,
                X_pred_hat == self.A @ X @ self.A.T + Sigma_w_hat,                
                X>>0,
                X_pred>>0,
                X_pred_hat>>0,
                Sigma_v>>0
                ]
        
        prob = cp.Problem(obj, constraints)
        return prob
    
    # --- SDP Placeholder ---
    def solve_sdp(self):
        """
        This function should solve an SDP to compute the worst-case measurement noise
        covariance.
        """
        prob = self.create_DR_sdp()
        
        params = prob.parameters()
        # for i, param in enumerate(params):
        #     print(f"params[{i}]: {param.name()}")
        # print("self.theta_x : ", self.theta_x)
        # print("self.theta_v : ", self.theta_v)
        params[0].value = self.theta_x
        params[1].value = self.nominal_Sigma_v
        params[2].value = self.theta_v
        params[3].value = self.nominal_Sigma_w
        
        prob.solve(solver=cp.MOSEK)
        
        if prob.status in ["infeasible", "unbounded"]:
            print(prob.status, 'False in inf DRKF !!!!!!!!')
            
        sol = prob.variables()
        # for i, var in enumerate(sol):
        #     print(f"var[{i}]: {var.name()}")
        # [var[0]: X
        # var[1]: X_pred
        # var[2]: Sigma_v
        # var[3]: X_pred_hat
        # var[4]: Y
        # var[5]: Z]
        
        worst_case_Sigma_v = sol[2].value 
        worst_case_Xprior = sol[1].value
        return worst_case_Sigma_v, worst_case_Xprior, prob.status

    # --- DR-KF Update Step ---
    def DR_kalman_filter(self, v_mean_hat, x, y, mu_w=None, u=None):
        """
        Performs one update step of the DRKF.
        
        Parameters:
          v_mean_hat : Nominal measurement noise mean.
          Sigma_v_hat      : Nominal measurement noise covariance.
          x          : Current (or predicted) state estimate.
          y          : New measurement.
          mu_w       : Nominal process noise mean (optional).
          u          : Control input (if any; optional).
        
        Returns:
          Updated state estimate.
          
        Note: This update uses the worst-case measurement noise covariance computed via SDP.
        """
        if u is None:
            # No control input.
            x_pred = x
        else:
            x_pred = self.A @ x + self.B @ u + mu_w
        # Predicted measurement based on x_pred:
        y_pred = self.C @ x_pred + v_mean_hat
        # Innovation:
        innovation = y - y_pred
        # Here we use the worst-case Prior & measurement noise covariance.
        S = self.C @ self.wc_Xprior @ self.C.T + self.wc_Sigma_v
        # Compute Kalman gain
        K = self.wc_Xprior @ self.C.T @ np.linalg.inv(S)
        # Update state estimate:
        x_new = x_pred + K @ innovation
        return x_new

    # --- Forward Simulation ---
    def forward(self):
        """
        Runs the DRKF forward in time.
        Returns a dictionary containing:
          - comp_time     : Computation time.
          - state_traj    : True state trajectory.
          - output_traj   : Measurement trajectory.
          - est_state_traj: DRKF state estimates.
          - mse           : Mean squared error (state estimation error squared) at each time step.
        """
        start_time = time.time()
        T = self.T
        nx = self.A.shape[0]
        ny = self.C.shape[0]
        A = self.A
        C = self.C
        
        # Allocate arrays.
        x = np.zeros((T+1, nx, 1))         # True state trajectory.
        y = np.zeros((T+1, ny, 1))           # Measurement trajectory.
        x_est = np.zeros((T+1, nx, 1))       # State estimates.
        
        # Initialization.
        x[0] = self.sample_initial_state()
        # Use nominal initial state as the filter's initial estimate.
        x_est[0] = self.nominal_x0_mean.copy()
        # Generate first measurement.
        v0 = self.sample_measurement_noise()
        y[0] = C @ x[0] + v0
        # Initial update.
        x_est[0] = self.DR_kalman_filter(self.nominal_mu_v, self.nominal_x0_mean, y[0])
        
        mse = np.zeros(T+1)
        mse[0] = np.linalg.norm(x_est[0] - x[0])**2
        
        # Time update and filtering loop.
        for t in range(T):
            # Propagate true state.
            w = self.sample_process_noise()
            x[t+1] = A @ x[t] + self.B @ w
            # Generate measurement.
            v = self.sample_measurement_noise()
            y[t+1] = C @ x[t+1] + v
            # Prediction step: use nominal process noise mean.
            x_pred = A @ x_est[t] + self.nominal_mu_w
            # Update using DR-KF.
            x_est[t+1] = self.DR_kalman_filter(self.nominal_mu_v, x_pred, y[t+1])
            # Compute squared error.
            mse[t+1] = np.linalg.norm(x_est[t+1] - x[t+1])**2
        
        comp_time = time.time() - start_time
        return {'comp_time': comp_time,
                'state_traj': x,
                'output_traj': y,
                'est_state_traj': x_est,
                'mse': mse}
