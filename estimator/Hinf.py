#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hinf.py implements a H∞ filter for state estimation.
This implementation is an alternative to using the FilterPy HInfinityFilter.
It uses a modified update equation:
 1. Prediction:
    x_pred = A x + nominal_mu_w
    P_pred = A P A^T + Q
 2. Update:
    Compute innovation:  y_tilde = z - C x_pred - nominal_mu_v
    Compute innovation covariance: S = C P_pred C^T + R - (1/γ^2) I
    Gain: K = P_pred C^T S^{-1}
    x_new = x_pred + K y_tilde
    P_new = P_pred - K C P_pred
Here, Q = nominal_Sigma_w and R = nominal_M.
The filter uses the nominal offsets (nominal_mu_w and nominal_mu_v)
obtained via EM.
"""

import numpy as np
import time
import scipy.linalg as linalg

class Hinf:
    def __init__(self, T, dist, noise_dist, system_data,
                 true_x0_mean, true_x0_cov,
                 true_mu_w, true_Sigma_w,
                 true_mu_v, true_M,
                 nominal_x0_mean, nominal_x0_cov,
                 nominal_mu_w, nominal_Sigma_w,
                 nominal_mu_v, nominal_M,
                 x0_max=None, x0_min=None, w_max=None, w_min=None, v_max=None, v_min=None,
                 gamma=10.0):
        """
        Parameters:
          T            : Horizon length.
          dist, noise_dist : Distribution types ('normal' or 'quadratic').
          system_data  : Tuple (A, C).
          true_*       : True parameters (for simulation).
          nominal_*    : Nominal parameters (obtained via EM, used in filtering),
                         including nonzero nominal offsets.
          gamma        : H∞ performance level.
        """
        self.T = T
        self.dist = dist
        self.noise_dist = noise_dist
        self.A, self.C = system_data
        self.nx = self.A.shape[0]
        self.ny = self.C.shape[0]
        
        # True parameters (used in simulation)
        self.true_x0_mean = true_x0_mean
        self.true_x0_cov = true_x0_cov
        self.true_mu_w = true_mu_w
        self.true_Sigma_w = true_Sigma_w
        self.true_mu_v = true_mu_v
        self.true_M = true_M
        
        # Nominal parameters (for filtering)
        self.nominal_x0_mean = nominal_x0_mean
        self.nominal_x0_cov = nominal_x0_cov
        self.nominal_mu_w = nominal_mu_w  # process noise offset
        self.nominal_Sigma_w = nominal_Sigma_w  # Q
        self.nominal_mu_v = nominal_mu_v  # measurement noise offset
        self.nominal_M = nominal_M       # R
        
        # Optional bounds for non-normal distributions
        self.x0_max = x0_max
        self.x0_min = x0_min
        self.w_max = w_max
        self.w_min = w_min
        self.v_max = v_max
        self.v_min = v_min
        
        self.gamma = gamma
        
        # Initialize the filter state and covariance using nominal parameters.
        self.x = self.nominal_x0_mean.copy()
        self.P = self.nominal_x0_cov.copy()
    
    # --- Distribution Sampling Functions ---
    def normal(self, mu, Sigma, N=1):
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
                    x[i, j] = beta + tmp ** (1.0/3.0)
                else:
                    x[i, j] = beta - (-tmp) ** (1.0/3.0)
        return x
    
    def quadratic(self, wmax, wmin, N=1):
        n = wmin.shape[0]
        x = np.random.rand(N, n)
        x = self.quad_inverse(x, wmax, wmin)
        return x.T
    
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
            return self.normal(self.true_mu_v, self.true_M)
        elif self.noise_dist == "quadratic":
            return self.quadratic(self.v_max, self.v_min)
        else:
            raise ValueError("Unsupported distribution for measurement noise.")
    
    # --- H∞ Filter Steps ---
    def predict(self):
        # Predict the state with the nominal process noise offset (no B matrix):
        self.x = self.A @ self.x + self.nominal_mu_w
        # Process noise covariance is added directly:
        self.P = self.A @ self.P @ self.A.T + self.nominal_Sigma_w
    
    def update(self, z):
        # Compute innovation with nominal measurement offset.
        y_tilde = z - self.C @ self.x - self.nominal_mu_v
        # Innovation covariance modified for H∞:
        S = self.C @ self.P @ self.C.T + self.nominal_M - (1.0/(self.gamma**2)) * np.eye(self.ny)
        # Check S for positive definiteness:
        if np.any(np.linalg.eigvals(S) <= 0):
            raise ValueError("Innovation covariance S is not positive definite. Adjust gamma.")
        # Compute gain:
        K = self.P @ self.C.T @ np.linalg.inv(S)
        # Update state estimate and covariance:
        self.x = self.x + K @ y_tilde
        self.P = self.P - K @ self.C @ self.P
        # Force symmetry:
        self.P = (self.P + self.P.T) / 2
    
    # --- Forward Simulation ---
    def forward(self):
        start_time = time.time()
        T = self.T
        nx = self.nx
        ny = self.ny
        A = self.A
        C = self.C
        
        # Allocate arrays for true state, measurements, and state estimates.
        x_true_all = np.zeros((T+1, nx, 1))
        y_all = np.zeros((T+1, ny, 1))
        x_est_all = np.zeros((T+1, nx, 1))
        
        # Initialization.
        x_true_all[0] = self.sample_initial_state()
        y_all[0] = C @ x_true_all[0] + self.sample_measurement_noise()
        self.x = self.nominal_x0_mean.copy()  # reset filter state
        self.P = self.nominal_x0_cov.copy()
        x_est_all[0] = self.x.copy()
        
        for t in range(T):
            # Propagate true state (without using a B matrix).
            w = self.sample_process_noise()
            x_true_all[t+1] = A @ x_true_all[t] + w
            v = self.sample_measurement_noise()
            y_all[t+1] = C @ x_true_all[t+1] + v
            
            # H∞ filter update:
            self.predict()
            self.update(y_all[t+1])
            x_est_all[t+1] = self.x.copy()
        
        comp_time = time.time() - start_time
        mse = np.zeros(T+1)
        for t in range(T+1):
            mse[t] = np.linalg.norm(x_est_all[t] - x_true_all[t])**2
        
        return {'comp_time': comp_time,
                'state_traj': x_true_all,
                'output_traj': y_all,
                'est_state_traj': x_est_all,
                'mse': mse}
