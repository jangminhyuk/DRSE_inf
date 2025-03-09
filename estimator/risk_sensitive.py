#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
risk_sensitive.py

This module implements a steady–state (infinite–horizon) risk–sensitive filter for
state estimation. It is based on the risk–sensitive optimal filtering approach described in

    A. R. Chowdhury and T. Basar, "A risk–sensitive approach to optimal filtering",
    IEEE Transactions on Automatic Control, Vol. 36, No. 4, 1991, pp. 496–501.

In this implementation, the standard Kalman filter update is modified by augmenting the
innovation covariance with a term proportional to the risk sensitivity parameter (theta_rs).
Critically, the filter uses the known (true) mean vectors for the initial state and noise,
while the covariances are taken from the EM estimates.
"""

import numpy as np
import time

class RiskSensitive:
    def __init__(self, T, dist, noise_dist, system_data,
                 true_x0_mean, true_x0_cov,
                 true_mu_w, true_Sigma_w,
                 true_mu_v, true_M,
                 nominal_x0_mean, nominal_x0_cov,
                 nominal_mu_w, nominal_Sigma_w,
                 nominal_mu_v, nominal_M,
                 theta_rs,   # risk sensitivity parameter
                 x0_max=None, x0_min=None, w_max=None, w_min=None, v_max=None, v_min=None):
        """
        Parameters:
          T: Time horizon.
          dist, noise_dist: 'normal' or 'quadratic'
          system_data: Tuple (A, C)
          true_x0_mean, true_x0_cov, true_mu_w, true_Sigma_w, true_mu_v, true_M:
              True parameters for simulation.
          nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w, nominal_mu_v, nominal_M:
              Nominal parameters used for filtering. In this revised implementation,
              the known (true) mean vectors are used while the covariances come from EM.
          theta_rs: Risk sensitivity parameter.
          x0_max, x0_min, etc.: Bounds for non–normal distributions.
        """
        self.T = T
        self.dist = dist
        self.noise_dist = noise_dist
        self.A, self.C = system_data

        # True parameters.
        self.true_x0_mean = true_x0_mean
        self.true_x0_cov = true_x0_cov
        self.true_mu_w = true_mu_w
        self.true_Sigma_w = true_Sigma_w
        self.true_mu_v = true_mu_v
        self.true_M = true_M

        # Nominal parameters for filtering.
        # In this revision, we use the known mean vectors:
        self.nominal_x0_mean = nominal_x0_mean    # known initial state mean
        self.nominal_mu_w = nominal_mu_w          # known process noise mean
        self.nominal_mu_v = nominal_mu_v          # known measurement noise mean
        # Covariances are taken from the EM estimates.
        self.nominal_x0_cov = nominal_x0_cov
        self.nominal_Sigma_w = nominal_Sigma_w
        self.nominal_M = nominal_M

        # Risk sensitivity parameter.
        self.theta_rs = theta_rs

        # Bounds for non–normal distributions.
        if self.dist in ["uniform", "quadratic"]:
            self.x0_max = x0_max
            self.x0_min = x0_min
            self.w_max = w_max
            self.w_min = w_min
        if self.noise_dist in ["uniform", "quadratic"]:
            self.v_max = v_max
            self.v_min = v_min

        self.nx = self.A.shape[0]
        self.ny = self.C.shape[0]

    # --- Sampling Functions (same as in other estimators) ---
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
                alpha = 12.0 / ((b[j] - a[j]) ** 3)
                tmp = 3 * x[i, j] / alpha - (beta - a[j]) ** 3
                if tmp >= 0:
                    x[i, j] = beta + tmp ** (1.0/3.0)
                else:
                    x[i, j] = beta - (-tmp) ** (1.0/3.0)
        return x

    def quadratic(self, max_val, min_val, N=1):
        n = min_val.shape[0]
        x = np.random.rand(N, n)
        x = self.quad_inverse(x, max_val, min_val)
        return x.T

    def sample_initial_state(self):
        if self.dist == "normal":
            return self.normal(self.true_x0_mean, self.true_x0_cov, N=1)
        elif self.dist == "quadratic":
            return self.quadratic(self.x0_max, self.x0_min, N=1)
        else:
            raise ValueError("Unsupported distribution for initial state.")

    def sample_process_noise(self):
        if self.dist == "normal":
            return self.normal(self.true_mu_w, self.true_Sigma_w, N=1)
        elif self.dist == "quadratic":
            return self.quadratic(self.w_max, self.w_min, N=1)
        else:
            raise ValueError("Unsupported distribution for process noise.")

    def sample_measurement_noise(self):
        if self.noise_dist == "normal":
            return self.normal(self.true_mu_v, self.true_M, N=1)
        elif self.noise_dist == "quadratic":
            return self.quadratic(self.v_max, self.v_min, N=1)
        else:
            raise ValueError("Unsupported distribution for measurement noise.")

    # --- Forward Simulation Using the Risk-Sensitive Update ---
    def forward(self):
        start_time = time.time()
        T = self.T
        nx = self.nx
        ny = self.ny
        A = self.A
        C = self.C
        # Use EM–estimated covariances for the noise
        Q = self.nominal_Sigma_w      # process noise covariance
        R = self.nominal_M            # measurement noise covariance
        theta_rs = self.theta_rs

        # Allocate arrays.
        x = np.zeros((T+1, nx, 1))
        y = np.zeros((T+1, ny, 1))
        x_est = np.zeros((T+1, nx, 1))
        mse = np.zeros(T+1)

        # --- Initialization ---
        x[0] = self.sample_initial_state()        
        # Use the known initial state mean.
        x_est[0] = self.nominal_x0_mean.copy()
        # Initialize the error covariance with the EM–estimated initial covariance.
        Sigma = self.nominal_x0_cov.copy()

        # First measurement at t=0.
        v0 = self.sample_measurement_noise()
        y[0] = C @ x[0] + v0

        mse[0] = np.linalg.norm(x_est[0] - x[0])**2

        # Iterate from t = 0 to T-1.
        for t in range(0, T):
            # Propagate true state.
            w = self.sample_process_noise()
            x[t+1] = A @ x[t] + w
            # Generate measurement at t+1.
            v = self.sample_measurement_noise()
            y[t+1] = C @ x[t+1] + v

            # Prediction step using known noise means.
            x_pred = A @ x_est[t] + self.nominal_mu_w
            y_pred = C @ x_pred + self.nominal_mu_v

            # Risk-sensitive innovation covariance.
            S = C @ Sigma @ C.T + R + theta_rs * (C @ Sigma @ C.T)
            # Kalman gain.
            K = A @ Sigma @ C.T @ np.linalg.inv(S)
            # Update estimate.
            innovation = y[t+1] - y_pred
            x_est[t+1] = x_pred + K @ innovation
            # Update covariance.
            Sigma = A @ Sigma @ A.T + Q - A @ Sigma @ C.T @ np.linalg.inv(S) @ C @ Sigma @ A.T

            mse[t+1] = np.linalg.norm(x_est[t+1] - x[t+1])**2

        comp_time = time.time() - start_time
        return {'comp_time': comp_time,
                'state_traj': x,
                'output_traj': y,
                'est_state_traj': x_est,
                'mse': mse}
