#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KL.py implements a steady–state (infinite–horizon) robust filter based on a
KL–divergence constraint for state estimation. In this LQR closed–loop version,
the system evolves as:
    xₜ₊₁ = A xₜ + B uₜ + wₜ,    yₜ = C xₜ + vₜ,
with control input computed via an LQR controller:
    u[t] = -K_lqr x_est[t].
The robust update is performed at every time step using the optimize function
(from utils.py with algo='KL').
"""

import numpy as np
import time
from estimator.utils import optimize

class KL:
    def __init__(self, T, dist, noise_dist, system_data, B,
                 true_x0_mean, true_x0_cov,
                 true_mu_w, true_Sigma_w,
                 true_mu_v, true_M,
                 nominal_x0_mean, nominal_x0_cov,
                 nominal_mu_w, nominal_Sigma_w,
                 nominal_mu_v, nominal_M,
                 radius, maxit=20,
                 x0_max=None, x0_min=None, w_max=None, w_min=None, v_max=None, v_min=None):
        """
        Parameters:
          T: time horizon.
          dist, noise_dist: 'normal' or 'quadratic'
          system_data: tuple (A, C)
          B: control input matrix.
          
          The following are provided in two sets:
            (i) True parameters (for simulating the system)
           (ii) Nominal parameters (obtained via EM, used in filtering)
           
          Additional parameters:
            radius: robustness parameter (KL constraint radius)
            maxit: maximum iterations for the optimization routine.
          
          x0_max, x0_min, etc.: bounds for non–normal distributions.
        """
        self.T = T
        self.dist = dist
        self.noise_dist = noise_dist
        self.A, self.C = system_data
        self.B = B
        
        # True parameters.
        self.true_x0_mean = true_x0_mean
        self.true_x0_cov = true_x0_cov
        self.true_mu_w = true_mu_w
        self.true_Sigma_w = true_Sigma_w
        self.true_mu_v = true_mu_v
        self.true_M = true_M
        
        # Nominal parameters.
        self.nominal_x0_mean = nominal_x0_mean
        self.nominal_x0_cov = nominal_x0_cov
        self.nominal_mu_w = nominal_mu_w
        self.nominal_Sigma_w = nominal_Sigma_w
        self.nominal_mu_v = nominal_mu_v
        self.nominal_M = nominal_M
        
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
        
        # KL-specific parameters.
        self.radius = radius
        self.maxit = maxit
        # Use nominal covariances as references.
        self.Bp = self.nominal_Sigma_w  # reference process noise covariance
        self.Dp = self.nominal_M         # reference measurement noise covariance
        
        # LQR gain will be assigned externally.
        self.K_lqr = None

    # --- Sampling Functions for True Noise ---
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
        x = np.random.rand(n, N)
        x = self.quad_inverse(x, max_val, min_val)
        return x

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

    # --- Forward Simulation Using the Robust KL Update with LQR ---
    def forward(self):
        start_time = time.time()
        T = self.T
        nx = self.nx
        ny = self.ny
        A = self.A
        C = self.C
        B = self.B
        
        # Allocate arrays for true state, measurements, and state estimates.
        x = np.zeros((T+1, nx, 1))
        y = np.zeros((T+1, ny, 1))
        x_est = np.zeros((T+1, nx, 1))
        
        # --- Initialization ---
        x[0] = self.sample_initial_state()
        # Set the filter initial estimate to the known initial state mean.
        x_est[0] = self.nominal_x0_mean.copy()
        pre_cov = self.nominal_x0_cov.copy()
        
        # First measurement.
        v0 = self.sample_measurement_noise()
        y[0] = C @ x[0] + v0
        
        # Initial robust update using the optimization routine.
        try:
            update_mean, update_cov = optimize(ny, nx, self.radius, A, self.Bp, C, self.Dp,
                                               pre_cov, y[0], self.nominal_x0_mean, self.maxit, algo='KL')
        except Exception as e:
            print("KL optimization failed at t=0, using fallback update. Error:", e)
            K_fallback = A @ pre_cov @ A.T + self.Bp
            update_mean = A @ self.nominal_x0_mean
            update_cov = K_fallback
        pre_mean = update_mean.copy()
        pre_cov = update_cov.copy()
        x_est[0] = update_mean.copy()
        
        mse = np.zeros(T+1)
        mse[0] = np.linalg.norm(x_est[0] - x[0])**2
        
        # --- Time Update and Robust Filtering with Control ---
        for t in range(T):
            # Compute control input: u[t] = -K_lqr * x_est[t]
            if self.K_lqr is None:
                raise ValueError("LQR gain (K_lqr) has not been assigned!")
            u = -self.K_lqr @ x_est[t]
            
            # True state propagation: x[t+1] = A*x[t] + B*u + w
            w = self.sample_process_noise()
            x[t+1] = A @ x[t] + B @ u + w
            
            # Generate measurement.
            v = self.sample_measurement_noise()
            y[t+1] = C @ x[t+1] + v
            
            # Prediction step: include control input and add nominal process noise mean.
            x_pred = A @ x_est[t] + B @ u + self.nominal_mu_w
            try:
                update_mean, update_cov = optimize(ny, nx, self.radius, A, self.Bp, C, self.Dp,
                                                   pre_cov, y[t+1], x_pred, self.maxit, algo='KL')
            except Exception as e:
                print(f"KL optimization failed at t={t+1}, using fallback update. Error:", e)
                K_fallback = A @ pre_cov @ A.T + self.Bp
                update_mean = A @ pre_mean
                update_cov = K_fallback
            x_est[t+1] = update_mean.copy()
            pre_mean = update_mean.copy()
            pre_cov = update_cov.copy()
            
            mse[t+1] = np.linalg.norm(x_est[t+1] - x[t+1])**2
        
        comp_time = time.time() - start_time
        return {'comp_time': comp_time,
                'state_traj': x,
                'output_traj': y,
                'est_state_traj': x_est,
                'mse': mse}
