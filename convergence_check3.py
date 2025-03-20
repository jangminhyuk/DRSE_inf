import numpy as np
import cvxpy as cp
import mosek
import control

np.random.seed(42)

# -------------------------------------------------------
# Sampling and distribution functions
# -------------------------------------------------------
def uniform(a, b, N=1):
    n = a.shape[0]
    return a[:, None] + (b[:, None] - a[:, None]) * np.random.rand(n, N)

def normal(mu, Sigma, N=1):
    return np.random.multivariate_normal(mu.ravel(), Sigma, size=N).T

def quad_inverse(x, b, a):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            beta = 0.5 * (a[j] + b[j])
            alpha = 12.0 / ((b[j] - a[j]) ** 3)
            tmp = 3 * x[i, j] / alpha - (beta - a[j]) ** 3
            x[i, j] = beta + (tmp if tmp >= 0 else -(-tmp) ** (1. / 3.)) ** (1. / 3.)
    return x

def quadratic(wmax, wmin, N=1):
    x = np.random.rand(N, wmin.shape[0]).T
    return quad_inverse(x, wmax, wmin)

def gen_sample_dist_inf(dist, N_sample, mu=None, Sigma=None, w_min=None, w_max=None):
    if dist == "normal":
        w = normal(mu, Sigma, N=N_sample)
    elif dist == "uniform":
        w = uniform(w_min, w_max, N=N_sample)
    elif dist == "quadratic":
        w = quadratic(wmax=w_max, wmin=w_min, N=N_sample)
    else:
        raise ValueError("Unsupported distribution.")
    mean_ = np.mean(w, axis=1, keepdims=True)
    var_ = np.cov(w)
    return mean_, var_

# -------------------------------------------------------
# DR Kalman filter measurement update (only theta_x version)
# -------------------------------------------------------
def dr_kf_solve_measurement_update(Sigma_x_minus_hat, C, Sigma_v_nom, theta_x, delta=1e-6):
    n = Sigma_x_minus_hat.shape[0]
    m = Sigma_v_nom.shape[0]
    # We use an SDP formulation to maximize the trace of the posterior state covariance.
    Z_var = cp.Variable((m, m), PSD=True)
    Sigma_x_var = cp.Variable((n, n), PSD=True)
    Sigma_x_minus_var = cp.Variable((n, n), PSD=True)
    Y = cp.Variable((n, n))
    
    obj = cp.Maximize(cp.trace(Sigma_x_var))
    
    constraints = [
        cp.bmat([
            [Sigma_x_minus_var - Sigma_x_var, Sigma_x_minus_var @ C.T],
            [C @ Sigma_x_minus_var, C @ Sigma_x_minus_var @ C.T + Sigma_v_nom]
        ]) >> 0,
        cp.trace(Sigma_x_minus_hat + Sigma_x_minus_var - 2*Y) <= theta_x**2,
        cp.bmat([[Sigma_x_minus_hat, Y],
                 [Y.T, Sigma_x_minus_var]
                ]) >> 0,
    ]
    
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)
    
    return Sigma_x_var.value

# -------------------------------------------------------
# Matrix computations for phi_T and related quantities
# -------------------------------------------------------
def compute_matrices(T, A, Sigma_w_nom, C, Sigma_v_nom):
    """
    For a given horizon T and system matrices, compute:
    
      1. R_T: [B, A·B, A²·B, …, A^(T–1)·B] where B = sqrt(Sigma_w_nom).
         (This is a horizontal concatenation; R_T has shape (n, T*n).)
      
      2. O_T: Vertical stacking of [C·A^(T–1); C·A^(T–2); …; C].
         (Shape: (T*m, n), where m = rows(C).)
      
      3. O_T^R: Vertical stacking of [A^(T–1); A^(T–2); …; I].
         (Shape: (T*n, n).)
      
      4. D_T: I_T ⊗ sqrt(Sigma_v_nom) (with sqrt via Cholesky).
         (Shape: (T*m, T*m).)
      
      5. L_T and H_T (the block Hankel matrices) 
      
      6. Compute tilde_phi_T:
             tilde_phi_T = 1 / lambda_max( L_T ( I_(T*n) + H_Tᵀ (D_TD_Tᵀ)⁻¹ H_T )⁻¹ L_Tᵀ ).
    
    Parameters:
      T             : integer, the time horizon.
      A             : state transition matrix (n×n)
      Sigma_w_nom   : nominal process noise covariance (n×n)
      C             : observation matrix (m×n)
      Sigma_v_nom   : nominal measurement noise covariance (m×m)
    
    Returns:
      A dictionary with keys:
         'R_T', 'O_T', 'O_T_R', 'D_T', 'L_T', 'H_T', and 'tilde_phi_T'.
    """
    import numpy as np

    n = A.shape[0]  # state dimension
    m = C.shape[0]  # measurement dimension

    # Compute square roots
    B = np.linalg.cholesky(Sigma_w_nom)      # B: (n x n)
    sqrt_Sigma_v_nom = np.linalg.cholesky(Sigma_v_nom)  # (m x m)

    # 1. R_T: [B, A·B, A²·B, …, A^(T-1)·B]
    R_T_blocks = []
    for i in range(T):
        A_power = np.linalg.matrix_power(A, i)
        R_T_blocks.append(A_power @ B)
    R_T = np.hstack(R_T_blocks)  # shape: (n, T*n)

    # 2. O_T: Vertical stacking of [C A^(T-1); C A^(T-2); ...; C]
    O_T_blocks = []
    for i in range(T):
        A_power = np.linalg.matrix_power(A, T-1-i)
        O_T_blocks.append(C @ A_power)
    O_T = np.vstack(O_T_blocks)  # shape: (T*m, n)

    # 3. O_T^R: Vertical stacking of [A^(T-1); A^(T-2); ...; I]
    O_T_R_blocks = []
    for i in range(T):
        A_power = np.linalg.matrix_power(A, T-1-i)
        O_T_R_blocks.append(A_power)
    O_T_R = np.vstack(O_T_R_blocks)  # shape: (T*n, n)

    # 4. D_T: I_T ⊗ sqrt(Sigma_v_nom)
    D_T = np.kron(np.eye(T), sqrt_Sigma_v_nom)  # shape: (T*m, T*m)

    # 5. Build block Hankel matrices L_T and H_T using the small–block definitions.
    # Initialize T x T block lists.
    L_blocks = [[None for _ in range(T)] for _ in range(T)]
    H_blocks = [[None for _ in range(T)] for _ in range(T)]
    for i in range(T):
        for j in range(T):
            if j - i >= 1:
                exponent = j - i - 1  # so that for j=i+1, exponent = 0.
                L_blocks[i][j] = np.linalg.matrix_power(A, exponent) @ B  # (n x n)
                H_blocks[i][j] = C @ (np.linalg.matrix_power(A, exponent) @ B)  # (m x n)
            else:
                L_blocks[i][j] = np.zeros((n, n))
                H_blocks[i][j] = np.zeros((m, n))
    # Form full block Hankel matrices.
    L_T = np.block(L_blocks)  # shape: (T*n, T*n)
    H_T = np.block(H_blocks)  # shape: (T*m, T*n)

    # 6. Compute tilde_phi_T:
    I_inner = np.eye(T * n)  # identity of size T*n
    DDT = D_T @ D_T.T       # shape: (T*m, T*m)
    inv_DDT = np.linalg.inv(DDT)
    inner_term = I_inner + H_T.T @ inv_DDT @ H_T  # shape: (T*n, T*n)
    inner_inv = np.linalg.inv(inner_term)
    M = L_T @ inner_inv @ L_T.T  # shape: (T*n, T*n)
    eigvals = np.linalg.eigvals(M)
    lambda_max_val = np.max(np.real(eigvals))
    tilde_phi_T = 1.0 / lambda_max_val

    return {
        "R_T": R_T,
        "O_T": O_T,
        "O_T_R": O_T_R,
        "D_T": D_T,
        "L_T": L_T,
        "H_T": H_T,
        "tilde_phi_T": tilde_phi_T
    }





def find_phi_T(O_T, O_T_R, L_T, H_T, D_T, tilde_phi_T, tol_eig=1e-6, bisection_tol=1e-8, max_iter=1000):
    """
    Search for phi in [0, tilde_phi_T] using bisection so that
      lambda_min(Omega_{phi I}) > tol_eig,
    where:
      M = D_T D_T^T + H_T H_T^T,    M_inv = inv(M),
      J_T = O_T_R - L_T @ H_T.T @ M_inv @ O_T,
      Omega_T = O_T.T @ M_inv @ O_T,
      I_N = identity of size L_T.shape[1],
      inv_DDT = inv(D_T D_T^T),
      inner_term = I_N + H_T.T @ inv_DDT @ H_T,   inner_inv = inv(inner_term),
      I_full = identity of size L_T.shape[0],
      S_{phi I} = - (1/phi)*I_full + L_T @ inner_inv @ L_T.T,
      Omega_{phi I} = Omega_T + J_T.T @ inv(S_{phi I}) @ J_T.
    The function returns the largest phi in [0, tilde_phi_T] such that
      lambda_min(Omega_{phi I}) > tol_eig.
    """
    import numpy as np
    
    # Precompute common matrices.
    M = D_T @ D_T.T + H_T @ H_T.T
    M_inv = np.linalg.inv(M)
    J_T = O_T_R - L_T @ H_T.T @ M_inv @ O_T
    Omega_T = O_T.T @ M_inv @ O_T
    I_N = np.eye(L_T.shape[1])
    inv_DDT = np.linalg.inv(D_T @ D_T.T)
    inner_term = I_N + H_T.T @ inv_DDT @ H_T
    inner_inv = np.linalg.inv(inner_term)
    I_full = np.eye(L_T.shape[0])
    
    def lambda_min_Omega(phi):
        S_phi = - (1.0 / phi) * I_full + L_T @ inner_inv @ L_T.T
        try:
            S_phi_inv = np.linalg.inv(S_phi)
        except np.linalg.LinAlgError:
            return -np.inf  # Treat noninvertibility as infeasible.
        Omega_phi = Omega_T + J_T.T @ S_phi_inv @ J_T
        eigvals = np.linalg.eigvals(Omega_phi)
        return np.min(np.real(eigvals))
    
    # Bisection search in [0, tilde_phi_T].
    phi_lower = 0.0
    phi_upper = tilde_phi_T
    iteration = 0
    while (phi_upper - phi_lower) > bisection_tol and iteration < max_iter:
        iteration += 1
        phi_mid = (phi_lower + phi_upper) / 2.0
        f_mid = lambda_min_Omega(phi_mid)
        print(f"Iteration {iteration}: phi = {phi_mid:.8f}, lambda_min = {f_mid:.8e}")
        if f_mid > tol_eig:
            # Candidate is feasible; try increasing phi.
            phi_lower = phi_mid
        else:
            # Candidate is not feasible; decrease phi.
            phi_upper = phi_mid
            
    phi_final = phi_lower
    print(f"Bisection converged after {iteration} iterations: phi_T = {phi_final:.8f}, lambda_min = {lambda_min_Omega(phi_final):.8e}")
    return phi_final



def compute_theta_max(A, C, Sigma_w_nom, Sigma_v_nom, phi_T, q=100, T=None):
    """
    Computes theta_max that guarantees convergence of the DR Kalman filter.
    
    1. Starting with P_bar0 = Sigma_w_nom, iterate the standard Kalman Riccati update:
         P_bar_{k+1} = A [P_bar_k - P_bar_k C^T (C P_bar_k C^T + Sigma_v_nom)^{-1} C P_bar_k] A^T + Sigma_w_nom
       for q iterations.
       
    2. Then compute:
         theta_max = sqrt(1 / (1/lambda_min(P_bar_q) - phi_T)) - sqrt(trace(P_bar_q))
    
    This function prints the eigenvalues of P_bar, and if T is provided it also prints q and T.
    """
    P_bar = Sigma_w_nom.copy()
    
    for q_ in range(q):
        CPCT = C @ P_bar @ C.T
        S = CPCT + Sigma_v_nom
        K = P_bar @ C.T @ np.linalg.inv(S)
        P_update = P_bar - K @ C @ P_bar
        P_bar = A @ P_update @ A.T + Sigma_w_nom

        # Compute eigenvalues and other quantities.
        eigvals = np.linalg.eigvals(P_bar)
        print("Eigenvalues of P_bar:", np.real(eigvals))
        lambda_min = np.min(np.real(eigvals))
        lambda_max = np.min(np.real(eigvals))
        trace_P = np.trace(P_bar)
        
        term = np.sqrt(trace_P / (1 - phi_T*lambda_max))#(1.0 / lambda_min) - phi_T
        
        if term <= 0:
            print("Warning: np.sqrt(trace_P / (1 - phi_T*lambda_max)) is not positive. Cannot compute theta_max.")
            return None
        
        theta_max = term - np.sqrt(trace_P)
        print(q_, theta_max)
        
    if theta_max > 0:
        print(f"theta_max: {theta_max}")
    else:
        print("Computed theta_max", theta_max, "is non-positive.")
        #exit()
        
    if T is not None:
        print(f"q = {q}, T = {T}")
        
    return theta_max

# -------------------------------------------------------
# System generation and DR Kalman filter simulation
# -------------------------------------------------------
def run_dr_kf_once(n=10, m=10, steps=200, N_samples=20, T=20, dist_type="normal", tol=1e-4):
    # Regenerate system matrices until detectability and controllability are met.
    while True:
        A = np.random.randn(n, n)
        Wr = np.random.randn(n, n)
        Sigma_w_true = Wr @ Wr.T + 1e-4 * np.eye(n)
        mu_w = np.zeros((n, 1))
        _, Sigma_w_nom = gen_sample_dist_inf("normal", N_samples, mu=mu_w, Sigma=Sigma_w_true)
        Sigma_w_nom += 1e-4 * np.eye(n)
        
        C = np.random.randn(m, n)
        
        Vr = np.random.randn(m, m)
        Sigma_v_true = Vr @ Vr.T + 1e-4 * np.eye(m)
        mu_v = np.zeros((m, 1))
        _, Sigma_v_nom = gen_sample_dist_inf("normal", N_samples, mu=mu_v, Sigma=Sigma_v_true)
        Sigma_v_nom += 1e-4 * np.eye(m)
        
        O = control.obsv(A, C)
        det_ok = (np.linalg.matrix_rank(O) == n)
        try:
            B = np.linalg.cholesky(Sigma_w_true)
        except np.linalg.LinAlgError:
            det_ok = False
        CC = control.ctrb(A, B)
        stab_ok = (np.linalg.matrix_rank(CC) == n)
        if det_ok and stab_ok:
            break

    ### !!! USE SYSTEM FROM the Risk-sensitive paper!        
    # A = np.array([[0.1, 1], [0, 1.2]])
    # C = np.array([[1, -1]])
    
    # Sigma_w_nom = np.eye(n)
    # Sigma_v_nom = np.eye(m)
    # Sigma_w_true = np.eye(n)
    # Sigma_v_true = np.eye(m)
    
    ##
    # nx = 4; nw = 4; ny = 2; dt = 0.5
    # A = np.array([[1, 0, dt, 0],
    #               [0, 1, 0, dt],
    #               [0, 0, 1, 0],
    #               [0, 0, 0, 1]])
    # C = np.array([[1, 0, 0, 0],
    #               [0, 1, 0, 0]])
    # system_data = (A, C)
    
    # # B matrix for control.
    # B = np.array([[0, 0],
    #               [0, 0],
    #               [1, 0],
    #               [0, 1]])
    # Sigma_w_nom = np.eye(nx)
    # Sigma_v_nom = np.eye(ny)
    # Sigma_w_true = np.eye(nx)
    # Sigma_v_true = np.eye(ny)
    
    # Compute phi_T and theta_max using a chosen horizon T.
    matrices = compute_matrices(T, A, Sigma_w_nom, C, Sigma_v_nom)
    tilde_phi_T = matrices["tilde_phi_T"]
    print(tilde_phi_T)
    phi_T = find_phi_T(matrices["O_T"], matrices["O_T_R"], matrices["L_T"], matrices["H_T"], matrices["D_T"], tilde_phi_T)
    theta_max = compute_theta_max(A, C, Sigma_w_nom, Sigma_v_nom, phi_T, q=35)
    
    print("--------------------------------------------------")
    print("Calculated phi_T:", phi_T)
    print("Calculated theta_max:", theta_max)
    print("--------------------------------------------------")
    
    Sigma_x_minus = np.eye(n)
    posterior_list = []
    conv_norms = []
    
    # Run the filter and early-stop if convergence is detected.
    for step in range(steps):
        Sigma_x_sol = dr_kf_solve_measurement_update(Sigma_x_minus, C, Sigma_v_nom, theta_max)
        posterior_list.append(Sigma_x_sol)
        
        if step > 0:
            diff_norm = np.linalg.norm(posterior_list[step] - posterior_list[step-1], 'fro')
            conv_norms.append(diff_norm)
            if diff_norm < tol:
                print(f"Early stopping at iteration {step} with convergence norm {diff_norm:.4e}")
                break
        
        Sigma_x_minus = A @ Sigma_x_sol @ A.T + Sigma_w_nom

    return {
        "A": A,
        "Sigma_w_nom": Sigma_w_nom,
        "C": C,
        "Sigma_v_nom": Sigma_v_nom,
        "phi_T": phi_T,
        "theta_max": theta_max,
        "posterior_list": posterior_list,
        "conv_norms": conv_norms
    }


# -------------------------------------------------------
# Main: Run 10 experiments using theta_max
# -------------------------------------------------------
if __name__=="__main__":
    tol = 1e-4  # convergence tolerance for final convergence norm
    n_experiments = 20
    
    for exp_num in range(n_experiments):
        print(f"\n=== Experiment {exp_num+1} ===")
        res = run_dr_kf_once(n=10, m=10, steps=200, N_samples=20, T=20, dist_type="normal")
        final_norm = res["conv_norms"][-1]
        final_trace = np.trace(res["posterior_list"][-1])
        converged = final_norm < tol
        
        print(f"Final convergence norm: {final_norm:.4e}")
        print(f"Final posterior trace: {final_trace:.4e}")
        print(f"Convergence: {'YES' if converged else 'NO'}")
