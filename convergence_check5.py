import numpy as np
import cvxpy as cp
import mosek
import control

np.random.seed(42)

# TUNABLE parameter for fixed theta run:
FIXED_THETA = 100

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
# Helper function to enforce positive definiteness
# -------------------------------------------------------
def enforce_positive_definiteness(M, epsilon=1e-3):
    M = (M + M.T) / 2.0
    eigvals = np.linalg.eigvalsh(M)
    min_eig = np.min(eigvals)
    if min_eig < epsilon:
        M = M + (epsilon - min_eig) * np.eye(M.shape[0])
    return M

# -------------------------------------------------------
# Check detectablity and stabilizability 
# -------------------------------------------------------
def is_detectable(A, C, tol=1e-6):
    n = A.shape[0]
    eigvals = np.linalg.eigvals(A)
    for lam in eigvals:
        if abs(lam) >= 1 - tol:
            M_detect = np.vstack([A - lam*np.eye(n), C])
            if np.linalg.matrix_rank(M_detect, tol=tol) < n:
                return False
    return True

def is_stabilizable(A, Sigma_w_nom, tol=1e-6):
    n = A.shape[0]
    try:
        B_nom = np.linalg.cholesky(Sigma_w_nom)
    except np.linalg.LinAlgError:
        return False
    eigvals = np.linalg.eigvals(A)
    for lam in eigvals:
        if abs(lam) >= 1 - tol:
            M_stab = np.hstack([A - lam*np.eye(n), B_nom])
            if np.linalg.matrix_rank(M_stab, tol=tol) < n:
                return False
    return True

# -------------------------------------------------------
# (Optional) Assumption check function.
# (Not redundantly called.)
# -------------------------------------------------------
def check_assumptions(A, Sigma_w_nom, C, Sigma_v_nom, T):
    n = A.shape[0]
    m = C.shape[0]
    if np.any(np.linalg.eigvals(Sigma_w_nom) <= 0):
        raise ValueError("Sigma_w_nom is not positive definite.")
    if np.any(np.linalg.eigvals(Sigma_v_nom) <= 0):
        raise ValueError("Sigma_v_nom is not positive definite.")
    O = control.obsv(A, C)
    if np.linalg.matrix_rank(O) < n:
        raise ValueError("The pair (A, C) is not observable.")
    try:
        B = np.linalg.cholesky(Sigma_w_nom)
    except np.linalg.LinAlgError:
        raise ValueError("Sigma_w_nom is not PD for Cholesky.")
    CC = control.ctrb(A, B)
    if np.linalg.matrix_rank(CC) < n:
        raise ValueError("The pair (A, sqrt(Sigma_w_nom)) is not reachable.")
    O_T_blocks = [C @ np.linalg.matrix_power(A, T-1-i) for i in range(T)]
    O_T = np.vstack(O_T_blocks)
    if np.linalg.matrix_rank(O_T) < n:
        raise ValueError("O_T does not have full column rank; (A, C) may not be observable with T={T}")
    print("Assumptions verified.")

# -------------------------------------------------------
# DR Kalman filter measurement update (only theta_x version)
# -------------------------------------------------------
def dr_kf_solve_measurement_update(Sigma_x_minus_hat, C, Sigma_v_nom, theta_x, delta=1e-6):
    n = Sigma_x_minus_hat.shape[0]
    m = Sigma_v_nom.shape[0]
    #Z_var = cp.Variable((m, m), PSD=True)
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
    import numpy as np

    n = A.shape[0]
    m = C.shape[0]

    Sigma_v_nom = enforce_positive_definiteness(Sigma_v_nom, epsilon=1e-3)
    B = np.linalg.cholesky(Sigma_w_nom)
    sqrt_Sigma_v_nom = np.linalg.cholesky(Sigma_v_nom)

    R_T_blocks = [np.linalg.matrix_power(A, i) @ B for i in range(T)]
    R_T = np.hstack(R_T_blocks)

    O_T_blocks = [C @ np.linalg.matrix_power(A, T-1-i) for i in range(T)]
    O_T = np.vstack(O_T_blocks)
    if np.linalg.matrix_rank(O_T) < n:
        raise ValueError(f"O_T does not have full column rank; (A,C) may not be observable with T={T}")

    O_T_R_blocks = [np.linalg.matrix_power(A, T-1-i) for i in range(T)]
    O_T_R = np.vstack(O_T_R_blocks)

    D_T = np.kron(np.eye(T), sqrt_Sigma_v_nom)

    L_blocks = [[(np.linalg.matrix_power(A, j-i-1) @ B) if (j - i >= 1) else np.zeros((n, n))
                 for j in range(T)] for i in range(T)]
    H_blocks = [[(C @ (np.linalg.matrix_power(A, j-i-1) @ B)) if (j - i >= 1) else np.zeros((m, n))
                 for j in range(T)] for i in range(T)]
    L_T = np.block(L_blocks)
    H_T = np.block(H_blocks)

    I_inner = np.eye(T * n)
    DDT = D_T @ D_T.T
    inv_DDT = np.linalg.inv(DDT)
    inner_term = I_inner + H_T.T @ inv_DDT @ H_T
    inner_inv = np.linalg.inv(inner_term)
    M = L_T @ inner_inv @ L_T.T
    eigvals = np.linalg.eigvals(M)
    lambda_max_val = np.max(np.real(eigvals))
    tilde_phi_T = 1.0 / lambda_max_val

    print(f"tilde_phi_T : {tilde_phi_T}")
    return {
        "R_T": R_T,
        "O_T": O_T,
        "O_T_R": O_T_R,
        "D_T": D_T,
        "L_T": L_T,
        "H_T": H_T,
        "tilde_phi_T": tilde_phi_T
    }

def find_phi_T(O_T, O_T_R, L_T, H_T, D_T, tilde_phi_T, tol_eig=1e-10, bisection_tol=1e-10, max_iter=1000):
    import numpy as np

    M = D_T @ D_T.T + H_T @ H_T.T
    M_inv = np.linalg.inv(M)
    J_T = O_T_R - L_T @ H_T.T @ M_inv @ O_T
    Omega_T = O_T.T @ M_inv @ O_T

    eig_vals = np.linalg.eigvals(Omega_T)
    lambda_min = np.min(np.real(eig_vals))
    print(f"min eig value of Omega_T: {lambda_min}")
    if lambda_min < 0:
        raise ValueError("Omega_T is not positive definite. Check that all assumptions are met.")

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
            return -np.inf
        Omega_phi = Omega_T + J_T.T @ S_phi_inv @ J_T
        Omega_phi = (Omega_phi + Omega_phi.T) / 2.0
        eigvals = np.linalg.eigvals(Omega_phi)
        return np.min(np.real(eigvals))

    phi_lower = 0.0
    phi_upper = tilde_phi_T
    iteration = 0
    while (phi_upper - phi_lower) > bisection_tol and iteration < max_iter:
        iteration += 1
        phi_mid = (phi_lower + phi_upper) / 2.0
        f_mid = lambda_min_Omega(phi_mid)
        if iteration%20==0:
            print(f"Iteration {iteration}: phi = {phi_mid:.12f}, lambda_min = {f_mid:.12e}")
        if f_mid > tol_eig:
            phi_lower = phi_mid
        else:
            phi_upper = phi_mid

    phi_final = phi_lower
    print(f"Bisection converged after {iteration} iterations: phi_T = {phi_final:.8f}, lambda_min = {lambda_min_Omega(phi_final):.8e}")
    return phi_final

def compute_theta_max(A, C, Sigma_w_nom, Sigma_v_nom, phi_T, q=100, T=None):
    """
    Computes theta_max that guarantees convergence of the DR Kalman filter.
    Iterates the standard Kalman Riccati update for q steps and then computes:
         theta_max = sqrt(trace(P_bar) / (1 - phi_T * lambda_max(P_bar))) - sqrt(trace(P_bar))
    """
    P_bar = Sigma_w_nom.copy()
    
    for q_ in range(q):
        CPCT = C @ P_bar @ C.T
        S = CPCT + Sigma_v_nom
        K = P_bar @ C.T @ np.linalg.inv(S)
        P_update = P_bar - K @ C @ P_bar
        P_bar = A @ P_update @ A.T + Sigma_w_nom

        eigvals = np.linalg.eigvals(P_bar)
        lambda_max_val = np.max(np.real(eigvals))
        trace_P = np.trace(P_bar)
        
        if q_ % 20 == 0:
            print(f"{q_}/{q} | Eigenvalues of P_bar:", np.real(eigvals))
            
    print(f"trace_P: {trace_P} , phi_T: {phi_T}, lambda_max_val: {lambda_max_val}")
    term = np.sqrt(trace_P / (1 - phi_T * lambda_max_val))
    
    if term <= 0:
        print("Warning: sqrt(trace_P / (1 - phi_T * lambda_max)) is not positive. Cannot compute theta_max.")
        return None
        
    theta_max = term - np.sqrt(trace_P)
    if theta_max > 0:
        print(f"theta_max: {theta_max}")
    else:
        print(f"Computed theta_max is non-positive. USE Fixed THETA = {FIXED_THETA} instead")
        
    if T is not None:
        print(f"q = {q}, T = {T}")
    return theta_max

def run_simulation_with_theta(A, Sigma_w_nom, C, Sigma_v_nom, theta, steps, tol):
    Sigma_x_minus = np.eye(A.shape[0])
    posterior_list = []
    conv_norms = []
    for step in range(steps):
        Sigma_x_sol = dr_kf_solve_measurement_update(Sigma_x_minus, C, Sigma_v_nom, theta)
        posterior_list.append(Sigma_x_sol)
        if step > 0:
            diff_norm = np.linalg.norm(posterior_list[step] - posterior_list[step-1], 'fro')
            conv_norms.append(diff_norm)
            if diff_norm < tol:
                print(f"Early stopping at iteration {step} with convergence norm {diff_norm:.4e}")
                break
        Sigma_x_minus = A @ Sigma_x_sol @ A.T + Sigma_w_nom
    final_norm = conv_norms[-1] if conv_norms else None
    return {"posterior_list": posterior_list, "conv_norms": conv_norms, "final_norm": final_norm}

def run_experiment_for_system(n=10, m=10, steps=200, T=20, q=100, tol=1e-4):
    # Generate system until (A,C) is observable and (A,sqrt(Sigma_w_nom)) is reachable.
    while True:
        A = np.random.randn(n, n)
        Wr = np.random.randn(n, n)
        mu_w = np.zeros((n, 1))
        #Sigma_w_nom = Wr @ Wr.T + 1e-3 * np.eye(n) # Ensure positive definiteness for nominal covariance
        Sigma_w_nom = np.eye(n)
        C = np.random.randn(m, n)
        Vr = np.random.randn(m, m)
        mu_v = np.zeros((m, 1))
        #Sigma_v_nom = Vr @ Vr.T + 1e-3 * np.eye(m) # Ensure positive definiteness for nominal covariance
        Sigma_v_nom = np.eye(m)
        
        O = control.obsv(A, C)
        observable = (np.linalg.matrix_rank(O) == n)
        try:
            B = np.linalg.cholesky(Sigma_w_nom)
        except np.linalg.LinAlgError:
            reachable = False
        else:
            CC = control.ctrb(A, B)
            reachable = (np.linalg.matrix_rank(CC) == n)
        
        if observable and reachable:
            break

    # Compute matrices and phi_T.
    matrices = compute_matrices(T, A, Sigma_w_nom, C, Sigma_v_nom)
    tilde_phi_T = matrices["tilde_phi_T"]
    print(tilde_phi_T)
    phi_T = find_phi_T(matrices["O_T"], matrices["O_T_R"], matrices["L_T"], matrices["H_T"], matrices["D_T"], tilde_phi_T)
    computed_theta = compute_theta_max(A, C, Sigma_w_nom, Sigma_v_nom, phi_T, q)
    
    # If computed theta is negative/invalid, run only one simulation with FIXED_THETA.
    if computed_theta is None or computed_theta <= 0 or np.isnan(computed_theta):
        computed_theta_used = False
        used_theta = FIXED_THETA
        # Only run fixed simulation.
        result_computed = None
        result_fixed = run_simulation_with_theta(A, Sigma_w_nom, C, Sigma_v_nom, FIXED_THETA, steps, tol)
    else:
        computed_theta_used = True
        used_theta = computed_theta
        # Run both simulations.
        result_computed = run_simulation_with_theta(A, Sigma_w_nom, C, Sigma_v_nom, used_theta, steps, tol)
        result_fixed = run_simulation_with_theta(A, Sigma_w_nom, C, Sigma_v_nom, FIXED_THETA, steps, tol)
    
    return {
        "A": A,
        "Sigma_w_nom": Sigma_w_nom,
        "C": C,
        "Sigma_v_nom": Sigma_v_nom,
        "phi_T": phi_T,
        "computed_theta": computed_theta,
        "theta_used": used_theta,
        "computed_theta_used": computed_theta_used,
        "result_computed": result_computed,
        "result_fixed": result_fixed
    }

if __name__=="__main__":
    tol = 1e-4
    n_experiments = 1000
    # Counters for experiments where computed theta is valid (positive) or negative/invalid.
    pos_count = 0
    neg_count = 0
    # Success counters for the simulation runs:
    success_computed = 0    # for computed theta simulation (only when computed theta is valid)
    success_fixed_pos = 0   # fixed theta simulation success in experiments with positive computed theta
    success_fixed_neg = 0   # fixed theta simulation success in experiments with negative computed theta
    
    for exp_num in range(n_experiments):
        print(f"\n=== Experiment {exp_num+1} ===")
        exp_result = run_experiment_for_system(n=5, m=5, steps=500, T=10, q=100, tol=tol)
        
        if not exp_result["computed_theta_used"]:
            neg_count += 1
            print(f"Experiment {exp_num+1}: Computed theta_max was negative or invalid; used FIXED_THETA = {FIXED_THETA}.")
        else:
            pos_count += 1
        
        if exp_result["computed_theta_used"]:
            final_norm_computed = exp_result["result_computed"]["final_norm"]
            if final_norm_computed is not None and final_norm_computed < tol:
                success_computed += 1
                print("Computed theta simulation converged.")
            else:
                print("Computed theta simulation did NOT converge.")
        else:
            print("No computed theta simulation for this experiment.")
        
        final_norm_fixed = exp_result["result_fixed"]["final_norm"]
        if final_norm_fixed is not None and final_norm_fixed < tol:
            if exp_result["computed_theta_used"]:
                success_fixed_pos += 1
            else:
                success_fixed_neg += 1
            print("Fixed theta simulation (FIXED_THETA) converged.")
        else:
            print("Fixed theta simulation (FIXED_THETA) did NOT converge.")
    
    total_experiments = n_experiments
    print("\n==================================================")
    print(f"Number of experiments where computed theta_max was negative/invalid: {neg_count}")
    print(f"Number of experiments where computed theta_max was positive: {pos_count}")
    if pos_count > 0:
        print(f"Success rate for computed theta_max simulation (positive cases): {success_computed}/{pos_count} = {success_computed/pos_count*100:.2f}%")
    else:
        print("No experiments had a valid computed theta_max.")
    if neg_count > 0:
        print(f"Success rate for fixed theta = {FIXED_THETA} simulation in negative computed theta cases: {success_fixed_neg}/{neg_count} = {success_fixed_neg/neg_count*100:.2f}%")
    if pos_count > 0:
        print(f"Success rate for fixed theta = {FIXED_THETA} simulation in positive computed theta cases: {success_fixed_pos}/{pos_count} = {success_fixed_pos/pos_count*100:.2f}%")
    overall_fixed_rate = (success_fixed_neg + success_fixed_pos) / total_experiments * 100
    print(f"Overall success rate for fixed theta = {FIXED_THETA} simulation: {overall_fixed_rate:.2f}%")
    print(f"Total experiments run: {total_experiments} (with additional fixed theta runs for each system)")
