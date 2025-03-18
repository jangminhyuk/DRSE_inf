import numpy as np
import scipy.linalg as la

# Function to generate a random symmetric positive definite matrix
def generate_spd_matrix(n):
    A = np.random.randn(n, n)
    return A.T @ A + n * np.eye(n)  # Ensuring positive definiteness

# Function to compute the Bures-Wasserstein distance and check the condition
def check_trace_condition(A, B, theta):
    B_sqrt = la.sqrtm(B)
    middle_term = la.sqrtm(B_sqrt @ A @ B_sqrt)
    trace_expr = np.trace(A + B - 2 * middle_term)
    return trace_expr <= theta**2, trace_expr

# Function to check the matrix inequality B^{-1} - A^{-1} \preceq \phi I
def check_inverse_condition(A, B, phi):
    A_inv = la.inv(A)
    B_inv = la.inv(B)
    max_eig = np.linalg.eigvalsh(B_inv - A_inv).max()
    return max_eig <= phi

# Function to generate A such that d(A, B) <= theta
def generate_psd_matrix_near_B(B, theta):
    n = B.shape[0]
    B_sqrt = la.sqrtm(B)

    for _ in range(100):  # Maximum attempts
        X = np.random.randn(n, n)
        A_candidate = X @ X.T  # Random PSD matrix

        # Modify A to be closer to B
        A = 0.5 * (A_candidate + B)  

        # Check if A satisfies the trace condition
        trace_condition_holds, trace_expr = check_trace_condition(A, B, theta)
        if trace_condition_holds:
            return A

    raise ValueError("Failed to generate a suitable A within the maximum attempts.")

# Parameters
num_tests = 1000  # Number of test cases
n_x = 5  # Dimension of the matrices
success_count = 0
iter = 0

while iter <= num_tests:
    # Generate random SPD matrix B
    B = generate_spd_matrix(n_x)

    # Compute Traces and Eigenvalues
    tr_B = np.trace(B)
    lambda_min_B = np.linalg.eigvalsh(B).min()

    # Ensure phi is chosen such that (1 / lambda_min_B - phi) > 0
    while True:
        phi_min = 1 / lambda_min_B - 1 / tr_B
        phi = np.random.uniform(phi_min, phi_min + 10)
        if (1 / lambda_min_B - phi) > 0:
            break

    # Compute theta_max safely
    term_inside_sqrt = 1 / (1 / lambda_min_B - phi)
    if term_inside_sqrt >= tr_B:
        theta_max = np.sqrt(term_inside_sqrt) - np.sqrt(tr_B)
    else:
        continue  # Skip this test case if theta_max is invalid

    # Ensure theta is within a valid range
    if theta_max > 0:
        theta = np.random.uniform(0, theta_max)

        # Generate a random SPD matrix A such that d(A, B) <= theta
        try:
            A = generate_psd_matrix_near_B(B, theta)
            print(f"Iteration {iter+1}/{num_tests}")
        except ValueError:
            print('Failed to generate matrix A, skipping...')
            continue  # Skip test case if we fail to generate a valid A

        iter += 1
        # Verify trace condition
        trace_condition_holds, _ = check_trace_condition(A, B, theta)
        trace_condition_holds_1 = np.trace(A) <= (theta + np.sqrt(tr_B)) ** 2

        # Compute lambda_max(A) bound
        lambda_max_A_bound = (theta + np.sqrt(tr_B)) ** 2

        # Check if the inequality holds
        condition_holds = lambda_max_A_bound <= 1 / (1 / lambda_min_B - phi)

        # Verify by checking eigenvalues of B^{-1} - A^{-1}
        actual_holds = check_inverse_condition(A, B, phi)

        if condition_holds == actual_holds and trace_condition_holds_1:
            success_count += 1

# Print results
print(f"Out of {iter} tests, {success_count} passed ({success_count / iter * 100:.2f}% accuracy).")
