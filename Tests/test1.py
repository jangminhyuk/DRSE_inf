import numpy as np
from scipy.linalg import eigh, inv

def generate_spd_matrix(n):
    M = np.random.rand(n, n)
    return np.dot(M, M.T) + n * np.eye(n)  

n = 5  
num_trials = 10000
tolerance = 1e-10  

inequality_holds = 0
all_equal = 0

for _ in range(num_trials):
    A = generate_spd_matrix(n)
    B = generate_spd_matrix(n)
    
    A_inv = inv(A)
    B_inv = inv(B)
    
    lambda_max_B_inv_A_inv = max(eigh(B_inv - A_inv, eigvals_only=True))
    lambda_max_B_inv = max(eigh(B_inv, eigvals_only=True))
    lambda_max_neg_A_inv = max(eigh(-A_inv, eigvals_only=True))
    lambda_min_A_inv = min(eigh(A_inv, eigvals_only=True))
    lambda_min_B = min(eigh(B, eigvals_only=True))
    lambda_max_A = max(eigh(A, eigvals_only=True))
    
    lhs = lambda_max_B_inv_A_inv
    rhs_1 = lambda_max_B_inv + lambda_max_neg_A_inv
    rhs_2 = lambda_max_B_inv - lambda_min_A_inv
    rhs_3 = (1 / lambda_min_B) - (1 / lambda_max_A)
    
    if lhs <= rhs_1 + tolerance and lhs <= rhs_2 + tolerance and lhs <= rhs_3 + tolerance:
        inequality_holds += 1
    if abs(rhs_1 - rhs_2) < tolerance and abs(rhs_2 - rhs_3) < tolerance:
        all_equal += 1

print(f"Out of {num_trials} trials:")
print(f"Inequality held: {inequality_holds} times")
print(f"rhs_1 ≈ rhs_2 ≈ rhs_3 within tolerance: {all_equal} times")
