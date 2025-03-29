import pickle

# --- Load Data from Both Files ---
results_file_gaussian = "./results/estimator8/overall_results_normal_normal.pkl"
results_file_quadratic = "./results/estimator8/overall_results_quadratic_quadratic.pkl"

with open(results_file_gaussian, 'rb') as f:
    overall_results_gaussian = pickle.load(f)
with open(results_file_quadratic, 'rb') as f:
    overall_results_quadratic = pickle.load(f)

# --- Define Parameters and Filter Labels ---
robust_params = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0]
filter_order = ['finite', 'inf', 'bcot', 'risk', 'drkf_inf']
filter_labels = {
    'finite': "Time-varying Kalman filter",
    'inf': "Time-invariant Kalman filter",
    'bcot': "BCOT filter",
    'risk': "Risk-Sensitive filter",
    'drkf_inf': "DR Kalman filter (ours)"
}

# --- Determine Optimal Robustness Parameter for Each Robust Filter ---
# For Gaussian noise:
best_rp_gaussian = {}
# For U-Quadratic noise:
best_rp_quadratic = {}

for filt in filter_order:
    if filt in ['finite', 'inf']:
        best_rp_gaussian[filt] = None
        best_rp_quadratic[filt] = None
    else:
        best_cost_gaussian = float('inf')
        best_cost_quadratic = float('inf')
        for rp in robust_params:
            cost_gauss = overall_results_gaussian[rp]['cost'][filt]
            if cost_gauss < best_cost_gaussian:
                best_cost_gaussian = cost_gauss
                best_rp_gaussian[filt] = rp
            cost_quad = overall_results_quadratic[rp]['cost'][filt]
            if cost_quad < best_cost_quadratic:
                best_cost_quadratic = cost_quad
                best_rp_quadratic[filt] = rp

# --- Generate Combined LaTeX Table ---
latex_table = ""
latex_table += "\\begin{table}[t]\n"
latex_table += "\\centering\n"
latex_table += "\\caption{Mean and standard deviation of LQR cost and average MSE under Gaussian and U-Quadratic noise, computed over 20 runs.}\n"
latex_table += "\\setlength{\\tabcolsep}{4pt}\n"
latex_table += "\\begin{tabular}{lccc}\n"
latex_table += "\\hline\n"
latex_table += "\\textbf{Method} & \\textbf{LQR cost} & \\textbf{Average MSE} & \\textbf{Best $\\theta$} \\\\\n"
latex_table += "\\hline\n"

# --- Gaussian noise section ---
latex_table += "\\multicolumn{4}{c}{\\textbf{Gaussian noise}} \\\\\n"
latex_table += "\\hline\n"

for filt in filter_order:
    # For Gaussian noise, use the optimal robust parameter if available
    if best_rp_gaussian[filt] is not None:
        rp = best_rp_gaussian[filt]
        best_theta = f"{rp}"
    else:
        rp = robust_params[0]  # use default value for metrics
        best_theta = "--"
    cost_mean = overall_results_gaussian[rp]['cost'][filt]
    cost_std  = overall_results_gaussian[rp]['cost_std'][filt]
    mse_mean  = overall_results_gaussian[rp]['mse'][filt]
    mse_std   = overall_results_gaussian[rp]['mse_std'][filt]
    
    row = f"{filter_labels[filt]} & {cost_mean:.1f} ({cost_std:.1f}) & {mse_mean:.3f} ({mse_std:.3f}) & {best_theta} \\\\ \n"
    latex_table += row

latex_table += "\\hline\n"

# --- U-Quadratic noise section ---
latex_table += "\\multicolumn{4}{c}{\\textbf{U-Quadratic noise}} \\\\\n"
latex_table += "\\hline\n"

for filt in filter_order:
    # For U-Quadratic noise, use the optimal robust parameter if available
    if best_rp_quadratic[filt] is not None:
        rp = best_rp_quadratic[filt]
        best_theta = f"{rp}"
    else:
        rp = robust_params[0]
        best_theta = "--"
    cost_mean = overall_results_quadratic[rp]['cost'][filt]
    cost_std  = overall_results_quadratic[rp]['cost_std'][filt]
    mse_mean  = overall_results_quadratic[rp]['mse'][filt]
    mse_std   = overall_results_quadratic[rp]['mse_std'][filt]
    
    row = f"{filter_labels[filt]} & {cost_mean:.1f} ({cost_std:.1f}) & {mse_mean:.3f} ({mse_std:.3f}) & {best_theta} \\\\ \n"
    latex_table += row

latex_table += "\\hline\n"
latex_table += "\\end{tabular}\n"
latex_table += "\\label{tab:combined_results}\n"
latex_table += "\\end{table}\n"

# Print the final LaTeX table code
print(latex_table)
