import pickle

# Load the overall results from the saved pickle file.
results_file = "./results/estimator8/overall_results_normal_normal.pkl"
with open(results_file, 'rb') as f:
    overall_results = pickle.load(f)

robust_params = [0.1, 0.2, 0.4, 0.5, 1.0, 2.0]
filter_order = ['finite', 'inf', 'bcot', 'risk', 'drkf_inf']
filter_labels = {
    'finite': "Time-varying KF",
    'inf': "Time-invariant KF",
    'bcot': "BCOT",
    'risk': "Risk-Sensitive",
    'drkf_inf': "DRKF (ours)"
}

best_rp = {}
for filt in filter_order:
    if filt in ['finite', 'inf']:
        best_rp[filt] = None
    else:
        best_cost = float('inf')
        for rp in robust_params:
            current_cost = overall_results[rp]['cost'][filt]
            if current_cost < best_cost:
                best_cost = current_cost
                best_rp[filt] = rp

# --- Generate combined LaTeX Table ---
latex_combined = ""
latex_combined += "\\begin{table*}[ht]\n"
latex_combined += "\\centering\n"
latex_combined += "\\caption{Mean and standard deviation of LQR cost and average MSE under -- noise distributions, computed over 20 runs.}\n"
latex_combined += "\\begin{tabular}{ll" + "c" * len(robust_params) + "}\n"
latex_combined += "\\hline\n"

# Combine first two columns in header.
header = "\\multicolumn{2}{c}{Robustness Parameter $\\theta$} & " + " & ".join(map(str, robust_params)) + " \\\\ \n"
latex_combined += header
latex_combined += "\\hline\n"

# LQR Cost section (removed 'Mean (Std)' from the label)
latex_combined += "\\multirow{5}{*}{\\textbf{LQR cost}}"
for filt in ['finite', 'inf']:
    value = overall_results[robust_params[0]]['cost'][filt]
    std = overall_results[robust_params[0]]['cost_std'][filt]
    row = " & " + filter_labels[filt] + " & \\multicolumn{" + str(len(robust_params)) + "}{c}{" + "{:.1f} ({:.1f})".format(value, std) + "} \\\\ \n"
    latex_combined += row

for filt in ['bcot', 'risk', 'drkf_inf']:
    row = " & " + filter_labels[filt]
    for rp in robust_params:
        value = overall_results[rp]['cost'][filt]
        std = overall_results[rp]['cost_std'][filt]
        if rp == best_rp[filt]:
            row += " & \\textbf{{{:.1f} ({:.1f})}}".format(value, std)
        else:
            row += " & {:.1f} ({:.1f})".format(value, std)
    row += " \\\\ \n"
    latex_combined += row

latex_combined += "\\hline\n"

# Averaged MSE section (removed 'Mean (Std)' from the label)
latex_combined += "\\multirow{5}{*}{\\textbf{Averaged MSE}}"
for filt in ['finite', 'inf']:
    mse_value = overall_results[robust_params[0]]['mse'][filt]
    mse_std = overall_results[robust_params[0]]['mse_std'][filt]
    row = " & " + filter_labels[filt] + " & \\multicolumn{" + str(len(robust_params)) + "}{c}{" + "{:.3f} ({:.3f})".format(mse_value, mse_std) + "} \\\\ \n"
    latex_combined += row

for filt in ['bcot', 'risk', 'drkf_inf']:
    row = " & " + filter_labels[filt]
    for rp in robust_params:
        mse_data = overall_results[rp]['mse'].get(filt, None)
        mse_std_data = overall_results[rp]['mse_std'].get(filt, None)
        if mse_data is not None and mse_std_data is not None:
            if rp == best_rp[filt]:
                row += " & \\textbf{{{:.3f} ({:.3f})}}".format(mse_data, mse_std_data)
            else:
                row += " & {:.3f} ({:.3f})".format(mse_data, mse_std_data)
        else:
            row += " & -"
    row += " \\\\ \n"
    latex_combined += row

latex_combined += "\\hline\n"
latex_combined += "\\end{tabular}\n"
latex_combined += "\\label{tab:combined_results}\n"
latex_combined += "\\end{table*}\n"

print(latex_combined)
