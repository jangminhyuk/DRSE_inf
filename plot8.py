import pickle
import matplotlib.pyplot as plt

# Load the overall results from the saved pickle file.
results_file = "./results/estimator8/overall_results_normal_normal.pkl"
with open(results_file, 'rb') as f:
    overall_results = pickle.load(f)

robust_params = [0.1, 0.2, 0.4, 0.5, 1.0, 2.0]
filter_order = ['finite', 'inf', 'bcot', 'risk', 'drkf_inf']
filter_labels = {
    'finite': "Standard KF (finite)",
    'inf': "Standard KF (infinite)",
    'bcot': "BCOT",
    'risk': "Risk-Sensitive",
    'drkf_inf': "DRKF (ours)"
}

# Compute best robust parameters for robust filters (for reference)
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

# -------------------------------
# Plot LQR Cost vs Robust Parameter
# -------------------------------
plt.figure(figsize=(8, 6))
for filt in filter_order:
    cost_means = []
    cost_stds = []
    for rp in robust_params:
        cost_means.append(overall_results[rp]['cost'][filt])
        cost_stds.append(overall_results[rp]['cost_std'][filt])
    plt.errorbar(robust_params, cost_means, yerr=cost_stds,
                 label=filter_labels[filt], capsize=5,
                 marker='o', linestyle='-')

plt.xlabel('Robust Parameter')
plt.ylabel('LQR Cost')
plt.title('LQR Cost vs Robust Parameter')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./results/estimator8/LQR_cost_plot.png', dpi=300)
plt.show()

# -------------------------------
# Plot Averaged MSE vs Robust Parameter
# -------------------------------
plt.figure(figsize=(8, 6))
for filt in filter_order:
    mse_means = []
    mse_stds = []
    for rp in robust_params:
        mse_means.append(overall_results[rp]['mse'][filt])
        mse_stds.append(overall_results[rp]['mse_std'][filt])
    plt.errorbar(robust_params, mse_means, yerr=mse_stds,
                 label=filter_labels[filt], capsize=5,
                 marker='o', linestyle='-')

plt.xlabel('Robust Parameter')
plt.ylabel('Averaged MSE')
plt.title('Averaged MSE vs Robust Parameter')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./results/estimator8/Averaged_MSE_plot.png', dpi=300)
plt.show()
