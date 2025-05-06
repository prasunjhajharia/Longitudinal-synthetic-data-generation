import matplotlib.pyplot as plt
import numpy as np

def plot_time_series_with_changepoints(data, true_cps, title="Time Series with True Change Points"):
    """Plot only the time series with ground-truth changepoints overlaid"""
    num_nodes, T = data.shape
    plt.figure(figsize=(12, 6))

    for i in range(num_nodes):
        plt.plot(data[i], label=f'Node {i}', linewidth=2)
        for cp in true_cps[i]:
            plt.axvline(x=cp, color='r', linestyle='--', alpha=0.5)

    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_segment_histogram(K_samples, node_names):
    """Plot posterior distribution over number of segments for each node."""
    K_array = np.array(K_samples)  # shape: [num_samples, num_nodes]
    num_nodes = K_array.shape[1]

    for i in range(num_nodes):
        plt.figure()
        plt.hist(K_array[:, i], bins=np.arange(1, 12), align='left', rwidth=0.8)
        plt.title(f"Posterior distribution of segments for Node {node_names[i]}")
        plt.xlabel("Number of segments (K)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
