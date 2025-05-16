import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Generate synthetic time series data with known structure
# X[t] depends on X[t−1], Y[t] depends on X[t-1] + noise
def generate_synthetic_data_2node(num_time_points=100, epsilon=0.5, snr=50):
    np.random.seed(42)
    m = num_time_points
    X = np.zeros(m)
    Y = np.zeros(m)
    phi_X = np.random.normal(0, 1, m)
    phi_Y = np.random.normal(0, 1, m)

    # Set beta with changepoints
    beta = np.ones(m)
    third = m // 3
    beta[third:(2 * third)] = -1

    # Generate X as autoregressive
    X[0] = np.random.normal(0, 1)
    for t in range(1, m):
        X[t] = np.sqrt(1 - epsilon**2) * X[t - 1] + epsilon * phi_X[t]

    # Compute noise scale for Y based on SNR
    sigma_beta_X = np.std(beta * X)
    c = sigma_beta_X / snr

    # Generate Y with a lagged dependence on X and noise
    for t in range(m - 1):
        Y[t + 1] = beta[t] * X[t] + c * phi_Y[t + 1]

    return np.vstack([X, Y]), [[third, 2 * third], [third, 2 * third]]

# Discretize continuous time series data into bins
def discretize_segment_data(segment_data, num_bins=3):
    binned_data = np.zeros_like(segment_data, dtype=int)
    bin_edges = []
    bin_probs = []

    for i in range(segment_data.shape[0]):
        bins = np.histogram_bin_edges(segment_data[i], bins=num_bins)
        binned = np.digitize(segment_data[i], bins=bins[:-1], right=False) - 1
        counts = np.bincount(binned, minlength=num_bins)
        probs = counts / counts.sum()

        binned_data[i] = binned
        bin_edges.append(bins)
        bin_probs.append(probs)

    return binned_data, bin_edges, bin_probs