# BGe scoring functions for Bayesian Network structure learning

import numpy as np
from scipy.special import gammaln
from numpy.linalg import slogdet

class BGeScorer:
    def __init__(self, alpha_mu=1.0):
        """
        Initializes the BGe scorer.

        Args:
            alpha_mu (float): Strength of prior mean. Controls influence of prior mean in posterior.
        """
        self.alpha_mu = alpha_mu

    def compute_score(self, X_full, target_node, parent_nodes):
        """
        Computes the BGe marginal likelihood score for a given target node with its parent set.

        This score is derived from the conjugate Normal-Wishart prior and
        evaluates how well a node is explained by its parents.

        Args:
            X_full (pd.DataFrame): The full dataset (n samples × d features)
            target_node (int): Index of the target node (column in X_full)
            parent_nodes (list of int): Indices of parent nodes

        Returns:
            float: BGe log marginal likelihood score
        """
        # Extract data matrix containing target node and its parents
        if len(parent_nodes) == 0:
            X = X_full.iloc[:, [target_node]].copy()
        else:
            X = X_full.iloc[:, [target_node] + parent_nodes].copy()

        n, d = X.shape  # n = number of samples, d = number of variables

        # === Prior parameters ===
        mu_0 = np.zeros(d)             # Prior mean vector (zero vector)
        nu_0 = d + 2                   # Degrees of freedom for the Wishart prior (standard choice)
        T_0 = np.identity(d)           # Prior scale matrix (identity matrix)
        alpha_w = nu_0                 # Precision hyperparameter (as per paper)
        alpha = self.alpha_mu          # Prior mean strength

        # === Empirical statistics ===
        X_bar = np.mean(X, axis=0)                     # Sample mean vector
        S = np.cov(X.T, bias=True) * n                 # Scaled sample covariance matrix

        # === Posterior update ===
        # Posterior scale matrix T_n
        delta_mu = X_bar - mu_0
        T_n = T_0 + S + (alpha * n) / (alpha + n) * np.outer(delta_mu, delta_mu)
        nu_n = nu_0 + n                                # Updated degrees of freedom

        # === Log determinants for T_0 and T_n (used in BGe score formula) ===
        sign0, logdet_T0 = slogdet(T_0)
        signn, logdet_Tn = slogdet(T_n)

        # Check for numerical issues with log determinant
        if sign0 <= 0 or signn <= 0:
            return -np.inf  # Log of non-positive definite matrix is undefined

        # === Compute final BGe score ===
        score = (
            - (n * d / 2) * np.log(np.pi)                             # Normalization term
            + np.sum(gammaln((nu_n + 1 - np.arange(1, d + 1)) / 2))   # Posterior gamma terms
            - np.sum(gammaln((nu_0 + 1 - np.arange(1, d + 1)) / 2))   # Prior gamma terms
            + (nu_0 / 2) * logdet_T0                                  # Prior log determinant
            - (nu_n / 2) * logdet_Tn                                  # Posterior log determinant
        )

        return score
