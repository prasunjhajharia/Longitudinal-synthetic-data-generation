import numpy as np
from scipy.special import gammaln
import math

class BaseModel:

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.mu0 = np.zeros(num_nodes + 1)  # Prior mean 
        self.nu = num_nodes + 3             # Prior degrees of freedom
        self.alpha = num_nodes + 1          # Prior hyperparameter 
        self.T0 = np.eye(num_nodes + 1)     # Prior scale matrix 
    
    def compute_local_marginal_likelihood(self, data_subset, parent_set):
        """
        Compute the local log marginal likelihood for a child node given a parent set
        using the Bayesian Gaussian equivalent (BGe) score.
        """
        m = data_subset.shape[1]  # Number of time points

        if m == 0:
            return 0.0

        N = data_subset.shape[0]  # Dimensionality = number of variables (parents + child)
        
        # Compute empirical covariance matrix
        data_mean = np.mean(data_subset, axis=1, keepdims=True)
        S = (data_subset - data_mean) @ (data_subset - data_mean).T
        mu0_subset = self.mu0[:N].reshape(-1, 1)
        diff = data_mean - mu0_subset
        T0_subset = self.T0[:N, :N]
        T = T0_subset + S + (self.nu * m / (self.nu + m)) * (diff @ diff.T)
        
        # Compute log marginal likelihood
        log_ml = (-N * m / 2) * np.log(2 * np.pi)
        log_ml += (N / 2) * np.log(self.nu / (self.nu + m))
        log_ml += self.log_c(N, self.alpha) - self.log_c(N, self.alpha + m)
        log_ml += (self.alpha / 2) * np.log(np.linalg.det(T0_subset))
        log_ml -= ((self.alpha + m) / 2) * np.log(np.linalg.det(T))
        
        return log_ml
    
    def log_c(self, N, alpha):
        """Log of normalizing constant for Wishart distribution"""
        log_c = -N * alpha / 2 * np.log(2)
        log_c -= N * (N - 1) / 4 * np.log(np.pi)
        for i in range(1, N + 1):
            log_c -= gammaln((alpha + 1 - i) / 2)
        return log_c
    
    def compute_bge_score(self, data, graph, allocation):
        """
        Compute the total BGe score for the entire dynamic Bayesian network
        by summing local scores across segments and nodes.
        """
        log_score = 0.0
        
        for node in range(self.num_nodes):
            parents = list(graph.predecessors(node))
            if node in parents:
                parents.remove(node)
            
            # Get all segments for this node
            segments = self.get_node_segments(allocation, node)
            
            for seg_idx, (start, end) in enumerate(segments):
                if end - start <= 1:
                    continue # Skip too-small segments
                
                # Align child (t+1 to end), parents (t to end-1)
                child_data = data[node, start+1:end].reshape(1, -1)
                
                if parents:
                    parent_data = np.vstack([data[p, start:end-1] for p in parents])
                    
                    # Check for self-loop
                    if graph.has_edge(node, node):
                        self_parent_data = data[node, start:end-1].reshape(1, -1)
                        if parent_data.shape[0] > 0:
                            parent_data = np.vstack([parent_data, self_parent_data])
                        else:
                            parent_data = self_parent_data
                    data_subset = np.vstack([parent_data, child_data])
                else:
                    if graph.has_edge(node, node):
                        parent_data = data[node, start:end-1].reshape(1, -1)
                        data_subset = np.vstack([parent_data, child_data])
                    else:
                        data_subset = child_data
                
                # Check if segment has enough data points
                if data_subset.shape[1] < 2:
                    continue
                
                # Compute score for node given its parents
                parent_set = list(graph.predecessors(node))
                log_score += self.compute_local_marginal_likelihood(data_subset, parent_set)
                
                # Subtract marginal likelihood of parents
                if len(parent_set) > 0:
                    parent_data_subset = data_subset[:-1, :]
                    if parent_data_subset.shape[0] > 0:
                        log_score -= self.compute_local_marginal_likelihood(parent_data_subset, [])
        
        return log_score
    
    def get_node_segments(self, allocation, node):
        """
        Given a node's allocation (segment labeling), return list of (start, end) time ranges.
        """
        segments = []
        current_segment_start = 0
        current_segment_value = allocation[node, 0]
        
        for t in range(1, allocation.shape[1]):
            if allocation[node, t] != current_segment_value:
                segments.append((current_segment_start, t))
                current_segment_start = t
                current_segment_value = allocation[node, t]
        
        segments.append((current_segment_start, allocation.shape[1]))
        return segments