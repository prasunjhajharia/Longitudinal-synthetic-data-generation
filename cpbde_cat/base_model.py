import numpy as np
from scipy.special import gammaln

class CategoricalBaseModel:
    def __init__(self, num_nodes, num_categories=3):
        self.num_nodes = num_nodes                        # Total number of nodes (variables) in the network
        self.num_categories = num_categories              # Number of discrete states each variable can take
        self.alpha = np.ones(num_categories)              # Dirichlet prior for each category (symmetric)

    def compute_local_marginal_likelihood(self, data_subset, parent_set):
        """
        Computes the local marginal likelihood (log-space) for a node given its parent set.
        Uses the BDe scoring formula for discrete data.
        """
        m = data_subset.shape[1]                          # Number of data points (columns = time points)
        if m == 0:
            return 0.0                                    # No data → no contribution to likelihood

        child_data = data_subset[-1, :]                   # Last row = child variable

        if not parent_set:
            # No parents, simple Dirichlet-multinomial marginal likelihood
            counts = np.bincount(child_data, minlength=self.num_categories)
            return np.sum(gammaln(counts + self.alpha)) - gammaln(np.sum(counts + self.alpha))

        parent_data = data_subset[:-1, :]                 # All rows except last = parents

        if len(parent_set) == 1:
            # Single parent, use direct 2D count table
            joint_counts = np.zeros((self.num_categories, self.num_categories))
            for p, c in zip(parent_data[0], child_data):
                joint_counts[p, c] += 1
        else:
            # Multiple parents, encode joint parent state as a single integer index
            parent_states = np.zeros(parent_data.shape[1], dtype=int)
            for i in range(parent_data.shape[1]):
                state = 0
                for j, val in enumerate(parent_data[:, i]):
                    state += val * (self.num_categories ** j)  # Mixed radix encoding
                parent_states[i] = state

            num_parent_states = self.num_categories ** len(parent_set)
            joint_counts = np.zeros((num_parent_states, self.num_categories))
            for ps, cs in zip(parent_states, child_data):
                joint_counts[ps, cs] += 1

        # Compute the log marginal likelihood across all parent configurations
        log_ml = 0
        for ps in range(joint_counts.shape[0]):
            counts = joint_counts[ps, :]
            log_ml += np.sum(gammaln(counts + self.alpha)) - gammaln(np.sum(counts + self.alpha))

        return log_ml

    def compute_bde_score(self, data, graph, allocation):
        """
        Computes the full BDe score of a dynamic Bayesian network (DBN) with changepoints.
        Iterates through each node and each segment, applying local marginal likelihoods.
        """
        log_score = 0.0

        for node in range(self.num_nodes):
            parents = list(graph.predecessors(node))      # Get parents of the node from the graph
            segments = self.get_node_segments(allocation, node)  # Get time segments (changepoint boundaries)

            for start, end in segments:
                if end - start <= 1:
                    continue 

                # Get child data: node's values from t = start+1 to end
                child_data = data[node, start+1:end].reshape(1, -1)

                # Get parent data: aligned with child (t = start to end-1)
                if parents:
                    parent_data = np.vstack([data[p, start:end-1] for p in parents])

                    # If node has a self-loop, include its own past values as a parent
                    if graph.has_edge(node, node):
                        self_data = data[node, start:end-1].reshape(1, -1)
                        parent_data = np.vstack([parent_data, self_data]) if parent_data.size else self_data

                    data_subset = np.vstack([parent_data, child_data])
                else:
                    # No parents, only self-loop
                    if graph.has_edge(node, node):
                        parent_data = data[node, start:end-1].reshape(1, -1)
                        data_subset = np.vstack([parent_data, child_data])
                    else:
                        data_subset = child_data

                # Ignore segments with very few points
                if data_subset.shape[1] < 5: # changed from 2 to 5 to align more with the paper
                    continue

                parent_set = parents.copy()
                if graph.has_edge(node, node):
                    parent_set.append(node)  # Add self-loop as parent if present

                # Add local log marginal likelihood
                log_score += self.compute_local_marginal_likelihood(data_subset, parent_set)

                # Subtract marginal likelihood of parents alone (to normalize for BDe)
                if len(parent_set) > 0:
                    parent_data_subset = data_subset[:-1, :]
                    if parent_data_subset.shape[0] > 0:
                        log_score -= self.compute_local_marginal_likelihood(parent_data_subset, [])

        return log_score

    def get_node_segments(self, allocation, node):
        """
        Given an allocation matrix [nodes x time], returns list of (start, end) time segments
        for a given node where the changepoint label remains constant.
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
