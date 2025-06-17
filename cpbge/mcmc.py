import numpy as np
import networkx as nx
from collections import defaultdict
from cpbge.base_model import BaseModel
from cpbge.allocation import Allocation
from cpbge.graph_operations import GraphOperations


class MCMC:
    def __init__(self, num_nodes, max_segments=10):
        self.base_model = BaseModel(num_nodes) # Handles scoring using BGe
        self.graph_ops = GraphOperations(num_nodes)  # Proposes graph modifications
        self.allocation_ops = Allocation(num_nodes, max_segments) # Proposes segmentation changes
        self.num_nodes = num_nodes 

    def mcmc_inference(self, data, num_iterations=10000, burn_in=1000):
        """
        Run MCMC to jointly infer the Bayesian network structure and per-node changepoint allocations.
        """
        num_time_points = data.shape[1] - 1  # For DBN, we lose one time point
        
        # Initialize graph with self-loops
        current_graph = nx.DiGraph()
        current_graph.add_nodes_from(range(self.num_nodes))
        for node in range(self.num_nodes):
            current_graph.add_edge(node, node)  # Add self-loops
        
        # Allocate all time points to a single segment for all nodes
        current_allocation = np.zeros((self.num_nodes, num_time_points), dtype=int)
        current_K = np.ones(self.num_nodes, dtype=int) # Each node starts with 1 segment
        
        # Compute initial model score
        current_score = self.base_model.compute_bge_score(data, current_graph, current_allocation)
        
        # Prepare containers for storing samples and stats
        samples = []
        graph_counts = defaultdict(int)
        allocation_samples = []
        K_samples = []
        scores = []
        
        # MCMC loop
        for iteration in range(num_iterations):
            # Alternate between graph and allocation movesg
            if np.random.rand() < 0.5:
                
                # Graph move
                new_graph = self.graph_ops.propose_graph_move(current_graph)
                if new_graph is not None:
                    new_score = self.base_model.compute_bge_score(data, new_graph, current_allocation)
                    acceptance_ratio = np.exp(new_score - current_score)
                    # Accept or reject the new graph
                    if np.random.rand() < min(1.0, acceptance_ratio):
                        current_graph = new_graph
                        current_score = new_score
            else:
                # Allocation move
                new_allocation, new_K, move_ratio = self.allocation_ops.propose_allocation_move(current_allocation, current_K)
                new_score = self.base_model.compute_bge_score(data, current_graph, new_allocation)
                acceptance_ratio = np.exp(new_score - current_score) * move_ratio
                # Accept or reject the new segmentation
                if np.random.rand() < min(1.0, acceptance_ratio):
                    current_allocation = new_allocation
                    current_K = new_K
                    current_score = new_score

            
            # Store samples after burn-in
            if iteration >= burn_in:
                if iteration % 1000 == 0:  # Save every 10th sample to reduce autocorrelation
                    samples.append((current_graph.copy(), current_allocation.copy(), current_K.copy()))
                    graph_counts[tuple(sorted(current_graph.edges()))] += 1
                    allocation_samples.append(current_allocation.copy())
                    K_samples.append(current_K.copy())
                    scores.append(current_score)

        # Compute marginal edge probabilities
        total_samples = len(samples)
        edge_probs = {}
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                edge_probs[(i, j)] = sum(1 for s in samples if s[0].has_edge(i, j)) / total_samples
        
        # Compute posterior mean number of segments per node
        mean_K = np.mean(K_samples, axis=0)
        
        # Find MAP allocation (maximum a posteriori)
        best_idx = np.argmax(scores)
        map_allocation = allocation_samples[best_idx]
        
        return edge_probs, mean_K, map_allocation, samples