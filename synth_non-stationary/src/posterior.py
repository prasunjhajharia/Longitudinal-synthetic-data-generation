# Aggregate and analyze posterior samples

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

class PosteriorTracker:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.edge_counts = np.zeros((num_nodes, num_nodes))
        self.sample_count = 0
        self.changepoint_configs = defaultdict(list)

    def update(self, graph, changepoints_by_subject=None):
        self.sample_count += 1
        for child in range(self.num_nodes):
            for parent in graph.get_parents(child):
                self.edge_counts[parent, child] += 1

        if changepoints_by_subject:
            for subj_id, cps in changepoints_by_subject.items():
                self.changepoint_configs[subj_id].append(tuple(sorted(cps)))

    def get_edge_probabilities(self):
        return self.edge_counts / self.sample_count

    def get_most_common_changepoints(self):
        return {
            subj_id: Counter(cps_list).most_common(1)[0]
            for subj_id, cps_list in self.changepoint_configs.items()
        }

    def plot_edge_probabilities(self):
        probs = self.get_edge_probabilities()
        plt.figure(figsize=(8, 6))
        plt.imshow(probs, cmap='Blues', interpolation='none')
        plt.colorbar(label="Posterior Probability")
        plt.title("Edge Posterior Probabilities")
        plt.xlabel("Child")
        plt.ylabel("Parent")
        plt.show()

    def print_summary(self):
        print("Posterior edge probabilities:")
        probs = self.get_edge_probabilities()
        flat = [((i, j), p) for i in range(self.num_nodes) for j in range(self.num_nodes) if (p := probs[i, j]) > 0]
        for (i, j), prob in sorted(flat, key=lambda x: -x[1]):
            print(f"{i} → {j}: {prob:.3f}")

        print("Most common changepoint configs:")
        for subj, (cps, freq) in self.get_most_common_changepoints().items():
            print(f"Subject {subj}: {list(cps)} (freq: {freq})")
