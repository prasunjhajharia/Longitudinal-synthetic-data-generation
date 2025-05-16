import numpy as np
import math

class Allocation:
    def __init__(self, num_nodes, max_segments=10):
        self.num_nodes = num_nodes
        self.max_segments = max_segments

    def propose_allocation_move(self, current_allocation, current_K):
        """
        Propose a changepoint move (birth, death, or reallocation) for one randomly chosen node.
        Returns:
            new_allocation: updated segment allocation matrix [nodes x time]
            new_K: updated number of segments per node
            acceptance_ratio: prior x proposal ratio (used in MH acceptance)
        """
        new_allocation = current_allocation.copy()
        new_K = current_K.copy()
        
        # Select random node for changepoint move
        node = np.random.randint(self.num_nodes)
        current_K_node = current_K[node]

        # Decide move type based on number of segments
        if current_K_node == 1:
            move_type = 'birth'
        elif current_K_node == self.max_segments:
            move_type = 'death'
        else:
            move_type = np.random.choice(['birth', 'death', 'reallocate'], 
                                         p=[0.45, 0.45, 0.1])

        # BIRTH MOVE: Add a new changepoint
        if move_type == 'birth':
            proposed_K_node = current_K_node + 1
            possible_positions = [
                t for t in range(1, current_allocation.shape[1])
                if current_allocation[node, t] == current_allocation[node, t-1]
            ]
            if not possible_positions:
                return current_allocation, current_K, 1.0
            
            # Select new changepoint position
            new_change_point = np.random.choice(possible_positions)
            current_val = current_allocation[node, new_change_point - 1]

            # Increment future segment labels
            new_allocation[node, new_change_point:] = current_val + 1
            new_K[node] = proposed_K_node

            # Compute prior ratio from truncated Poisson(λ=1) on segment count
            λ = 1.0
            prior_ratio = (math.exp(-λ) / math.factorial(proposed_K_node - 1)) / \
                          (math.exp(-λ) / math.factorial(current_K_node - 1))

            # Proposal ratio
            forward_prob = 0.45 / len(possible_positions)
            backward_prob = 0.45 / proposed_K_node
            proposal_ratio = forward_prob / backward_prob if backward_prob > 0 else 1.0

            return new_allocation, new_K, prior_ratio * proposal_ratio

        # DEATH MOVE: Remove a changepoint
        elif move_type == 'death':
            if current_K_node <= 1:
                return current_allocation, current_K, 1.0

            # Identify current changepoints (segment boundaries)
            change_points = [
                t for t in range(1, current_allocation.shape[1])
                if current_allocation[node, t] != current_allocation[node, t - 1]
            ]
            if not change_points:
                return current_allocation, current_K, 1.0

            # Randomly select changepoint to remove
            cp_to_remove = np.random.choice(change_points)
            current_val = current_allocation[node, cp_to_remove]

            # Merge segment by decrementing all future segment IDs
            mask = new_allocation[node, :] > current_val
            new_allocation[node, mask] -= 1
            new_K[node] = current_K_node - 1

            # Prior ratio (same λ = 1.0)
            prior_ratio = new_K[node] * math.exp(0)  # simplified ratio

            # Proposal ratio
            forward_prob = 0.45 / len(change_points)
            possible_new_positions = current_allocation.shape[1] - new_K[node]
            backward_prob = 0.45 / max(1, possible_new_positions)
            proposal_ratio = forward_prob / backward_prob if backward_prob > 0 else 1.0

            return new_allocation, new_K, prior_ratio * proposal_ratio

        # REALLOCATE MOVE: Move a changepoint
        else:
            change_points = [
                t for t in range(1, current_allocation.shape[1])
                if current_allocation[node, t] != current_allocation[node, t - 1]
            ]
            if not change_points:
                return current_allocation, current_K, 1.0

            # Choose changepoint to move
            cp_to_move = np.random.choice(change_points)
            sorted_cps = sorted(change_points)
            idx = sorted_cps.index(cp_to_move)
            prev_cp = sorted_cps[idx - 1] if idx > 0 else 0
            next_cp = sorted_cps[idx + 1] if idx < len(sorted_cps) - 1 else current_allocation.shape[1]

            # Valid new positions for changepoint
            possible_positions = list(range(prev_cp + 1, next_cp))
            if cp_to_move in possible_positions:
                possible_positions.remove(cp_to_move)

            if not possible_positions:
                return current_allocation, current_K, 1.0

            # Reallocate changepoint
            new_cp = np.random.choice(possible_positions)
            temp_allocation = current_allocation.copy()
            segment_val = temp_allocation[node, cp_to_move]

            if new_cp < cp_to_move:
                temp_allocation[node, new_cp:cp_to_move] = segment_val
            else:
                temp_allocation[node, cp_to_move:new_cp] = temp_allocation[node, cp_to_move - 1]

            # Re-normalize labels to be contiguous (e.g., 0,1,2,...)
            _, new_allocation[node, :] = np.unique(temp_allocation[node, :], return_inverse=True)

            prior_ratio = 1.0
            proposal_ratio = 1.0

            return new_allocation, new_K, prior_ratio * proposal_ratio
