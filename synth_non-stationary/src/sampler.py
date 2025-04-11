# RJMCMC for structure and change-points
import random
import numpy as np

class RJMCMCSampler:
    def __init__(self, graph, data, scorer, changepoints):
        """
        Initializes the sampler with the graph structure, dataset, scoring function, and changepoints.

        Args:
            graph (GraphStructure): The graph model to be updated.
            data (list of pd.DataFrame): Time-series data per subject.
            scorer (BGeScorer): Score function for evaluating graph structures.
            changepoints (dict): subject_id → list of changepoint indices.
        """
        self.graph = graph
        self.data = data
        self.scorer = scorer
        self.changepoints = changepoints

    def run(self, num_iterations, burn_in=200, sample_every=10, posterior=None):
        """
        Main RJMCMC loop: iteratively propose structure and changepoint moves.

        Args:
            num_iterations (int): Total MCMC steps.
            burn_in (int): Initial samples to discard.
            sample_every (int): Frequency of recording samples.
            posterior (PosteriorTracker or None): Tracks samples if provided.
        """
        for i in range(num_iterations):
            self.structure_move()
            self.changepoint_move()

            if i >= burn_in and i % sample_every == 0 and posterior is not None:
                posterior.update(self.graph, self.changepoints)

            if i % 100 == 0:
                print(f"[INFO] Iteration {i} — Current edges: "
                      f"{[(p, c) for c in range(self.graph.num_nodes) for p in self.graph.get_parents(c)]}")

    def structure_move(self):
        """
        Proposes an add/remove edge operation on the graph and accepts based on BGe score change.
        """
        num_nodes = self.graph.num_nodes
        child = random.randint(0, num_nodes - 1)
        parent = random.randint(0, num_nodes - 1)

        if parent == child:
            return  # Avoid self-loop

        current_parents = self.graph.get_parents(child)

        if parent in current_parents:
            self.graph.remove_edge(parent, child)
            new_score = self._score_whole_dataset(child)
            self.graph.add_edge(parent, child)  # revert
        else:
            self.graph.add_edge(parent, child)
            new_score = self._score_whole_dataset(child)
            self.graph.remove_edge(parent, child)  # revert

        old_score = self._score_whole_dataset(child, current_parents)
        delta_score = new_score - old_score

        accepted = False
        if delta_score >= 0 or random.uniform(0, 1) < np.exp(delta_score):
            if parent in current_parents:
                self.graph.remove_edge(parent, child)
            else:
                self.graph.add_edge(parent, child)
            accepted = True

        print(f"[STRUCTURE MOVE] Proposed: {parent} → {child} | Accepted: {accepted} | ΔScore: {delta_score:.3f}")

    def _score_whole_dataset(self, child, parents_override=None):
        """
        Computes the total BGe score across all subjects and segments for a given node.
        """
        total = 0.0
        for subject in range(len(self.data)):
            cps = self.changepoints[subject]
            bounds = [0] + cps + [len(self.data[subject])]

            for i in range(len(bounds) - 1):
                segment = self.data[subject][bounds[i]:bounds[i+1]]
                if len(segment) < 2:
                    continue
                parents = parents_override if parents_override is not None else self.graph.get_parents(child)
                score = self.scorer.compute_score(segment, child, parents)
                total += score
        return total

    def changepoint_move(self):
        """
        Proposes add, remove, or shift moves to update changepoints per subject.
        Uses Metropolis-Hastings acceptance with BGe scoring.
        """
        move_type = random.choice(['add', 'remove', 'shift'])
        subject = random.choice(list(self.changepoints.keys()))
        t_list = self.changepoints[subject]

        if move_type == 'add':
            if len(t_list) >= len(self.data[subject]) - 2:
                return
            proposed_t = random.randint(1, len(self.data[subject]) - 2)
            if proposed_t in t_list:
                return
            new_cps = sorted(t_list + [proposed_t])
            score_before = self._score_segments(subject, t_list)
            score_after = self._score_segments(subject, new_cps)
            if random.uniform(0, 1) < np.exp(score_after - score_before):
                self.changepoints[subject] = new_cps

        elif move_type == 'remove':
            if not t_list:
                return
            to_remove = random.choice(t_list)
            new_cps = [t for t in t_list if t != to_remove]
            score_before = self._score_segments(subject, t_list)
            score_after = self._score_segments(subject, new_cps)
            if random.uniform(0, 1) < np.exp(score_after - score_before):
                self.changepoints[subject] = new_cps

        elif move_type == 'shift':
            if not t_list:
                return
            idx = random.randint(0, len(t_list) - 1)
            t = t_list[idx]
            new_t = max(1, min(len(self.data[subject]) - 2, t + random.choice([-1, 1])))
            if new_t in t_list:
                return
            new_cps = list(t_list)
            new_cps[idx] = new_t
            new_cps.sort()
            score_before = self._score_segments(subject, t_list)
            score_after = self._score_segments(subject, new_cps)
            if random.uniform(0, 1) < np.exp(score_after - score_before):
                self.changepoints[subject] = new_cps

        print(f"[CHANGE POINT MOVE] Proposed: {move_type.upper()}")

    def _score_segments(self, subject, cps):
        """
        Computes BGe score for a single subject over all its segments.
        """
        data = self.data[subject]
        segment_bounds = [0] + cps + [len(data)]
        total_score = 0.0
        for i in range(len(segment_bounds) - 1):
            seg = data[segment_bounds[i]:segment_bounds[i+1]]
            if len(seg) < 2:
                continue
            for child in range(self.graph.num_nodes):
                parents = self.graph.get_parents(child)
                score = self.scorer.compute_score(seg, child, parents)
                total_score += score
        return total_score
