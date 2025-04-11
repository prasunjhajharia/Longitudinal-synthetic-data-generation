# src/changepoint.py

class ChangePointManager:
    def __init__(self, T):
        self.T = T  # total time steps

    def initialize_segments(self, K):
        """
        Uniform segmentation into K regions (returns K-1 changepoints).
        """
        segment_length = self.T // K
        return [i * segment_length for i in range(1, K)]

    def assign_segments(self, breakpoints):
        """
        Maps each time step to a segment index.
        """
        breakpoints = sorted(breakpoints)
        segments = [0] * self.T
        current_segment = 0
        bp_index = 0

        for t in range(self.T):
            if bp_index < len(breakpoints) and t >= breakpoints[bp_index]:
                current_segment += 1
                bp_index += 1
            segments[t] = current_segment

        return segments
