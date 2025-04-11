# Graph structure representation

class GraphStructure:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.adjacency = {i: [] for i in range(num_nodes)}

    def add_edge(self, parent, child):
        if parent not in self.adjacency[child]:
            self.adjacency[child].append(parent)

    def remove_edge(self, parent, child):
        if parent in self.adjacency[child]:
            self.adjacency[child].remove(parent)

    def get_parents(self, node):
        return self.adjacency[node]

    def copy(self):
        new_graph = GraphStructure(self.num_nodes)
        for node, parents in self.adjacency.items():
            new_graph.adjacency[node] = parents[:]
        return new_graph
