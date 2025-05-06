import numpy as np
import networkx as nx

class GraphOperations:
    def __init__(self, num_nodes, fan_in=3):
        self.num_nodes = num_nodes # Number of variables/nodes in the graph
        self.fan_in = fan_in # Max number of parents allowed per node

    def propose_graph_move(self, current_graph):
        """Propose a new graph by adding, removing or reversing an edge"""
        new_graph = current_graph.copy()
        nodes = list(current_graph.nodes())
        
        # Randomly choose an edge operation
        operation = np.random.choice(['add', 'remove', 'reverse'])
        
        if operation == 'add':
            # Select two distinct nodes
            u, v = np.random.choice(nodes, 2, replace=True)
            # Only add if edge doesn't already exist and fan-in limit is respected
            if not new_graph.has_edge(u, v) and len(list(new_graph.predecessors(v))) < self.fan_in:
                new_graph.add_edge(u, v)
            else:
                return None
            
        elif operation == 'remove':
            # Try to remove a randomly selected edge
            edges = list(current_graph.edges())
            if edges:
                u, v = edges[np.random.randint(len(edges))]
                new_graph.remove_edge(u, v)
            else:
                return None 
            
        elif operation == 'reverse':
             # Try to reverse a randomly selected edge
            edges = list(current_graph.edges())
            if edges:
                u, v = edges[np.random.randint(len(edges))]
                # Reverse edge only if reversed edge doesn't already exist and fan-in is respected
                if not new_graph.has_edge(v, u) and len(list(new_graph.predecessors(u))) < self.fan_in:
                    new_graph.remove_edge(u, v)
                    new_graph.add_edge(v, u)
                else:
                    return None  
            else:
                return None 
        
        return new_graph