import matplotlib.pyplot as plt
import networkx as nx
from time import sleep

class GraphVisualizer:
    def __init__(self):
        """
        Initializes the graph visualizer.
        """
        self.graph = nx.Graph()

    def add_edge(self, node1, node2):
        """
        Adds an edge to the graph.
        """
        self.graph.add_edge(node1, node2)

    def visualize(self):
        """
        Displays the graph.
        """
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=1500)
        plt.pause(0.5)

    def bfs(self, start_node):
        """
        Visualizes BFS traversal.
        """
        visited = set()
        queue = [start_node]
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                print(f"Visited: {node}")
                self.visualize()
                neighbors = list(self.graph.neighbors(node))
                queue.extend(n for n in neighbors if n not in visited)

    def dfs(self, node, visited=None):
        """
        Visualizes DFS traversal.
        """
        if visited is None:
            visited = set()
        if node not in visited:
            visited.add(node)
            print(f"Visited: {node}")
            self.visualize()
            for neighbor in self.graph.neighbors(node):
                self.dfs(neighbor, visited)

# Example usage:
# graph_viz = GraphVisualizer()
# graph_viz.add_edge(1, 2)
# graph_viz.add_edge(1, 3)
# graph_viz.add_edge(2, 4)
# graph_viz.add_edge(3, 4)
# plt.ion()
# graph_viz.bfs(1)  # Or graph_viz.dfs(1)
