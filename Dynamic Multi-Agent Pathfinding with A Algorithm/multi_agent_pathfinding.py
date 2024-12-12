import heapq
import numpy as np

class AStarPathfinder:
    def __init__(self, grid, start, goal):
        """
        Initializes the pathfinder with a grid, start, and goal positions.
        """
        self.grid = grid
        self.start = start
        self.goal = goal
        self.open_list = []
        self.came_from = {}
        self.g_score = {start: 0}
        self.f_score = {start: self.heuristic(start, goal)}

    def heuristic(self, node, goal):
        """
        Calculates Manhattan distance heuristic.
        """
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    def neighbors(self, node):
        """
        Returns valid neighbors for a given node.
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        neighbors = []
        for d in directions:
            neighbor = (node[0] + d[0], node[1] + d[1])
            if 0 <= neighbor[0] < self.grid.shape[0] and 0 <= neighbor[1] < self.grid.shape[1]:
                if self.grid[neighbor] == 0:  # Check for obstacles
                    neighbors.append(neighbor)
        return neighbors

    def find_path(self):
        """
        Executes the A* search algorithm to find a path.
        """
        heapq.heappush(self.open_list, (self.f_score[self.start], self.start))
        while self.open_list:
            _, current = heapq.heappop(self.open_list)
            if current == self.goal:
                return self.reconstruct_path(current)

            for neighbor in self.neighbors(current):
                tentative_g_score = self.g_score[current] + 1
                if neighbor not in self.g_score or tentative_g_score < self.g_score[neighbor]:
                    self.came_from[neighbor] = current
                    self.g_score[neighbor] = tentative_g_score
                    self.f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, self.goal)
                    if neighbor not in [i[1] for i in self.open_list]:
                        heapq.heappush(self.open_list, (self.f_score[neighbor], neighbor))
        return None  # No path found

    def reconstruct_path(self, current):
        """
        Reconstructs the path from start to goal.
        """
        path = []
        while current in self.came_from:
            path.append(current)
            current = self.came_from[current]
        path.append(self.start)
        return path[::-1]

# Example Usage
grid = np.zeros((10, 10))
grid[3:7, 5] = 1  # Add obstacles
pathfinder = AStarPathfinder(grid, (0, 0), (9, 9))
path = pathfinder.find_path()
print("Path:", path)
