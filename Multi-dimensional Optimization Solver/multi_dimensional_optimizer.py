import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class MultiDimensionalOptimizer:
    def __init__(self, dimensions, objective_function, constraints=None):
        self.dimensions = dimensions
        self.objective_function = objective_function
        self.constraints = constraints

    def gradient_descent(self, initial_guess, learning_rate=0.01, max_iter=1000):
        """Gradient Descent method for optimization."""
        current_point = np.array(initial_guess)
        history = [current_point]
        
        for _ in range(max_iter):
            grad = self.compute_gradient(current_point)
            current_point = current_point - learning_rate * grad
            history.append(current_point)
            
            if np.linalg.norm(grad) < 1e-5:  # Convergence check
                break
        return np.array(history)
    
    def compute_gradient(self, point, epsilon=1e-5):
        """Compute the gradient of the objective function numerically."""
        grad = np.zeros_like(point)
        for i in range(len(point)):
            point_copy = point.copy()
            point_copy[i] += epsilon
            grad[i] = (self.objective_function(point_copy) - self.objective_function(point)) / epsilon
        return grad

    def simulated_annealing(self, initial_guess, temp=100, cooling_rate=0.99, max_iter=1000):
        """Simulated Annealing optimization."""
        current_point = np.array(initial_guess)
        current_value = self.objective_function(current_point)
        history = [current_point]
        
        for _ in range(max_iter):
            new_point = current_point + np.random.randn(len(current_point))
            new_value = self.objective_function(new_point)
            
            if new_value < current_value or np.random.rand() < np.exp((current_value - new_value) / temp):
                current_point = new_point
                current_value = new_value
            temp *= cooling_rate
            history.append(current_point)
        
        return np.array(history)

    def genetic_algorithm(self, population_size=100, generations=500, mutation_rate=0.01):
        """Genetic Algorithm for optimization."""
        population = np.random.randn(population_size, self.dimensions)
        history = []

        for gen in range(generations):
            fitness = np.array([self.objective_function(ind) for ind in population])
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]

            new_population = population[:10]  # Elite selection

            # Crossover
            for i in range(10, population_size, 2):
                parent1, parent2 = population[np.random.randint(10)], population[np.random.randint(10)]
                crossover_point = np.random.randint(1, self.dimensions)
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                new_population = np.vstack([new_population, child1, child2])
            
            # Mutation
            for i in range(10, population_size):
                if np.random.rand() < mutation_rate:
                    mutation_point = np.random.randint(self.dimensions)
                    population[i, mutation_point] += np.random.randn()

            history.append(population[0])
        
        return np.array(history)

    def visualize_optimization(self, history):
        """Visualize the optimization progress."""
        history = np.array(history)
        plt.plot(history[:, 0], history[:, 1], marker="o")
        plt.title("Optimization Process")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()

    def optimize(self, method='gradient_descent', initial_guess=None, **kwargs):
        """Optimize the function using the specified method."""
        if method == 'gradient_descent':
            return self.gradient_descent(initial_guess, **kwargs)
        elif method == 'simulated_annealing':
            return self.simulated_annealing(initial_guess, **kwargs)
        elif method == 'genetic_algorithm':
            return self.genetic_algorithm(**kwargs)
        else:
            raise ValueError("Unknown method specified.")

# Example Usage
if __name__ == "__main__":
    # Define a sample multi-dimensional objective function (e.g., Rosenbrock function)
    def objective_function(x):
        return sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    # Instantiate the optimizer
    optimizer = MultiDimensionalOptimizer(dimensions=2, objective_function=objective_function)

    # Optimize using Gradient Descent
    initial_guess = np.array([1.5, 1.5])
    history_gd = optimizer.optimize(method='gradient_descent', initial_guess=initial_guess)
    optimizer.visualize_optimization(history_gd)

    # Optimize using Simulated Annealing
    history_sa = optimizer.optimize(method='simulated_annealing', initial_guess=initial_guess)
    optimizer.visualize_optimization(history_sa)

    # Optimize using Genetic Algorithm
    history_ga = optimizer.optimize(method='genetic_algorithm', population_size=50, generations=100)
    optimizer.visualize_optimization(history_ga)
