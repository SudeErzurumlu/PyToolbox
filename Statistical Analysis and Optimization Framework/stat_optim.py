import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize
import random

# Statistical Methods

class BayesianInference:
    def __init__(self, prior, likelihood, evidence):
        self.prior = prior
        self.likelihood = likelihood
        self.evidence = evidence

    def posterior(self):
        return (self.prior * self.likelihood) / self.evidence


class Bootstrapping:
    def __init__(self, data, num_samples=1000):
        self.data = data
        self.num_samples = num_samples

    def resample(self):
        return np.random.choice(self.data, size=len(self.data), replace=True)

    def confidence_interval(self, percentile=95):
        samples = [np.mean(self.resample()) for _ in range(self.num_samples)]
        lower = np.percentile(samples, (100 - percentile) / 2)
        upper = np.percentile(samples, 100 - (100 - percentile) / 2)
        return lower, upper


class KDE:
    def __init__(self, data, bandwidth=1.0):
        self.data = data
        self.bandwidth = bandwidth

    def estimate(self, x):
        kernel_values = np.array([np.exp(-(x - xi)**2 / (2 * self.bandwidth**2)) for xi in self.data])
        return np.sum(kernel_values) / len(self.data)

    def plot(self, x_range):
        y_values = np.array([self.estimate(x) for x in x_range])
        plt.plot(x_range, y_values)
        plt.title("Kernel Density Estimate")
        plt.show()


# Linear Programming: Simplex Method Example

class LinearOptimization:
    def __init__(self, c, A, b):
        self.c = c
        self.A = A
        self.b = b

    def solve(self):
        result = linprog(c=self.c, A_ub=self.A, b_ub=self.b, method='simplex')
        return result


# Non-Linear Optimization using Gradient Descent

class GradientDescentOptimizer:
    def __init__(self, func, grad_func, learning_rate=0.01, tolerance=1e-6, max_iter=1000):
        self.func = func
        self.grad_func = grad_func
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iter = max_iter

    def optimize(self, x_init):
        x = x_init
        for _ in range(self.max_iter):
            grad = self.grad_func(x)
            x_new = x - self.learning_rate * grad
            if np.linalg.norm(x_new - x) < self.tolerance:
                break
            x = x_new
        return x


# Optimization Example using Genetic Algorithm

def genetic_algorithm(func, bounds, generations=100, population_size=50, mutation_rate=0.1):
    population = np.random.uniform(bounds[0], bounds[1], (population_size, len(bounds)))
    for generation in range(generations):
        fitness = np.array([func(ind) for ind in population])
        selected = population[np.argsort(fitness)[:population_size//2]]
        offspring = []
        for i in range(population_size//2):
            parent1, parent2 = selected[i], selected[(i + 1) % (population_size//2)]
            crossover_point = random.randint(1, len(bounds) - 1)
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            offspring.append(child)
        offspring = np.array(offspring)
        mutation = np.random.uniform(-mutation_rate, mutation_rate, offspring.shape)
        offspring += mutation
        population = np.concatenate((selected, offspring))
    best_solution = population[np.argmin([func(ind) for ind in population])]
    return best_solution


# Machine Learning Example

class MLModel:
    def __init__(self, model_type="logistic"):
        if model_type == "logistic":
            self.model = LogisticRegression()
        elif model_type == "random_forest":
            self.model = RandomForestClassifier()

    def train(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        return self.model.score(X, y)

    def predict(self, X):
        return self.model.predict(X)


# Test Example Usage

if __name__ == "__main__":
    # Bayesian Inference Example
    prior = 0.5
    likelihood = 0.8
    evidence = 0.7
    bayesian = BayesianInference(prior, likelihood, evidence)
    print(f"Posterior Probability: {bayesian.posterior()}")

    # Bootstrapping Example
    data = np.random.normal(0, 1, 1000)
    bootstrap = Bootstrapping(data)
    lower, upper = bootstrap.confidence_interval()
    print(f"Confidence Interval: [{lower}, {upper}]")

    # KDE Example
    kde = KDE(data)
    kde.plot(np.linspace(-3, 3, 100))

    # Linear Optimization Example
    c = [-1, -2]
    A = [[1, 1], [2, 1]]
    b = [4, 6]
    lin_opt = LinearOptimization(c, A, b)
    print(f"Linear Optimization Solution: {lin_opt.solve()}")

    # Gradient Descent Example
    def func(x):
        return x[0]**2 + x[1]**2

    def grad_func(x):
        return np.array([2*x[0], 2*x[1]])

    gd_optimizer = GradientDescentOptimizer(func, grad_func)
    opt_solution = gd_optimizer.optimize(np.array([10, 10]))
    print(f"Gradient Descent Solution: {opt_solution}")

    # Genetic Algorithm Example
    def func_to_optimize(x):
        return np.sum(x**2)

    bounds = [(-5, 5), (-5, 5)]
    genetic_solution = genetic_algorithm(func_to_optimize, bounds)
    print(f"Genetic Algorithm Solution: {genetic_solution}")

    # Machine Learning Example
    X, y = np.random.rand(100, 5), np.random.randint(0, 2, 100)
    ml_model = MLModel("random_forest")
    ml_model.train(X, y)
    accuracy = ml_model.evaluate(X, y)
    print(f"Model Accuracy: {accuracy}")
