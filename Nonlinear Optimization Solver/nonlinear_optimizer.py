import numpy as np
from scipy.optimize import minimize
import sympy as sp
import matplotlib.pyplot as plt

# Symbolic Gradient and Hessian Computation
def symbolic_gradient_hessian(expr, variables):
    """
    Computes the symbolic gradient and Hessian matrix for a given expression.
    Args:
        expr (sympy.Expr): Objective function.
        variables (list): List of variables.
    Returns:
        tuple: Symbolic gradient (vector) and Hessian (matrix).
    """
    gradient = [sp.diff(expr, var) for var in variables]
    hessian = [[sp.diff(grad, var) for var in variables] for grad in gradient]
    return gradient, hessian

# Example: Nonlinear Function
x, y = sp.symbols('x y')
objective = x**4 + y**4 - 4 * x**2 - 4 * y**2 + x * y
gradient, hessian = symbolic_gradient_hessian(objective, [x, y])

print("Objective Function:")
sp.pprint(objective)
print("\nGradient:")
sp.pprint(gradient)
print("\nHessian:")
sp.pprint(hessian)

# Numerical Optimization
def numerical_optimizer(func, grad, x0, constraints=None):
    """
    Solves nonlinear optimization problems using Sequential Quadratic Programming.
    Args:
        func (callable): Objective function.
        grad (callable): Gradient function.
        x0 (array-like): Initial guess.
        constraints (dict, optional): Constraints for the optimization problem.
    Returns:
        OptimizeResult: Optimization result.
    """
    result = minimize(func, x0, jac=grad, constraints=constraints, method='trust-constr')
    return result

# Example Numerical Objective and Gradient
def func(x):
    return x[0]**4 + x[1]**4 - 4 * x[0]**2 - 4 * x[1]**2 + x[0] * x[1]

def grad(x):
    return np.array([4 * x[0]**3 - 8 * x[0] + x[1], 4 * x[1]**3 - 8 * x[1] + x[0]])

# Constraints: x + y <= 2, x >= 0, y >= 0
constraints = [{'type': 'ineq', 'fun': lambda x: 2 - (x[0] + x[1])},
               {'type': 'ineq', 'fun': lambda x: x[0]},
               {'type': 'ineq', 'fun': lambda x: x[1]}]

# Initial Guess
x0 = np.array([0.5, 0.5])

# Solve
result = numerical_optimizer(func, grad, x0, constraints)
print("\nOptimization Result:")
print(result)

# Plotting the Optimization Landscape
def plot_landscape(func, x_range, y_range, result):
    """
    Plots the optimization landscape and solution path.
    Args:
        func (callable): Objective function.
        x_range (tuple): Range for x-axis.
        y_range (tuple): Range for y-axis.
        result (OptimizeResult): Optimization result.
    """
    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    X, Y = np.meshgrid(x, y)
    Z = func([X, Y])
    
    plt.figure(figsize=(10, 7))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar()
    plt.plot(result.x[0], result.x[1], 'ro', label='Optimal Solution')
    plt.title("Optimization Landscape")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

plot_landscape(lambda xy: func([xy[0], xy[1]]), (-3, 3), (-3, 3), result)

# Multi-Objective Optimization (Pareto Frontier)
def pareto_frontier(funcs, bounds, n_points=50):
    """
    Computes the Pareto frontier for a set of objective functions.
    Args:
        funcs (list): List of objective functions.
        bounds (list): Bounds for decision variables.
        n_points (int): Number of points to evaluate.
    Returns:
        np.ndarray: Pareto frontier points.
    """
    pareto_points = []
    for i in range(n_points):
        weights = np.random.dirichlet(np.ones(len(funcs)))
        combined_func = lambda x: sum(w * f(x) for w, f in zip(weights, funcs))
        result = minimize(combined_func, x0, bounds=bounds)
        pareto_points.append(result.x)
    return np.array(pareto_points)

# Example Multi-Objective Optimization
def f1(x):
    return x[0]**2 + x[1]**2

def f2(x):
    return (x[0] - 1)**2 + (x[1] - 1)**2

pareto_points = pareto_frontier([f1, f2], bounds=[(0, 2), (0, 2)])
print("\nPareto Frontier Points:")
print(pareto_points)

# Plot Pareto Frontier
plt.figure(figsize=(8, 6))
plt.scatter(pareto_points[:, 0], pareto_points[:, 1], c='blue', label='Pareto Points')
plt.title("Pareto Frontier")
plt.xlabel("f1")
plt.ylabel("f2")
plt.legend()
plt.show()
