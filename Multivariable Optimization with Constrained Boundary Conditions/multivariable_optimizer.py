import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

# Define variables
x, y = sp.symbols('x y')

# Define objective function and constraint
objective_func = x**2 + y**2  # Objective: Minimize the distance to origin
constraint_1 = x + y - 1  # Constraint 1: x + y = 1
constraint_2 = x - y  # Constraint 2: x - y = 0

# Convert to numeric functions for optimization
obj_numeric = sp.lambdify((x, y), objective_func, "numpy")
constr_1_numeric = sp.lambdify((x, y), constraint_1, "numpy")
constr_2_numeric = sp.lambdify((x, y), constraint_2, "numpy")

# Gradient of the objective function
grad_objective = [sp.diff(objective_func, var) for var in [x, y]]
grad_numeric = [sp.lambdify((x, y), g, "numpy") for g in grad_objective]

# Lagrange Multipliers Method
def lagrange_multiplier_method(obj, constraints, variables):
    """
    Solve optimization problem using Lagrange multipliers.
    Args:
        obj (sympy.Expr): Objective function to minimize.
        constraints (list): List of constraints (equality).
        variables (list): List of variables.
    Returns:
        sympy.Solution: Solution for the optimization problem.
    """
    lagrangian = obj
    lambdas = sp.symbols('lambda1 lambda2')
    for i, constr in enumerate(constraints):
        lagrangian += lambdas[i] * constr

    # Compute the gradients of the Lagrangian
    gradients = [sp.diff(lagrangian, var) for var in variables + list(lambdas)]
    solution = sp.solve(gradients, variables + list(lambdas))
    return solution

# Apply Lagrange multiplier method
lagrange_solution = lagrange_multiplier_method(objective_func, [constraint_1, constraint_2], [x, y])
print("\nLagrange Multiplier Method Solution:")
for var, sol in zip([x, y], lagrange_solution[:2]):
    print(f"{var}: {sol}")

# Numerical Solution with Constraints using Conjugate Gradient
def numerical_optimizer():
    """
    Optimize the objective function with the given constraints using numerical methods.
    Returns:
        dict: Resulting optimization.
    """
    # Define constraints
    cons = [
        NonlinearConstraint(constr_1_numeric, 0, 0),  # x + y = 1
        NonlinearConstraint(constr_2_numeric, 0, 0)   # x - y = 0
    ]

    # Initial guess
    initial_guess = [0.5, 0.5]

    # Minimize using conjugate gradient method
    result = minimize(lambda vars: obj_numeric(*vars), initial_guess, jac=lambda vars: [grad_numeric[0](*vars), grad_numeric[1](*vars)], constraints=cons, method='trust-constr')
    return result

# Solve numerically
numerical_result = numerical_optimizer()
print("\nNumerical Optimization Result:")
print(f"Optimal x: {numerical_result.x[0]}, Optimal y: {numerical_result.x[1]}")
print(f"Objective Function Value: {numerical_result.fun}")

# Visualization of the optimization
def plot_optimization():
    """
    Plot the contour of the objective function and the optimal solution.
    """
    x_vals = np.linspace(-2, 2, 400)
    y_vals = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = X**2 + Y**2

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, 50, cmap='viridis')
    plt.scatter(numerical_result.x[0], numerical_result.x[1], color='red', label='Optimal Solution')
    plt.title("Optimization Contour with Constraints")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot the optimization process
plot_optimization()

# KKT Conditions for Constrained Optimization
def kkt_conditions(obj, constraints, grad_obj, grad_constraints):
    """
    Solve optimization problem using KKT conditions.
    Args:
        obj (sympy.Expr): Objective function.
        constraints (list): List of constraints (inequality or equality).
        grad_obj (list): Gradients of the objective function.
        grad_constraints (list): Gradients of the constraints.
    Returns:
        sympy.Solution: Solution for the optimization problem.
    """
    # Lagrange multiplier (lambda) for each constraint
    lambdas = sp.symbols('lambda1 lambda2')
    
    # KKT conditions
    kkt_system = []
    for i in range(len(constraints)):
        kkt_system.append(grad_obj[i] - lambdas[i] * grad_constraints[i])

    # Add the complementary slackness condition (for inequality constraints)
    kkt_system.append([constraint for constraint in constraints])

    # Solve the system
    solution = sp.solve(kkt_system, [x, y] + list(lambdas))
    return solution

# KKT Analysis (applicable for inequality constraints, placeholder in this example)
kkt_solution = kkt_conditions(objective_func, [constraint_1, constraint_2], grad_objective, [sp.diff(constraint_1, var) for var in [x, y]] + [sp.diff(constraint_2, var) for var in [x, y]])
print("\nKKT Conditions Solution:")
for var, sol in zip([x, y], kkt_solution[:2]):
    print(f"{var}: {sol}")
