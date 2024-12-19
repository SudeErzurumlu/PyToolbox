import numpy as np
import cvxpy as cp
from scipy.linalg import cholesky, LinAlgError
import matplotlib.pyplot as plt

# Define the optimization problem parameters
num_variables = 50  # Number of decision variables (e.g., energy, time)
num_constraints = 100  # Number of nonlinear constraints
A = np.random.randn(num_constraints, num_variables)  # Random constraint matrix
b = np.random.randn(num_constraints)  # Random right-hand side vector
Q = np.random.randn(num_variables, num_variables)  # Quadratic cost term for the objective function
Q = Q.T @ Q  # Ensure it is positive definite for convexity

# Define optimization variables
x = cp.Variable(num_variables)
objective = cp.Minimize(0.5 * cp.quad_form(x, Q))  # Quadratic objective

# Nonlinear constraints (e.g., power limits, availability, etc.)
constraints = [
    A @ x <= b,  # Linear constraints
    cp.norm(x, 'inf') <= 1,  # Bounded decision variables
    x >= 0  # Non-negativity constraint
]

# Form the problem
problem = cp.Problem(objective, constraints)

# Solve the problem using the interior-point method
try:
    problem.solve(solver=cp.SCS)  # Using SCS solver for large-scale problems
    print(f"Optimal solution found: {x.value}")
except cp.error.SolverError as e:
    print(f"Solver error: {e}")

# Visualization of the solution
x_value = x.value
plt.plot(x_value, label="Optimized Variable x")
plt.title("Optimization Result")
plt.xlabel("Variable Index")
plt.ylabel("Value of x")
plt.legend()
plt.grid(True)
plt.show()
