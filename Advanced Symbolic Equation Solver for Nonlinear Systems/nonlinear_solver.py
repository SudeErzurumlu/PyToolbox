import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Define variables
x, y, z = sp.symbols('x y z')

# Define a system of nonlinear equations
equations = [
    sp.sin(x) + sp.exp(y) - z**2 - 1,
    sp.cos(y) + z - x**3 + 4,
    x**2 + y**2 + z**2 - 10
]

# Convert equations to functions
eq_funcs = [sp.lambdify((x, y, z), eq) for eq in equations]

# Jacobian Matrix
def jacobian(equations, variables):
    """
    Compute the Jacobian matrix of a system of equations.
    Args:
        equations (list): List of sympy equations.
        variables (list): List of sympy symbols (variables).
    Returns:
        sympy.Matrix: Jacobian matrix.
    """
    return sp.Matrix([[sp.diff(eq, var) for var in variables] for eq in equations])

jacobian_matrix = jacobian(equations, [x, y, z])
print("\nJacobian Matrix:")
sp.pprint(jacobian_matrix)

# Numerical Solver for Nonlinear Systems
def solve_numerically(funcs, initial_guess):
    """
    Solve a nonlinear system of equations numerically.
    Args:
        funcs (list): List of functions representing the system.
        initial_guess (list): Initial guess for the solution.
    Returns:
        list: Numerical solution of the system.
    """
    def wrapped_system(vars):
        return [f(*vars) for f in funcs]
    solution = fsolve(wrapped_system, initial_guess)
    return solution

# Example: Solve numerically with an initial guess
initial_guess = [1, 1, 1]
numerical_solution = solve_numerically(eq_funcs, initial_guess)
print("\nNumerical Solution:")
print(f"x = {numerical_solution[0]}, y = {numerical_solution[1]}, z = {numerical_solution[2]}")

# Homotopy Continuation for Parameterized Systems
def homotopy_continuation(eq, param, start, end, steps=100):
    """
    Solve a parameterized nonlinear equation using homotopy continuation.
    Args:
        eq (sympy.Expr): Equation to solve.
        param (sympy.Symbol): Parameter to vary.
        start (float): Start value of the parameter.
        end (float): End value of the parameter.
        steps (int): Number of steps to evaluate.
    Returns:
        tuple: Parameter values and corresponding solutions.
    """
    param_values = np.linspace(start, end, steps)
    solutions = []
    for value in param_values:
        substituted_eq = eq.subs(param, value)
        solution = sp.solvers.solve(substituted_eq, x)
        solutions.append(solution)
    return param_values, solutions

# Example: Homotopy continuation with a parameterized equation
param = sp.Symbol('a')
parametric_eq = sp.sin(x) - param * x**2 + 1
param_values, solutions = homotopy_continuation(parametric_eq, param, 0, 5)

# Visualization of Homotopy Continuation
plt.figure()
for sol in solutions:
    if sol:
        plt.plot(param_values, [float(s) for s in sol], label=f"Root")
plt.xlabel("Parameter (a)")
plt.ylabel("Solution (x)")
plt.title("Homotopy Continuation")
plt.legend()
plt.grid()
plt.show()

# Critical Points Analysis
def critical_points(jacobian, variables):
    """
    Find critical points of a system using the Jacobian determinant.
    Args:
        jacobian (sympy.Matrix): Jacobian matrix of the system.
        variables (list): List of sympy symbols (variables).
    Returns:
        list: Critical points of the system.
    """
    determinant = jacobian.det()
    critical_solutions = sp.solve(determinant, variables)
    return critical_solutions

critical_pts = critical_points(jacobian_matrix, [x, y, z])
print("\nCritical Points:")
sp.pprint(critical_pts)

# Visualize the system's solutions in 3D
def plot_3d_solutions(equations, variables, solution_range):
    """
    Plot solutions to nonlinear equations in 3D space.
    Args:
        equations (list): List of sympy equations.
        variables (list): List of sympy symbols.
        solution_range (tuple): Range for plotting solutions (min, max).
    """
    x_vals = np.linspace(solution_range[0], solution_range[1], 100)
    y_vals = np.linspace(solution_range[0], solution_range[1], 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            z_sol = sp.solvers.solve(equations[2].subs({variables[0]: X[i, j], variables[1]: Y[i, j]}), variables[2])
            Z[i, j] = float(z_sol[0]) if z_sol else 0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title("3D Solutions to the System")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

# Plot solutions
plot_3d_solutions(equations, [x, y, z], (-2, 2))
