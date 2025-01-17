import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class MultiVarOptimization:
    def __init__(self):
        self.variables = []

    def define_variables(self, names):
        """Define symbolic variables for the system."""
        self.variables = sp.symbols(names)
        return self.variables

    def symbolic_gradient(self, function):
        """Compute the symbolic gradient of a multi-variable function."""
        return sp.derive_by_array(function, self.variables)

    def symbolic_hessian(self, function):
        """Compute the symbolic Hessian matrix of a multi-variable function."""
        gradient = self.symbolic_gradient(function)
        return sp.derive_by_array(gradient, self.variables)

    def solve_lagrange(self, objective, constraints):
        """Solve a constrained optimization problem using Lagrange multipliers."""
        lagrange_multipliers = sp.symbols(f"λ:{len(constraints)}")
        lagrangian = objective + sum(l * g for l, g in zip(lagrange_multipliers, constraints))

        # Solve the system of equations for critical points
        equations = list(sp.derive_by_array(lagrangian, self.variables + list(lagrange_multipliers)))
        solutions = sp.solve(equations, self.variables + list(lagrange_multipliers), dict=True)
        return solutions

    def eigen_analysis(self, matrix):
        """Perform eigenvalue and eigenvector analysis."""
        eigenvalues = matrix.eigenvals()
        eigenvectors = matrix.eigenvects()
        return eigenvalues, eigenvectors

    def visualize_function(self, function):
        """Visualize a 2D function surface."""
        if len(self.variables) != 2:
            raise ValueError("Visualization is supported only for functions with two variables.")

        x, y = self.variables
        func = sp.lambdify((x, y), function, modules='numpy')

        x_vals = np.linspace(-10, 10, 100)
        y_vals = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = func(X, Y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_title("Function Surface")
        plt.show()


# Example Usage
if __name__ == "__main__":
    optimizer = MultiVarOptimization()

    # Define symbolic variables
    x, y = optimizer.define_variables("x y")

    # Define a sample function and constraints
    objective_function = x**2 + y**2 - 4 * x * y + 5
    constraints = [x + y - 2]

    # Compute symbolic gradient and Hessian
    gradient = optimizer.symbolic_gradient(objective_function)
    hessian = optimizer.symbolic_hessian(objective_function)
    print("Gradient:", gradient)
    print("Hessian:", hessian)

    # Solve the constrained optimization problem using Lagrange multipliers
    solutions = optimizer.solve_lagrange(objective_function, constraints)
    print("Solutions to the constrained optimization problem:")
    for sol in solutions:
        print(sol)

    # Perform eigenvalue and eigenvector analysis on the Hessian matrix
    hessian_matrix = sp.Matrix(hessian)
    eigenvalues, eigenvectors = optimizer.eigen_analysis(hessian_matrix)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:")
    for ev in eigenvectors:
        print(ev)

    # Visualize the function surface
    optimizer.visualize_function(objective_function)
