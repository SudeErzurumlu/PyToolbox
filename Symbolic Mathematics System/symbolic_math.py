import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Core Symbolic Math Framework

class SymbolicMath:
    def __init__(self):
        self.x, self.y, self.z = sp.symbols("x y z")

    def simplify_expr(self, expr):
        return sp.simplify(expr)

    def expand_expr(self, expr):
        return sp.expand(expr)

    def factor_expr(self, expr):
        return sp.factor(expr)

    def solve_eq(self, equation, variable=None):
        if variable is None:
            variable = self.x
        return sp.solve(equation, variable)

    def differentiate(self, expr, variable=None):
        if variable is None:
            variable = self.x
        return sp.diff(expr, variable)

    def integrate(self, expr, variable=None, a=None, b=None):
        if variable is None:
            variable = self.x
        if a is not None and b is not None:
            return sp.integrate(expr, (variable, a, b))
        return sp.integrate(expr, variable)

# Linear Algebra Framework

class LinearAlgebra:
    def __init__(self):
        pass

    def determinant(self, matrix):
        return sp.Matrix(matrix).det()

    def eigenvalues(self, matrix):
        return sp.Matrix(matrix).eigenvals()

    def eigenvectors(self, matrix):
        return sp.Matrix(matrix).eigenvects()

    def inverse(self, matrix):
        return sp.Matrix(matrix).inv()

# Graphing Utilities

class Graphing:
    def __init__(self):
        pass

    def plot_function(self, func, var, x_range=(-10, 10), num_points=1000):
        x_vals = np.linspace(x_range[0], x_range[1], num_points)
        y_vals = [sp.lambdify(var, func)(x) for x in x_vals]
        plt.plot(x_vals, y_vals, label=f"{func}")
        plt.title("Function Plot")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_derivative(self, func, var, x_range=(-10, 10), num_points=1000):
        derivative = sp.diff(func, var)
        x_vals = np.linspace(x_range[0], x_range[1], num_points)
        y_vals = [sp.lambdify(var, derivative)(x) for x in x_vals]
        plt.plot(x_vals, y_vals, label=f"{derivative}")
        plt.title("Derivative Plot")
        plt.xlabel("x")
        plt.ylabel("f'(x)")
        plt.grid(True)
        plt.legend()
        plt.show()

# Example Usage

if __name__ == "__main__":
    sm = SymbolicMath()
    la = LinearAlgebra()
    graph = Graphing()

    # Symbolic Algebra
    expr = sm.x**3 - 3*sm.x**2 + 4*sm.x - 12
    print("Original Expression:", expr)
    print("Simplified:", sm.simplify_expr(expr))
    print("Expanded:", sm.expand_expr(expr))
    print("Factored:", sm.factor_expr(expr))

    # Solving Equations
    eq = sm.x**2 - 5*sm.x + 6
    print("Roots of Equation:", sm.solve_eq(eq))

    # Differentiation and Integration
    func = sp.sin(sm.x) * sp.exp(sm.x)
    print("Derivative:", sm.differentiate(func))
    print("Indefinite Integral:", sm.integrate(func))
    print("Definite Integral (0 to pi):", sm.integrate(func, a=0, b=sp.pi))

    # Linear Algebra Operations
    matrix = [[1, 2], [3, 4]]
    print("Determinant:", la.determinant(matrix))
    print("Eigenvalues:", la.eigenvalues(matrix))
    print("Eigenvectors:", la.eigenvectors(matrix))
    print("Inverse Matrix:", la.inverse(matrix))

    # Graphing
    graph.plot_function(func, sm.x, x_range=(-2, 2))
    graph.plot_derivative(func, sm.x, x_range=(-2, 2))
