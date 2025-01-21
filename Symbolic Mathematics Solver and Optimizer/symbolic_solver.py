import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class SymbolicSolver:
    def __init__(self):
        pass

    # Simplify symbolic expressions
    def simplify_expression(self, expression):
        """Simplify a symbolic expression."""
        return sp.simplify(expression)

    # Solve equations
    def solve_equation(self, equation, variable):
        """Solve a symbolic equation for the given variable."""
        return sp.solve(equation, variable)

    # Differentiation
    def differentiate(self, expression, variables):
        """Perform symbolic differentiation with respect to given variables."""
        derivatives = [sp.diff(expression, var) for var in variables]
        return derivatives

    # Integration
    def integrate(self, expression, variable, limits=None):
        """Perform symbolic integration. Use limits for definite integrals."""
        if limits:
            return sp.integrate(expression, (variable, *limits))
        else:
            return sp.integrate(expression, variable)

    # Optimization
    def optimize(self, objective_function, initial_guess):
        """Optimize a symbolic objective function numerically."""
        variables = list(objective_function.free_symbols)
        if len(variables) > 1:
            raise ValueError("Optimization supports single-variable functions only.")

        func = sp.lambdify(variables[0], objective_function, "numpy")
        result = minimize(func, initial_guess)
        return result

    # Visualization
    def plot_function(self, expression, variable, range_vals):
        """Plot a symbolic function over a specified range."""
        func = sp.lambdify(variable, expression, "numpy")
        x_vals = np.linspace(*range_vals, 500)
        y_vals = func(x_vals)

        plt.plot(x_vals, y_vals, label=str(expression))
        plt.xlabel(f"{variable}")
        plt.ylabel("f(x)")
        plt.title("Function Plot")
        plt.legend()
        plt.grid()
        plt.show()


# Example Usage
if __name__ == "__main__":
    solver = SymbolicSolver()

    # Define symbolic variables
    x, y = sp.symbols("x y")

    # Example 1: Simplify expression
    expr = (x**2 + 2*x + 1) / (x + 1)
    simplified = solver.simplify_expression(expr)
    print("Simplified Expression:", simplified)

    # Example 2: Solve equation
    equation = sp.Eq(x**2 - 4, 0)
    solutions = solver.solve_equation(equation, x)
    print("Solutions:", solutions)

    # Example 3: Differentiation
    expr = sp.sin(x) * sp.exp(x)
    derivatives = solver.differentiate(expr, [x])
    print("Derivative:", derivatives)

    # Example 4: Integration
    integral = solver.integrate(expr, x)
    print("Indefinite Integral:", integral)

    # Example 5: Definite Integral
    definite_integral = solver.integrate(expr, x, (0, 2))
    print("Definite Integral:", definite_integral)

    # Example 6: Optimization
    objective = x**2 - 4 * x + 4
    result = solver.optimize(objective, initial_guess=0)
    print("Optimization Result:", result)

    # Example 7: Plot Function
    solver.plot_function(objective, x, range_vals=(-2, 6))
