import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Symbolic Integration
def symbolic_integration(expr, variable, limits=None):
    """
    Performs symbolic integration of a given expression.
    Args:
        expr (sympy.Expr): The mathematical expression.
        variable (sympy.Symbol): The variable of integration.
        limits (tuple, optional): Integration limits (a, b).
    Returns:
        sympy.Expr: Integrated expression or definite integral result.
    """
    if limits:
        return sp.integrate(expr, (variable, *limits))
    return sp.integrate(expr, variable)

# Example: Symbolic Integration
x = sp.symbols('x')
expression = sp.sin(x**2) + x**3
indefinite_integral = symbolic_integration(expression, x)
definite_integral = symbolic_integration(expression, x, (0, 2))

print("Expression:")
sp.pprint(expression)
print("\nIndefinite Integral:")
sp.pprint(indefinite_integral)
print("\nDefinite Integral (0 to 2):")
sp.pprint(definite_integral)

# Numerical Validation
def numerical_integration(func, a, b):
    """
    Computes a numerical definite integral using SciPy.
    Args:
        func (callable): Function to integrate.
        a (float): Lower limit.
        b (float): Upper limit.
    Returns:
        float: Numerical integral result.
    """
    result, _ = quad(func, a, b)
    return result

# Example Numerical Validation
def func(x):
    return np.sin(x**2) + x**3

num_integral = numerical_integration(func, 0, 2)
print(f"\nNumerical Definite Integral (0 to 2): {num_integral}")

# Solving Symbolic Equations
def solve_equation(equation, variable):
    """
    Solves a given symbolic equation.
    Args:
        equation (sympy.Eq): The equation to solve.
        variable (sympy.Symbol): The variable to solve for.
    Returns:
        list: Solutions of the equation.
    """
    return sp.solve(equation, variable)

# Example: Solving Equations
equation = sp.Eq(x**3 - 3 * x + 2, 0)
solutions = solve_equation(equation, x)
print("\nEquation:")
sp.pprint(equation)
print("\nSolutions:")
print(solutions)

# High-Dimensional Calculus: Vector Fields
def vector_calculus():
    """
    Demonstrates vector calculus operations like divergence and curl.
    """
    x, y, z = sp.symbols('x y z')
    F = sp.Matrix([x**2 * y, y**2 * z, z**2 * x])  # Example vector field

    # Divergence
    div_F = sp.diff(F[0], x) + sp.diff(F[1], y) + sp.diff(F[2], z)
    print("\nDivergence of F:")
    sp.pprint(div_F)

    # Curl
    curl_F = sp.Matrix([
        sp.diff(F[2], y) - sp.diff(F[1], z),
        sp.diff(F[0], z) - sp.diff(F[2], x),
        sp.diff(F[1], x) - sp.diff(F[0], y)
    ])
    print("\nCurl of F:")
    sp.pprint(curl_F)

vector_calculus()

# Visualization of Solutions
def plot_function(func, x_range):
    """
    Plots a given function within a specified range.
    Args:
        func (callable): Function to plot.
        x_range (tuple): Range (min, max).
    """
    x_vals = np.linspace(*x_range, 500)
    y_vals = [func(val) for val in x_vals]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label="f(x)", color='blue')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title("Function Plot")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.show()

# Example Visualization
plot_function(lambda x: np.sin(x**2) + x**3, (-2, 2))
