import numpy as np
import sympy as sp
from sympy.integrals.transforms import inverse_fourier_transform, fourier_transform
import matplotlib.pyplot as plt

# Define Inner Product for Hilbert Space
def inner_product(f, g, domain):
    """
    Computes the inner product of two functions in a Hilbert space.
    Args:
        f (sympy.Function): First function.
        g (sympy.Function): Second function.
        domain (tuple): Domain of integration (start, end).
    Returns:
        sympy.Expr: The inner product <f, g>.
    """
    x = sp.symbols('x')
    product = sp.conjugate(f(x)) * g(x)
    return sp.integrate(product, (x, *domain))

# Example: Compute Inner Product
x = sp.symbols('x')
f = sp.Lambda(x, sp.exp(-x**2))
g = sp.Lambda(x, sp.sin(x))
inner = inner_product(f, g, (-sp.oo, sp.oo))
print("Inner Product <f, g>:")
sp.pprint(inner)

# Operator on Hilbert Space
def apply_operator(operator, function):
    """
    Applies a linear operator to a function.
    Args:
        operator (sympy.Derivative or sympy.Function): Linear operator.
        function (sympy.Function): Function to apply the operator to.
    Returns:
        sympy.Function: Resulting function after applying the operator.
    """
    return operator(function)

# Example: Apply Derivative Operator
D = sp.Derivative
result = apply_operator(D(x), sp.sin(x))
print("\nOperator Applied (Derivative):")
sp.pprint(result)

# Fourier Series Expansion
def fourier_series(f, domain, n_terms=5):
    """
    Computes the Fourier series expansion of a function.
    Args:
        f (sympy.Function): Function to expand.
        domain (tuple): Interval for expansion (-L, L).
        n_terms (int): Number of terms in the Fourier series.
    Returns:
        sympy.Expr: Fourier series expansion.
    """
    x, n = sp.symbols('x n')
    L = (domain[1] - domain[0]) / 2
    a0 = (1 / (2 * L)) * sp.integrate(f(x), (x, *domain))
    an = (1 / L) * sp.integrate(f(x) * sp.cos(n * sp.pi * x / L), (x, *domain))
    bn = (1 / L) * sp.integrate(f(x) * sp.sin(n * sp.pi * x / L), (x, *domain))
    series = a0 + sp.Sum(an * sp.cos(n * sp.pi * x / L) + bn * sp.sin(n * sp.pi * x / L), (n, 1, n_terms))
    return series

# Example: Fourier Series
f = sp.Lambda(x, x**2)
fourier_expansion = fourier_series(f, (-sp.pi, sp.pi), n_terms=5)
print("\nFourier Series Expansion:")
sp.pprint(fourier_expansion)

# Spectral Analysis
def spectral_decomposition(operator, domain):
    """
    Computes the eigenvalues and eigenfunctions of an operator.
    Args:
        operator (sympy.Derivative or sympy.Function): Operator.
        domain (tuple): Domain of definition.
    Returns:
        list: Eigenvalues and corresponding eigenfunctions.
    """
    x, lmbda = sp.symbols('x lambda')
    eigenfunction = sp.Function('eigenfunction')
    eigen_eq = sp.Eq(apply_operator(operator, eigenfunction(x)), lmbda * eigenfunction(x))
    solutions = sp.dsolve(eigen_eq, eigenfunction(x))
    return solutions

# Example: Spectral Decomposition
spectral_result = spectral_decomposition(sp.Derivative(x, x, 2), (-sp.pi, sp.pi))
print("\nSpectral Decomposition:")
sp.pprint(spectral_result)

# Visualization of Norms
def visualize_unit_ball(p_norm, dimension=2):
    """
    Visualizes the unit ball for a given p-norm.
    Args:
        p_norm (float): The p-norm to visualize (e.g., 1, 2, or np.inf).
        dimension (int): Dimension of the space.
    """
    assert dimension == 2, "Visualization supports only 2D."
    theta = np.linspace(0, 2 * np.pi, 1000)
    if p_norm == 1:
        x = np.sign(np.cos(theta)) * np.abs(np.cos(theta))
        y = np.sign(np.sin(theta)) * np.abs(np.sin(theta))
    elif p_norm == 2:
        x = np.cos(theta)
        y = np.sin(theta)
    elif p_norm == np.inf:
        x = np.sign(np.cos(theta))
        y = np.sign(np.sin(theta))
    else:
        x = np.abs(np.cos(theta))**(2/p_norm)
        y = np.abs(np.sin(theta))**(2/p_norm)
    plt.figure()
    plt.plot(x, y, label=f"{p_norm}-Norm Unit Ball")
    plt.title(f"Unit Ball for {p_norm}-Norm")
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.show()

# Example: Visualize Unit Ball
visualize_unit_ball(2)  # Euclidean Norm
