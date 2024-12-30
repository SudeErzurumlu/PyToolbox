import sympy as sp
import numpy as np
from sympy.tensor.array import derive_by_array, tensorcontraction, tensorproduct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Tensor Construction
def create_metric_tensor(dimensions, variables):
    """
    Creates a metric tensor for a given dimensional space.
    Args:
        dimensions (int): Number of dimensions (e.g., 2 for 2D).
        variables (list): Variables representing the space.
    Returns:
        sympy.Matrix: Metric tensor.
    """
    assert dimensions == len(variables), "Dimensions must match the number of variables."
    return sp.Matrix([[sp.Function(f"g_{i}{j}")(*variables) for j in range(dimensions)] for i in range(dimensions)])

# Example: 2D Metric Tensor
x, y = sp.symbols('x y')
metric_tensor = create_metric_tensor(2, [x, y])
print("Metric Tensor:")
sp.pprint(metric_tensor)

# Christoffel Symbols
def christoffel_symbols(metric, variables):
    """
    Computes Christoffel symbols for a given metric tensor.
    Args:
        metric (sympy.Matrix): Metric tensor.
        variables (list): Variables representing the space.
    Returns:
        list: Christoffel symbols (rank-3 tensor).
    """
    dim = len(variables)
    christoffels = [[[0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
    metric_inverse = metric.inv()

    for k in range(dim):
        for i in range(dim):
            for j in range(dim):
                term = 0
                for l in range(dim):
                    term += metric_inverse[k, l] * (
                        sp.diff(metric[l, j], variables[i])
                        + sp.diff(metric[l, i], variables[j])
                        - sp.diff(metric[i, j], variables[l])
                    ) / 2
                christoffels[k][i][j] = term

    return sp.MutableDenseNDimArray(christoffels)

# Example: Compute Christoffel Symbols
christoffel = christoffel_symbols(metric_tensor, [x, y])
print("\nChristoffel Symbols (2D):")
sp.pprint(christoffel)

# Geodesic Equations
def geodesic_equations(christoffels, variables):
    """
    Generates geodesic equations for a given Christoffel tensor.
    Args:
        christoffels (list): Christoffel symbols.
        variables (list): Variables representing the space.
    Returns:
        list: Geodesic equations.
    """
    dim = len(variables)
    t = sp.symbols('t')
    gamma = [sp.Function(f"gamma_{i}")(t) for i in range(dim)]
    eqs = []

    for i in range(dim):
        d2gamma = sp.diff(gamma[i], t, t)
        term = 0
        for j in range(dim):
            for k in range(dim):
                term += christoffels[i][j][k] * sp.diff(gamma[j], t) * sp.diff(gamma[k], t)
        eqs.append(sp.Eq(d2gamma, -term))

    return eqs

# Example: Generate Geodesic Equations
geodesics = geodesic_equations(christoffel, [x, y])
print("\nGeodesic Equations:")
for eq in geodesics:
    sp.pprint(eq)

# Tensor Contraction and Product
def tensor_operations_example():
    """
    Demonstrates tensor contraction and tensor product operations.
    """
    A = sp.MutableDenseNDimArray(range(1, 9), (2, 2, 2))  # Rank-3 tensor
    B = sp.MutableDenseNDimArray(range(1, 5), (2, 2))     # Rank-2 tensor

    # Tensor Product
    product = tensorproduct(A, B)

    # Tensor Contraction
    contraction = tensorcontraction(product, (0, 3))  # Contract indices 0 and 3

    print("\nTensor Product Result:")
    sp.pprint(product)
    print("\nTensor Contraction Result:")
    sp.pprint(contraction)

tensor_operations_example()

# Visualization of Geodesics
def plot_geodesic_2d(solution, t_range):
    """
    Plots a geodesic in 2D space based on its parametric solution.
    Args:
        solution (list): Parametric solution of the geodesic (x(t), y(t)).
        t_range (tuple): Range of the parameter t (min, max).
    """
    t_vals = np.linspace(*t_range, 500)
    x_vals = [solution[0].subs(sp.symbols('t'), t).evalf() for t in t_vals]
    y_vals = [solution[1].subs(sp.symbols('t'), t).evalf() for t in t_vals]

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label="Geodesic", color='green')
    plt.title("Geodesic in 2D Space")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()

# Example Visualization
# (You can replace the parametric solution with computed geodesics)
param_solution = [sp.sin(sp.symbols('t')), sp.cos(sp.symbols('t'))]
plot_geodesic_2d(param_solution, (0, 2 * np.pi))
