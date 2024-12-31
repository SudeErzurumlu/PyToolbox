import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define Lie Algebra Generators
def lie_bracket(X, Y, variables):
    """
    Computes the Lie bracket [X, Y] for two vector fields X and Y.
    Args:
        X (sympy.Matrix): First vector field (generator).
        Y (sympy.Matrix): Second vector field (generator).
        variables (list): List of variables.
    Returns:
        sympy.Matrix: The Lie bracket [X, Y].
    """
    dim = len(variables)
    bracket = sp.zeros(dim, 1)
    for i in range(dim):
        for j in range(dim):
            bracket[i] += X[j] * Y[i].diff(variables[j]) - Y[j] * X[i].diff(variables[j])
    return bracket

# Example Lie Algebra Generators
x, y, z = sp.symbols('x y z')
X1 = sp.Matrix([x, -y, 0])
X2 = sp.Matrix([y, x, z])
variables = [x, y, z]

bracket_result = lie_bracket(X1, X2, variables)
print("Lie Bracket [X1, X2]:")
sp.pprint(bracket_result)

# Structure Constants
def compute_structure_constants(generators, variables):
    """
    Computes the structure constants of a Lie algebra.
    Args:
        generators (list): List of Lie algebra generators.
        variables (list): List of variables.
    Returns:
        sympy.MutableDenseNDimArray: Structure constants tensor.
    """
    dim = len(generators)
    structure_constants = sp.MutableDenseNDimArray.zeros(dim, dim, dim)
    for i, X in enumerate(generators):
        for j, Y in enumerate(generators):
            bracket = lie_bracket(X, Y, variables)
            for k, Z in enumerate(generators):
                coeff = sp.simplify(bracket.dot(Z))
                structure_constants[i, j, k] = coeff
    return structure_constants

# Example: Compute Structure Constants
generators = [X1, X2, sp.Matrix([0, 0, z])]
structure_constants = compute_structure_constants(generators, variables)
print("\nStructure Constants:")
sp.pprint(structure_constants)

# Adjoint Representation
def adjoint_representation(generator, basis, variables):
    """
    Computes the adjoint representation of a Lie algebra.
    Args:
        generator (sympy.Matrix): A single generator.
        basis (list): List of basis generators.
        variables (list): List of variables.
    Returns:
        sympy.Matrix: Adjoint matrix representation.
    """
    dim = len(basis)
    adj_matrix = sp.zeros(dim, dim)
    for i, B in enumerate(basis):
        adj_B = lie_bracket(generator, B, variables)
        for j, C in enumerate(basis):
            coeff = sp.simplify(adj_B.dot(C))
            adj_matrix[j, i] = coeff
    return adj_matrix

# Example: Compute Adjoint Matrix
adj_matrix = adjoint_representation(X1, generators, variables)
print("\nAdjoint Representation of X1:")
sp.pprint(adj_matrix)

# Visualization of Lie Group Orbits
def visualize_orbit(generator_func, t_range, initial_point):
    """
    Visualizes the orbit of a Lie group action.
    Args:
        generator_func (callable): Function that generates the vector field.
        t_range (tuple): Range of the parameter t (min, max).
        initial_point (list): Initial point in space.
    """
    t_vals = np.linspace(*t_range, 500)
    orbit = np.zeros((len(t_vals), len(initial_point)))

    for i, t in enumerate(t_vals):
        orbit[i] = generator_func(t, initial_point)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], label="Orbit")
    ax.scatter(*initial_point, color='red', label="Initial Point")
    ax.legend()
    ax.set_title("Lie Group Orbit")
    plt.show()

# Example Visualization
# Replace this with a concrete generator function for your Lie group
def example_generator(t, point):
    x, y, z = point
    return [x * np.cos(t) - y * np.sin(t), x * np.sin(t) + y * np.cos(t), z]

visualize_orbit(example_generator, (0, 2 * np.pi), [1, 0, 0])
