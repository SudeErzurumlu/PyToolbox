import sympy as sp
import numpy as np

# Define the coordinate system (4D for general relativity)
coords = sp.symbols('x0 x1 x2 x3')
dim = len(coords)

# Define the metric tensor as a symbolic matrix
metric = sp.Matrix([
    [1, 0, 0, 0],
    [0, -sp.Function('f1')(coords[1]), 0, 0],
    [0, 0, -sp.Function('f2')(coords[1]), 0],
    [0, 0, 0, -sp.Function('f3')(coords[1])]
])

# Calculate the inverse metric
inverse_metric = metric.inv()

# Define Christoffel symbols
def christoffel_symbols(metric, inverse_metric, dim):
    """
    Compute the Christoffel symbols of the second kind.
    """
    gamma = [[[0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
    for k in range(dim):
        for i in range(dim):
            for j in range(dim):
                gamma[k][i][j] = 0.5 * sum(
                    inverse_metric[k, l] * (
                        sp.diff(metric[l, j], coords[i]) +
                        sp.diff(metric[l, i], coords[j]) -
                        sp.diff(metric[i, j], coords[l])
                    )
                    for l in range(dim)
                )
    return gamma

gamma = christoffel_symbols(metric, inverse_metric, dim)

# Define the Riemann curvature tensor
def riemann_tensor(gamma, dim):
    """
    Compute the Riemann curvature tensor.
    """
    R = [[[[0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
    for rho in range(dim):
        for sigma in range(dim):
            for mu in range(dim):
                for nu in range(dim):
                    R[rho][sigma][mu][nu] = sp.diff(gamma[rho][mu][nu], coords[sigma]) - sp.diff(gamma[rho][sigma][nu], coords[mu])
                    R[rho][sigma][mu][nu] += sum(
                        gamma[rho][sigma][beta] * gamma[beta][mu][nu] -
                        gamma[rho][mu][beta] * gamma[beta][sigma][nu]
                        for beta in range(dim)
                    )
    return R

R = riemann_tensor(gamma, dim)

# Define the Ricci tensor
def ricci_tensor(R, dim):
    """
    Compute the Ricci tensor by contracting the Riemann tensor.
    """
    Ric = [[0 for _ in range(dim)] for _ in range(dim)]
    for mu in range(dim):
        for nu in range(dim):
            Ric[mu][nu] = sum(R[rho][mu][rho][nu] for rho in range(dim))
    return Ric

Ricci = ricci_tensor(R, dim)

# Define the scalar curvature
def scalar_curvature(Ricci, inverse_metric, dim):
    """
    Compute the scalar curvature by contracting the Ricci tensor with the metric.
    """
    return sum(inverse_metric[mu, nu] * Ricci[mu][nu] for mu in range(dim) for nu in range(dim))

scalar_curv = scalar_curvature(Ricci, inverse_metric, dim)

# Define geodesic equations
def geodesic_equations(gamma, dim):
    """
    Compute the geodesic equations.
    """
    geodesics = [0 for _ in range(dim)]
    for i in range(dim):
        geodesics[i] = sp.Function(f'lambda_{i}')(sp.Symbol('s'))
        geodesic_eq = sp.diff(geodesics[i], sp.Symbol('s'), 2)
        geodesic_eq += sum(
            gamma[i][j][k] * sp.diff(geodesics[j], sp.Symbol('s')) * sp.diff(geodesics[k], sp.Symbol('s'))
            for j in range(dim) for k in range(dim)
        )
        geodesics[i] = sp.Eq(geodesic_eq, 0)
    return geodesics

geodesic_eqs = geodesic_equations(gamma, dim)

# Display results
print("\nChristoffel Symbols:")
for i in range(dim):
    for j in range(dim):
        for k in range(dim):
            if gamma[i][j][k] != 0:
                print(f"Γ^{i}_{j}{k} = {gamma[i][j][k]}")

print("\nRiemann Curvature Tensor:")
for i in range(dim):
    for j in range(dim):
        for k in range(dim):
            for l in range(dim):
                if R[i][j][k][l] != 0:
                    print(f"R^{i}_{j}{k}{l} = {R[i][j][k][l]}")

print("\nRicci Tensor:")
for i in range(dim):
    for j in range(dim):
        if Ricci[i][j] != 0:
            print(f"Ric^{i}{j} = {Ricci[i][j]}")

print(f"\nScalar Curvature: {scalar_curv}")

print("\nGeodesic Equations:")
for eq in geodesic_eqs:
    print(eq)
