import sympy as sp

# Define coordinates
t, r, theta, phi = sp.symbols('t r theta phi')
coords = [t, r, theta, phi]

# Input Metric Tensor (Example: Schwarzschild Metric)
M = sp.Symbol('M')  # Mass parameter
g = sp.Matrix([
    [-(1 - 2*M/r), 0, 0, 0],
    [0, 1/(1 - 2*M/r), 0, 0],
    [0, 0, r**2, 0],
    [0, 0, 0, r**2 * sp.sin(theta)**2]
])

# Inverse Metric
g_inv = g.inv()

# Christoffel Symbols of the Second Kind
def christoffel_symbols(metric, coordinates):
    """
    Compute Christoffel symbols for a given metric tensor.
    Args:
        metric (sympy.Matrix): Metric tensor.
        coordinates (list): List of coordinate symbols.
    Returns:
        list: Christoffel symbols Γ^k_ij.
    """
    n = len(coordinates)
    Γ = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                Γ[k][i][j] = sp.simplify(
                    0.5 * sum(
                        g_inv[k, l] * (sp.diff(metric[l, j], coordinates[i]) +
                                       sp.diff(metric[l, i], coordinates[j]) -
                                       sp.diff(metric[i, j], coordinates[l]))
                        for l in range(n)
                    )
                )
    return Γ

# Compute Christoffel Symbols
Γ = christoffel_symbols(g, coords)
print("\nChristoffel Symbols Γ^k_ij:")
for k in range(4):
    for i in range(4):
        for j in range(4):
            if Γ[k][i][j] != 0:
                print(f"Γ^{k}_{i}{j} = {Γ[k][i][j]}")

# Riemann Tensor
def riemann_tensor(christoffel, coordinates):
    """
    Compute the Riemann curvature tensor.
    Args:
        christoffel (list): Christoffel symbols.
        coordinates (list): List of coordinate symbols.
    Returns:
        list: Riemann curvature tensor R^l_ijk.
    """
    n = len(coordinates)
    R = [[[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for l in range(n):
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    R[l][i][j][k] = sp.simplify(
                        sp.diff(christoffel[l][i][k], coordinates[j]) -
                        sp.diff(christoffel[l][i][j], coordinates[k]) +
                        sum(christoffel[l][m][j] * christoffel[m][i][k] -
                            christoffel[l][m][k] * christoffel[m][i][j]
                            for m in range(n))
                    )
    return R

# Compute Riemann Tensor
R = riemann_tensor(Γ, coords)
print("\nRiemann Curvature Tensor R^l_ijk:")
for l in range(4):
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if R[l][i][j][k] != 0:
                    print(f"R^{l}_{i}{j}{k} = {R[l][i][j][k]}")

# Ricci Tensor
def ricci_tensor(riemann):
    """
    Compute the Ricci tensor by contracting the Riemann tensor.
    Args:
        riemann (list): Riemann curvature tensor.
    Returns:
        sympy.Matrix: Ricci tensor.
    """
    n = len(riemann)
    Ric = sp.Matrix.zeros(n, n)
    for i in range(n):
        for j in range(n):
            Ric[i, j] = sp.simplify(sum(riemann[k][i][k][j] for k in range(n)))
    return Ric

# Compute Ricci Tensor and Ricci Scalar
Ric = ricci_tensor(R)
R_scalar = sp.simplify(sum(Ric[i, i] * g_inv[i, i] for i in range(4)))

print("\nRicci Tensor:")
sp.pprint(Ric)

print("\nRicci Scalar:")
sp.pprint(R_scalar)

# Einstein Tensor
def einstein_tensor(ricci, metric, scalar):
    """
    Compute the Einstein tensor.
    Args:
        ricci (sympy.Matrix): Ricci tensor.
        metric (sympy.Matrix): Metric tensor.
        scalar (sympy.Expr): Ricci scalar.
    Returns:
        sympy.Matrix: Einstein tensor.
    """
    dim = metric.shape[0]
    G = sp.Matrix.zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            G[i, j] = sp.simplify(ricci[i, j] - 0.5 * scalar * metric[i, j])
    return G

# Compute Einstein Tensor
G_tensor = einstein_tensor(Ric, g, R_scalar)
print("\nEinstein Tensor:")
sp.pprint(G_tensor)

# Geodesic Equations
def geodesic_equations(metric, coordinates):
    """
    Compute geodesic equations for a given metric.
    Args:
        metric (sympy.Matrix): Metric tensor.
        coordinates (list): List of coordinate symbols.
    Returns:
        list: Geodesic equations.
    """
    Γ = christoffel_symbols(metric, coordinates)
    geodesics = []
    for i in range(len(coordinates)):
        eq = sp.Add(*[
            Γ[i][j][k] * sp.Symbol(f'd{coordinates[j]}/dλ') * sp.Symbol(f'd{coordinates[k]}/dλ')
            for j in range(len(coordinates)) for k in range(len(coordinates))
        ])
        geodesics.append(sp.Eq(sp.Symbol(f'd²{coordinates[i]}/dλ²'), -eq))
    return geodesics

# Example: Geodesic Equations
geodesics = geodesic_equations(g, coords)
print("\nGeodesic Equations:")
for ge in geodesics:
    sp.pprint(ge)
