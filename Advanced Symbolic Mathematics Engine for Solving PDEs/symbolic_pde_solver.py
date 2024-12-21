import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Symbolic Variables
x, y, t, alpha = sp.symbols('x y t alpha')
u = sp.Function('u')(x, y, t)

# Define PDE: Heat Equation
pde = sp.Eq(sp.Derivative(u, t), alpha * (sp.diff(u, x, 2) + sp.diff(u, y, 2)))

# Display PDE Symbolically
print("Partial Differential Equation:")
sp.pprint(pde)

# Boundary and Initial Conditions (Symbolic)
u_initial = sp.Lambda((x, y), sp.sin(sp.pi * x) * sp.sin(sp.pi * y))
u_boundary = 0  # Dirichlet boundary condition

# Discretization Parameters
nx, ny = 50, 50  # Number of grid points
lx, ly = 1.0, 1.0  # Domain size
dx, dy = lx / (nx - 1), ly / (ny - 1)
dt = 0.0005  # Time step
alpha_value = 0.01
nt = 500  # Number of time steps

# Create Grid
x_vals = np.linspace(0, lx, nx)
y_vals = np.linspace(0, ly, ny)
X, Y = np.meshgrid(x_vals, y_vals)

# Initial Condition (Numerical)
u_curr = np.sin(np.pi * X) * np.sin(np.pi * Y)

# Finite Difference Method for Time Evolution
coeff = alpha_value * dt / dx**2
A = diags([-coeff, 1 + 2 * coeff, -coeff], offsets=[-1, 0, 1], shape=(nx, nx)).toarray()

# Time Stepping
for _ in range(nt):
    for j in range(1, ny - 1):
        b = u_curr[1:-1, j]  # Current temperature column
        u_curr[1:-1, j] = spsolve(A, b)  # Solve for the next step

# Visualization
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, u_curr, levels=100, cmap='hot')
plt.colorbar(label='Temperature')
plt.title("2D Heat Equation Solution")
plt.xlabel('x')
plt.ylabel('y')
plt.show()
