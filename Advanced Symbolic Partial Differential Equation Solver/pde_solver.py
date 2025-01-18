import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

class PDESolver:
    def __init__(self):
        self.variables = []
        self.pde = None

    def define_variables(self, variables):
        """Define symbolic variables (time and spatial dimensions)."""
        self.variables = sp.symbols(variables)
        return self.variables

    def define_pde(self, equation):
        """Define the symbolic PDE."""
        self.pde = equation
        return self.pde

    def solve_symbolically(self, initial_condition, boundary_conditions):
        """Solve the PDE symbolically."""
        if not self.pde:
            raise ValueError("Define the PDE before solving.")
        
        x, t = self.variables
        solution = sp.dsolve(self.pde, hint="1st_linear")
        return solution

    def solve_numerically(self, domain, time_steps, dx, dt, initial_condition, boundary_condition):
        """Solve the PDE numerically using finite difference."""
        x_min, x_max = domain
        time_max = time_steps * dt

        # Create spatial and temporal grids
        x = np.arange(x_min, x_max + dx, dx)
        t = np.arange(0, time_max + dt, dt)
        u = np.zeros((len(t), len(x)))

        # Apply initial condition
        u[0, :] = initial_condition(x)

        # Time-stepping loop (example: heat equation)
        for n in range(0, len(t) - 1):
            for i in range(1, len(x) - 1):
                u[n + 1, i] = (
                    u[n, i] 
                    + dt * boundary_condition(x[i]) 
                    + dt * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1]) / dx**2
                )
            # Apply boundary conditions
            u[n + 1, 0] = boundary_condition(x[0])
            u[n + 1, -1] = boundary_condition(x[-1])
        
        return x, t, u

    def visualize_solution(self, x, t, u):
        """Visualize the numerical solution in 3D."""
        X, T = np.meshgrid(x, t)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, T, u, cmap="viridis")
        ax.set_xlabel("Space (x)")
        ax.set_ylabel("Time (t)")
        ax.set_zlabel("Solution (u)")
        ax.set_title("PDE Solution")
        plt.show()


# Example Usage
if __name__ == "__main__":
    solver = PDESolver()

    # Define variables
    x, t = solver.define_variables("x t")

    # Define a PDE (heat equation example)
    u = sp.Function("u")(x, t)
    heat_eq = sp.Eq(u.diff(t), u.diff(x, 2))

    # Solve symbolically
    symbolic_solution = solver.solve_symbolically(heat_eq, None, None)
    print("Symbolic Solution:", symbolic_solution)

    # Solve numerically
    domain = (0, 10)
    time_steps = 100
    dx = 0.1
    dt = 0.01

    def initial_condition(x):
        return np.sin(np.pi * x / 10)

    def boundary_condition(x):
        return 0  # Dirichlet condition (fixed at zero)

    x_vals, t_vals, u_vals = solver.solve_numerically(
        domain, time_steps, dx, dt, initial_condition, boundary_condition
    )

    # Visualize the numerical solution
    solver.visualize_solution(x_vals, t_vals, u_vals)
