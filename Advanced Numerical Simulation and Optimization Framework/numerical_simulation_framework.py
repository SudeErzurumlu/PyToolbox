import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class NumericalSimulationFramework:
    def __init__(self):
        pass

    # Solve PDE using finite difference method
    def solve_pde_fd(self, initial_condition, diffusion_coefficient, dx, dt, T):
        """Solve 1D heat equation using explicit finite difference method."""
        n_points = len(initial_condition)
        u = np.array(initial_condition)
        alpha = diffusion_coefficient * dt / dx**2

        if alpha > 0.5:
            raise ValueError("Stability condition violated: alpha <= 0.5 required.")

        time_steps = int(T / dt)
        results = [u.copy()]
        for t in range(time_steps):
            u_new = u.copy()
            for i in range(1, n_points - 1):
                u_new[i] = u[i] + alpha * (u[i+1] - 2*u[i] + u[i-1])
            u = u_new
            results.append(u.copy())

        return np.array(results)

    # Simulate nonlinear dynamic systems
    def simulate_dynamics(self, f, t_span, y0, t_eval):
        """Simulate a nonlinear dynamic system."""
        solution = solve_ivp(f, t_span, y0, t_eval=t_eval, method='RK45')
        return solution

    # Perform global optimization
    def optimize_function(self, func, bounds, method="L-BFGS-B"):
        """Optimize a multivariable function."""
        result = minimize(func, x0=np.mean(bounds, axis=1), bounds=bounds, method=method)
        return result

    # Plot PDE solution
    def plot_pde_solution(self, solution, dx, dt):
        """Visualize the solution of a PDE."""
        x = np.linspace(0, len(solution[0]) * dx, len(solution[0]))
        t = np.linspace(0, len(solution) * dt, len(solution))
        X, T = np.meshgrid(x, t)

        plt.figure()
        plt.contourf(X, T, solution, levels=50, cmap='viridis')
        plt.colorbar(label="Temperature")
        plt.xlabel("x")
        plt.ylabel("Time")
        plt.title("PDE Solution (1D Heat Equation)")
        plt.show()

    # Plot dynamic system phase portrait
    def plot_phase_portrait(self, solution, variables=(0, 1)):
        """Visualize phase portrait of a dynamic system."""
        x, y = solution.y[variables[0]], solution.y[variables[1]]
        plt.figure()
        plt.plot(x, y)
        plt.xlabel(f"Variable {variables[0]}")
        plt.ylabel(f"Variable {variables[1]}")
        plt.title("Phase Portrait")
        plt.show()

# Example Usage
if __name__ == "__main__":
    framework = NumericalSimulationFramework()

    # Example 1: Solve 1D heat equation using finite difference
    initial_condition = [100 if 0.4 < x < 0.6 else 0 for x in np.linspace(0, 1, 50)]
    diffusion_coefficient = 0.01
    dx, dt = 0.02, 0.005
    T = 1.0

    pde_solution = framework.solve_pde_fd(initial_condition, diffusion_coefficient, dx, dt, T)
    framework.plot_pde_solution(pde_solution, dx, dt)

    # Example 2: Simulate nonlinear dynamic system (Van der Pol oscillator)
    def van_der_pol(t, y, mu=1.0):
        x, v = y
        dxdt = v
        dvdt = mu * (1 - x**2) * v - x
        return [dxdt, dvdt]

    t_span = (0, 20)
    y0 = [2.0, 0.0]
    t_eval = np.linspace(0, 20, 1000)
    solution = framework.simulate_dynamics(lambda t, y: van_der_pol(t, y), t_span, y0, t_eval)
    framework.plot_phase_portrait(solution)

    # Example 3: Global optimization of a multivariable function
    def objective(x):
        return np.sin(x[0]) * np.cos(x[1]) + x[0]**2 + x[1]**2

    bounds = [(-2, 2), (-2, 2)]
    result = framework.optimize_function(objective, bounds)
    print("Optimization Result:", result)
