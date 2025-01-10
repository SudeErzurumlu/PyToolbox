import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

class NonlinearOptimizer:
    def __init__(self, variables):
        self.variables = variables

    def gradient(self, objective):
        """Compute the gradient of the objective function."""
        return [sp.diff(objective, var) for var in self.variables]

    def hessian(self, objective):
        """Compute the Hessian matrix of the objective function."""
        return sp.hessian(objective, self.variables)

    def gradient_descent(self, objective, initial_point, learning_rate=0.01, max_iters=1000, tol=1e-6):
        """Perform gradient descent optimization."""
        grad = self.gradient(objective)
        point = np.array(initial_point, dtype=float)
        grad_funcs = [sp.lambdify(self.variables, g) for g in grad]

        trajectory = [point]
        for _ in range(max_iters):
            grad_vals = np.array([g(*point) for g in grad_funcs])
            new_point = point - learning_rate * grad_vals
            trajectory.append(new_point)

            if np.linalg.norm(new_point - point) < tol:
                break
            point = new_point

        return new_point, trajectory

    def lagrange_method(self, objective, constraints):
        """Solve constrained optimization using Lagrange multipliers."""
        lagrange_vars = [sp.symbols(f"lambda_{i}") for i in range(len(constraints))]
        lagrange_expr = objective + sum(l * c for l, c in zip(lagrange_vars, constraints))

        all_vars = self.variables + lagrange_vars
        lagrange_grad = [sp.diff(lagrange_expr, v) for v in all_vars]
        solutions = sp.solve(lagrange_grad, all_vars, dict=True)
        return solutions

    def visualize_trajectory(self, objective, trajectory):
        """Visualize the optimization trajectory for 2D functions."""
        if len(self.variables) != 2:
            raise ValueError("Visualization is only supported for 2D functions.")

        x_vals, y_vals = np.linspace(-10, 10, 400), np.linspace(-10, 10, 400)
        X, Y = np.meshgrid(x_vals, y_vals)
        func = sp.lambdify(self.variables, objective, modules="numpy")
        Z = func(X, Y)

        plt.contour(X, Y, Z, levels=50, cmap="viridis")
        trajectory = np.array(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], marker="o", color="red", label="Trajectory")
        plt.title("Optimization Trajectory")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()

# Example Usage
if __name__ == "__main__":
    x, y = sp.symbols("x y")
    optimizer = NonlinearOptimizer([x, y])

    # Example 1: Unconstrained Optimization
    objective = x**2 + y**2 + 2*x*y - 4*x + 6*y + 5
    initial_point = [5, 5]
    solution, trajectory = optimizer.gradient_descent(objective, initial_point, learning_rate=0.1)
    print("Unconstrained Optimization Solution:", solution)
    optimizer.visualize_trajectory(objective, trajectory)

    # Example 2: Constrained Optimization
    constraint1 = x + y - 5
    constraint2 = x**2 + y**2 - 25
    lagrange_solution = optimizer.lagrange_method(objective, [constraint1, constraint2])
    print("Constrained Optimization Solution (Lagrange Multipliers):", lagrange_solution)
