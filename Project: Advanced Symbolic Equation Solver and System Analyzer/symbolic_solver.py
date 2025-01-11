import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

class SymbolicSolver:
    def __init__(self):
        pass

    def solve_equation(self, equation, variables):
        """Solve a single symbolic equation."""
        solutions = sp.solve(equation, variables)
        return solutions

    def solve_system(self, equations, variables):
        """Solve a system of equations symbolically."""
        solutions = sp.solve(equations, variables, dict=True)
        return solutions

    def solve_differential(self, equation, dependent_var):
        """Solve a differential equation symbolically."""
        solution = sp.dsolve(equation, dependent_var)
        return solution

    def eigenvalue_analysis(self, matrix):
        """Compute eigenvalues and eigenvectors of a matrix."""
        eigenvalues = matrix.eigenvals()
        eigenvectors = matrix.eigenvects()
        return eigenvalues, eigenvectors

    def numerical_solver(self, system_func, initial_conditions, time_span, step_size=0.01):
        """Solve a system of ODEs numerically using Runge-Kutta method."""
        t0, t1 = time_span
        t_values = np.arange(t0, t1, step_size)
        num_vars = len(initial_conditions)
        solutions = np.zeros((len(t_values), num_vars))
        solutions[0] = initial_conditions

        for i in range(1, len(t_values)):
            t = t_values[i - 1]
            y = solutions[i - 1]
            k1 = step_size * system_func(t, y)
            k2 = step_size * system_func(t + step_size / 2, y + k1 / 2)
            k3 = step_size * system_func(t + step_size / 2, y + k2 / 2)
            k4 = step_size * system_func(t + step_size, y + k3)
            solutions[i] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return t_values, solutions

    def plot_solution(self, t_values, solutions, labels):
        """Plot numerical solutions."""
        for i, label in enumerate(labels):
            plt.plot(t_values, solutions[:, i], label=label)
        plt.title("Numerical Solution")
        plt.xlabel("Time")
        plt.ylabel("Values")
        plt.legend()
        plt.grid(True)
        plt.show()

# Example Usage
if __name__ == "__main__":
    solver = SymbolicSolver()

    # Example 1: Solve a symbolic equation
    x, y = sp.symbols("x y")
    eq1 = sp.Eq(x**2 + y**2, 25)
    eq2 = sp.Eq(x - y, 3)
    solutions = solver.solve_system([eq1, eq2], [x, y])
    print("System Solutions:", solutions)

    # Example 2: Solve a differential equation
    t = sp.symbols("t")
    y = sp.Function("y")
    diff_eq = sp.Eq(y(t).diff(t, t) + 4 * y(t).diff(t) + 4 * y(t), sp.sin(t))
    diff_solution = solver.solve_differential(diff_eq, y(t))
    print("Differential Equation Solution:", diff_solution)

    # Example 3: Eigenvalue analysis
    M = sp.Matrix([[4, 1], [2, 3]])
    eigenvalues, eigenvectors = solver.eigenvalue_analysis(M)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:", eigenvectors)

    # Example 4: Numerical solution of a system of ODEs
    def system_func(t, y):
        dydt = np.array([
            y[1],
            -4 * y[1] - 4 * y[0] + np.sin(t)
        ])
        return dydt

    initial_conditions = [0, 0]  # Initial y and y'
    t_span = (0, 10)
    t_vals, num_solutions = solver.numerical_solver(system_func, initial_conditions, t_span)
    solver.plot_solution(t_vals, num_solutions, labels=["y(t)", "y'(t)"])
