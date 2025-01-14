import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class SymbolicSolver:
    def __init__(self):
        self.symbols = {}
        self.equations = []

    def define_symbols(self, *symbols):
        """Define symbols for variables and parameters."""
        for symbol in symbols:
            self.symbols[symbol] = sp.Symbol(symbol)
        return list(self.symbols.values())

    def add_equation(self, equation):
        """Add an equation to the system."""
        self.equations.append(equation)

    def solve_symbolic(self):
        """Solve the system of equations symbolically."""
        if not self.equations:
            raise ValueError("No equations added to the system.")
        solutions = sp.solve(self.equations, list(self.symbols.values()))
        return solutions

    def solve_numeric(self, initial_values, t_span, t_eval, dynamics):
        """Solve dynamic systems numerically using Runge-Kutta."""
        solution = solve_ivp(dynamics, t_span, initial_values, t_eval=t_eval, method='RK45')
        return solution

    def analyze_stability(self, jacobian, equilibrium_point):
        """Perform stability analysis around an equilibrium point."""
        eigenvalues = jacobian.subs(equilibrium_point).eigenvals()
        stability = "Stable" if all(sp.re(ev) < 0 for ev in eigenvalues) else "Unstable"
        return eigenvalues, stability

    def visualize_dynamics(self, time, states, labels=None):
        """Visualize system dynamics over time."""
        for i, state in enumerate(states):
            plt.plot(time, state, label=f"State {i + 1}" if not labels else labels[i])
        plt.title("Dynamic System Trajectories")
        plt.xlabel("Time")
        plt.ylabel("State Values")
        plt.legend()
        plt.grid(True)
        plt.show()


# Example Usage
if __name__ == "__main__":
    # Create an instance of SymbolicSolver
    solver = SymbolicSolver()

    # Define symbols for a two-variable system
    x, y = solver.define_symbols('x', 'y')

    # Define a symbolic system of equations (e.g., nonlinear equations)
    equation1 = sp.Eq(x**2 + y**2 - 4, 0)  # Circle equation
    equation2 = sp.Eq(2 * x - y - 1, 0)    # Line equation
    solver.add_equation(equation1)
    solver.add_equation(equation2)

    # Solve the system symbolically
    symbolic_solutions = solver.solve_symbolic()
    print("Symbolic Solutions:", symbolic_solutions)

    # Define a dynamic system using differential equations
    def dynamics(t, z):
        dxdt = z[1]
        dydt = -0.5 * z[1] - 9.8  # Simple pendulum example
        return [dxdt, dydt]

    # Solve the dynamic system numerically
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 100)
    initial_conditions = [0, 1]  # Initial angle and angular velocity
    numerical_solution = solver.solve_numeric(initial_conditions, t_span, t_eval, dynamics)

    # Visualize the dynamic system's trajectory
    solver.visualize_dynamics(numerical_solution.t, numerical_solution.y, labels=["Angle", "Angular Velocity"])

    # Perform stability analysis (Jacobian and eigenvalues)
    jacobian = sp.Matrix([[0, 1], [-9.8, -0.5]])  # Linearized pendulum system
    equilibrium_point = {x: 0, y: 0}
    eigenvalues, stability = solver.analyze_stability(jacobian, equilibrium_point)
    print("Eigenvalues:", eigenvalues)
    print("Stability:", stability)
