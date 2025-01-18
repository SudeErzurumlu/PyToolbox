import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import odeint

class QuantumMechanicsSolver:
    def __init__(self):
        self.variables = []
        self.hamiltonian = None
        self.wavefunction = None

    def define_variables(self, variables):
        """Define symbolic variables (time and spatial dimensions)."""
        self.variables = sp.symbols(variables)
        return self.variables

    def define_hamiltonian(self, potential_function, mass=1):
        """Define the Hamiltonian operator for the system."""
        x, t = self.variables
        p = -sp.I * sp.diff  # momentum operator in position space
        H = -sp.Rational(1, 2 * mass) * p(x)**2 + potential_function(x)
        self.hamiltonian = H
        return H

    def solve_schrodinger_equation(self, potential_function, mass=1):
        """Solve the time-dependent Schrödinger equation symbolically."""
        x, t = self.variables
        H = self.define_hamiltonian(potential_function, mass)
        psi = sp.Function("psi")(x, t)

        schrodinger_eq = sp.Eq(sp.diff(psi, t), -sp.I * H * psi)
        solution = sp.dsolve(schrodinger_eq, psi)
        self.wavefunction = solution
        return solution

    def time_evolution_operator(self, hamiltonian, time, dt):
        """Calculate the time evolution operator for the system."""
        U = expm(-sp.I * hamiltonian * time)
        return U

    def simulate_time_evolution(self, initial_wavefunction, potential_function, time_range, mass=1):
        """Simulate the time evolution of the quantum state."""
        x, t = self.variables
        H = self.define_hamiltonian(potential_function, mass)
        U = self.time_evolution_operator(H, t, 0.01)

        # Simulate wavefunction over time
        wavefunctions = []
        for time in time_range:
            evolved_wavefunction = U @ initial_wavefunction
            wavefunctions.append(evolved_wavefunction)
        
        return wavefunctions

    def calculate_probability_density(self, wavefunction):
        """Calculate the probability density from the wavefunction."""
        return np.abs(wavefunction)**2

    def plot_wavefunction(self, wavefunction, x_range):
        """Plot the real and imaginary parts of the wavefunction."""
        x_vals = np.linspace(*x_range, 400)
        y_vals_real = np.real(wavefunction(x_vals))
        y_vals_imag = np.imag(wavefunction(x_vals))

        plt.plot(x_vals, y_vals_real, label='Real part')
        plt.plot(x_vals, y_vals_imag, label='Imaginary part')
        plt.title("Wavefunction")
        plt.xlabel("Position (x)")
        plt.ylabel("Wavefunction Amplitude")
        plt.legend()
        plt.show()

    def plot_probability_density(self, wavefunction, x_range):
        """Plot the probability density from the wavefunction."""
        x_vals = np.linspace(*x_range, 400)
        probability_density = self.calculate_probability_density(wavefunction(x_vals))

        plt.plot(x_vals, probability_density)
        plt.title("Probability Density")
        plt.xlabel("Position (x)")
        plt.ylabel("Probability Density")
        plt.show()


# Example Usage
if __name__ == "__main__":
    solver = QuantumMechanicsSolver()

    # Define variables (space and time)
    x, t = solver.define_variables("x t")

    # Define a potential function (e.g., infinite potential well)
    V = sp.Piecewise((0, (x > 0) & (x < 10)), (sp.oo, True))

    # Solve the Schrödinger equation symbolically
    solution = solver.solve_schrodinger_equation(V)
    print("Schrödinger Equation Solution:", solution)

    # Simulate the time evolution of the quantum state
    def initial_wavefunction(x):
        return np.sin(np.pi * x / 10)

    time_range = np.linspace(0, 10, 100)
    evolved_wavefunctions = solver.simulate_time_evolution(
        initial_wavefunction, V, time_range
    )

    # Plot the wavefunction at the initial state
    solver.plot_wavefunction(initial_wavefunction, (0, 10))

    # Plot the probability density of the wavefunction at the initial state
    solver.plot_probability_density(initial_wavefunction, (0, 10))
