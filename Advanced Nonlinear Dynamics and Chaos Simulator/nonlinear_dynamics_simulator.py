import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq

class NonlinearDynamicsSimulator:
    def __init__(self):
        pass

    # Define the system of ODEs
    def define_system(self, equations, initial_conditions, params):
        """Define a nonlinear system of ODEs."""
        self.equations = equations
        self.initial_conditions = initial_conditions
        self.params = params

    # Solve the system using numerical integration
    def solve_system(self, t_span, t_eval):
        """Solve the system of ODEs over a given time span."""
        def system(t, state):
            return self.equations(state, t, **self.params)

        sol = solve_ivp(system, t_span, self.initial_conditions, t_eval=t_eval, method='RK45')
        return sol

    # Plot phase portrait
    def plot_phase_portrait(self, sol, variables=(0, 1)):
        """Plot phase portrait for two variables."""
        plt.figure()
        plt.plot(sol.y[variables[0]], sol.y[variables[1]], color='blue')
        plt.title(f"Phase Portrait: Var{variables[0]} vs Var{variables[1]}")
        plt.xlabel(f"Variable {variables[0]}")
        plt.ylabel(f"Variable {variables[1]}")
        plt.grid()
        plt.show()

    # Compute Lyapunov Exponents
    def compute_lyapunov(self, system_jacobian, t_span, dt):
        """Estimate the largest Lyapunov exponent using a variational method."""
        state = np.array(self.initial_conditions)
        perturbation = np.random.normal(0, 1e-5, len(state))
        perturbation /= np.linalg.norm(perturbation)
        
        lyapunov_exponent = 0
        for _ in np.arange(t_span[0], t_span[1], dt):
            state_next = self.equations(state, 0, **self.params)
            perturbed_next = self.equations(state + perturbation, 0, **self.params)
            
            delta = perturbed_next - state_next
            delta_norm = np.linalg.norm(delta)
            perturbation = delta / delta_norm  # Normalize perturbation
            lyapunov_exponent += np.log(delta_norm)
            state = state_next
        
        return lyapunov_exponent / (t_span[1] - t_span[0])

    # Generate and plot Poincaré Section
    def poincare_section(self, sol, plane_var_index, plane_value):
        """Extract and plot Poincaré section for a given variable."""
        poincare_points = []
        for i in range(1, len(sol.t)):
            if (sol.y[plane_var_index, i - 1] - plane_value) * (sol.y[plane_var_index, i] - plane_value) <= 0:
                poincare_points.append([sol.y[j, i] for j in range(len(sol.y))])
        poincare_points = np.array(poincare_points)

        plt.figure()
        plt.scatter(poincare_points[:, 0], poincare_points[:, 1], color='red', s=10)
        plt.title("Poincaré Section")
        plt.xlabel("Variable 0")
        plt.ylabel("Variable 1")
        plt.grid()
        plt.show()

    # Fourier Transform Analysis
    def time_series_analysis(self, sol, variable_index):
        """Perform Fourier transform on time series of a variable."""
        time_series = sol.y[variable_index]
        freq = fftfreq(len(time_series), d=sol.t[1] - sol.t[0])
        spectrum = np.abs(fft(time_series))

        plt.figure()
        plt.plot(freq, spectrum, color='purple')
        plt.title(f"Fourier Transform (Variable {variable_index})")
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()


# Example Usage
if __name__ == "__main__":
    simulator = NonlinearDynamicsSimulator()

    # Example System: Lorenz Attractor
    def lorenz(state, t, sigma, rho, beta):
        x, y, z = state
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        return [dx_dt, dy_dt, dz_dt]

    # Parameters and Initial Conditions
    params = {'sigma': 10, 'rho': 28, 'beta': 8 / 3}
    initial_conditions = [1.0, 1.0, 1.0]
    t_span = (0, 50)
    t_eval = np.linspace(*t_span, 10000)

    # Define and solve the system
    simulator.define_system(lorenz, initial_conditions, params)
    solution = simulator.solve_system(t_span, t_eval)

    # Phase Portrait
    simulator.plot_phase_portrait(solution, variables=(0, 1))

    # Lyapunov Exponent
    lyapunov = simulator.compute_lyapunov(lorenz, t_span, dt=0.01)
    print(f"Estimated Largest Lyapunov Exponent: {lyapunov:.4f}")

    # Poincaré Section
    simulator.poincare_section(solution, plane_var_index=2, plane_value=20)

    # Time Series Fourier Analysis
    simulator.time_series_analysis(solution, variable_index=0)
