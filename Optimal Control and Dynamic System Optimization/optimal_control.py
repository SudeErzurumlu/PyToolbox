import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sympy import symbols, Matrix, exp, diff

class OptimalControl:
    def __init__(self, dynamics, cost_function, initial_state, final_state, time_span):
        self.dynamics = dynamics
        self.cost_function = cost_function
        self.initial_state = initial_state
        self.final_state = final_state
        self.time_span = time_span

    def solve_using_pmp(self):
        """Solve optimal control problem using Pontryagin’s Maximum Principle."""
        # Define state and control symbols
        t, x1, x2, u = symbols('t x1 x2 u')
        state = Matrix([x1, x2])
        
        # Define the Hamiltonian: H = L(x, u) + λ * f(x, u)
        L = self.cost_function(state, u)
        f = self.dynamics(state, u)
        lamda = Matrix([symbols(f"λ{i}") for i in range(len(state))])
        H = L + lamda.T * f
        
        # Compute the optimal control using PMP
        H_u = diff(H, u)
        optimal_control = H_u.subs(u, 0)  # Simple case for linear system
        return optimal_control

    def solve_using_bellman(self):
        """Solve the optimal control problem using Bellman’s Dynamic Programming."""
        def dp(t, x):
            return self.cost_function(x, 0) + np.sum(np.abs(self.dynamics(x, 0))**2)
        
        solution = minimize(dp, self.initial_state)
        return solution.x

    def simulate_dynamic_system(self, control_input):
        """Simulate the system dynamics using the optimal control input."""
        def state_derivatives(t, state):
            return self.dynamics(state, control_input)

        result = solve_ivp(state_derivatives, self.time_span, self.initial_state)
        return result.t, result.y

    def visualize_results(self, time, state_trajectory, control_trajectory):
        """Visualize the results of the simulation."""
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

        ax[0].plot(time, state_trajectory[0], label="State x1")
        ax[0].plot(time, state_trajectory[1], label="State x2")
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('States')
        ax[0].legend()

        ax[1].plot(time, control_trajectory, label="Control input (u)")
        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Control Input')
        ax[1].legend()

        plt.tight_layout()
        plt.show()


# Example Usage
if __name__ == "__main__":
    # Define the system dynamics (e.g., a simple double integrator)
    def dynamics(state, control):
        x1, x2 = state
        dx1dt = x2
        dx2dt = control  # The control input directly influences acceleration
        return [dx1dt, dx2dt]

    # Define the cost function (e.g., minimize state and control cost)
    def cost_function(state, control):
        x1, x2 = state
        return 0.5 * (x1**2 + x2**2) + 0.1 * control**2  # LQR-like cost

    # Define initial and final states and the time span
    initial_state = [1, 0]
    final_state = [0, 0]
    time_span = (0, 10)

    # Create an OptimalControl problem
    optimal_control_system = OptimalControl(dynamics, cost_function, initial_state, final_state, time_span)

    # Solve the problem using Pontryagin’s Maximum Principle (PMP)
    optimal_control = optimal_control_system.solve_using_pmp()
    print("Optimal Control using PMP:", optimal_control)

    # Solve using Bellman’s Dynamic Programming
    optimal_control_bellman = optimal_control_system.solve_using_bellman()
    print("Optimal Control using Bellman’s DP:", optimal_control_bellman)

    # Simulate the dynamic system with the optimal control input
    time, state_trajectory = optimal_control_system.simulate_dynamic_system(optimal_control_bellman)
    control_trajectory = np.full_like(time, optimal_control_bellman)

    # Visualize the system behavior
    optimal_control_system.visualize_results(time, state_trajectory, control_trajectory)
