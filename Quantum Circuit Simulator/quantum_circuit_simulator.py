from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
from qiskit.providers.aer import AerSimulator

class QuantumSimulator:
    def __init__(self, qubits):
        """
        Initializes a quantum circuit simulator.
        Args:
            qubits (int): Number of qubits in the circuit.
        """
        self.circuit = QuantumCircuit(qubits, qubits)

    def apply_hadamard(self, qubit):
        """
        Applies a Hadamard gate to a qubit.
        """
        self.circuit.h(qubit)

    def apply_cnot(self, control, target):
        """
        Applies a CNOT gate.
        """
        self.circuit.cx(control, target)

    def measure_all(self):
        """
        Measures all qubits.
        """
        self.circuit.measure(range(self.circuit.num_qubits), range(self.circuit.num_qubits))

    def simulate(self):
        """
        Simulates the quantum circuit.
        """
        simulator = Aer.get_backend("aer_simulator")
        compiled_circuit = transpile(self.circuit, simulator)
        result = simulator.run(assemble(compiled_circuit)).result()
        counts = result.get_counts()
        plot_histogram(counts).show()

# Example Usage:
# qsim = QuantumSimulator(2)
# qsim.apply_hadamard(0)
# qsim.apply_cnot(0, 1)
# qsim.measure_all()
# qsim.simulate()
