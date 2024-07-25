import numpy as np
from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import Aer
from qiskit.circuit import ParameterVector
from scipy.optimize import minimize
from qiskit.circuit.library import RealAmplitudes
import matplotlib.pyplot as plt
from time import time
import sys

def print_progress(message):
    sys.stdout.write(f"\r{message}")
    sys.stdout.flush()

def qft_matrix(n):
    omega = np.exp(2 * np.pi * 1j / 2**n)
    return np.array([[omega**(i * j) / np.sqrt(2**n) for j in range(2**n)] for i in range(2**n)])

def iqft_matrix(n):
    return np.conj(qft_matrix(n).T)

def create_ansatz(n, layers):
    ansatz = RealAmplitudes(n, reps=layers)
    return ansatz

def cost_function(params, ansatz, target_state, backend, shots=1024):
    bound_circuit = ansatz.assign_parameters(params)
    bound_circuit.measure_all()
    transpiled_qc = transpile(bound_circuit, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()
    probabilities = {state: count / shots for state, count in counts.items()}
    target_counts = np.abs(target_state)**2
    error = sum((probabilities.get(bin(i)[2:].zfill(ansatz.num_qubits), 0) - target_counts[i])**2 for i in range(2**ansatz.num_qubits))
    return error

def train_ansatz(ansatz, target_state, backend, maxiter=1000):
    initial_params = np.random.random(ansatz.num_parameters) * 2 * np.pi
    result = minimize(cost_function, initial_params, args=(ansatz, target_state, backend),
                      method='COBYLA', options={'maxiter': maxiter})
    return result.x

def get_approximated_state(ansatz, params, input_state, backend, shots=100000):
    qc = QuantumCircuit(ansatz.num_qubits, ansatz.num_qubits)
    qc.initialize(input_state, range(ansatz.num_qubits))
    qc = qc.compose(ansatz)
    bound_circuit = qc.assign_parameters(params)
    bound_circuit.measure_all()
    transpiled_qc = transpile(bound_circuit, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()
    probabilities = np.zeros(2**ansatz.num_qubits, dtype=complex)
    for state, count in counts.items():
        # Remove spaces and reverse the bit string
        state_index = int(state.replace(' ', '')[::-1], 2)
        probabilities[state_index] = np.sqrt(count / shots)
    return probabilities

def evaluate_accuracy_and_matrices(n_values, layers_values):
    backend = Aer.get_backend('qasm_simulator')
    accuracies = np.zeros((len(n_values), len(layers_values)))

    total_iterations = len(n_values) * len(layers_values)
    current_iteration = 0
    
    start_time = time()

    approximated_iqft_matrices = {}
    
    for i, n in enumerate(n_values):
        iqft = iqft_matrix(n)
        for j, layers in enumerate(layers_values):
            current_iteration += 1
            print_progress(f"Progress: {current_iteration}/{total_iterations} " 
                           f"(n={n}, layers={layers})")
            
            try:
                ansatz = create_ansatz(n, layers)
                input_state = np.random.rand(2**n) + 1j * np.random.rand(2**n)
                input_state /= np.linalg.norm(input_state)
                target_state = iqft @ input_state
                opt_params = train_ansatz(ansatz, target_state, backend)
                accuracies[i, j] = 1 - cost_function(opt_params, ansatz, target_state, backend)
                approximated_iqft = np.zeros((2**n, 2**n), dtype=complex)
                for basis_state in range(2**n):
                    input_state = np.zeros(2**n)
                    input_state[basis_state] = 1
                    approximated_iqft[:, basis_state] = get_approximated_state(ansatz, opt_params, input_state, backend)
                
                approximated_iqft_matrices[(n, layers)] = approximated_iqft
            except Exception as e:
                print(f"\nError occurred for n={n}, layers={layers}: {str(e)}")
                accuracies[i, j] = np.nan
                approximated_iqft_matrices[(n, layers)] = None
    
    end_time = time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    
    return accuracies, approximated_iqft_matrices

# Set up parameters
n_values = [1, 2, 3, 4]
layers_values = [1, 2, 3, 4, 5]

# Evaluate accuracy and get approximated states
print("Starting evaluation...")
accuracies, approximated_iqft_matrices = evaluate_accuracy_and_matrices(n_values, layers_values)

# Plot accuracy as a function of the number of qubits
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for j, layers in enumerate(layers_values):
    plt.plot(n_values, accuracies[:, j], marker='o', label=f'{layers} layers')
plt.xlabel('Number of qubits')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Qubits')
plt.legend()

# Plot accuracy as a function of the number of layers
plt.subplot(1, 2, 2)
for i, n in enumerate(n_values):
    plt.plot(layers_values, accuracies[i, :], marker='o', label=f'{n} qubits')
plt.xlabel('Number of layers')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Layers')
plt.legend()

plt.tight_layout()
plt.show()

for n in n_values:
    for j, layers in enumerate(layers_values):
        approx_iqft = approximated_iqft_matrices.get((n, layers))
        if approx_iqft is not None:
            print(f'matrix for IQFT of n={n} and layers={layers} is:\n')
            print(approx_iqft)
        else:
            print(f'matrix for IQFT of n={n} and layers={layers} failed:\n')

print("Execution completed.")