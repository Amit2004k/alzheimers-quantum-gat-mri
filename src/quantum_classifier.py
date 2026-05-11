"""
quantum_classifier.py
---------------------
Quantum-Inspired Variational Circuit (VQC) classifier using PennyLane.
Used as a hybrid classical-quantum classifier on top of GAT embeddings.
"""

import numpy as np
import torch
import torch.nn as nn

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("PennyLane not found. Install: pip install pennylane")


class QuantumLayer(nn.Module):
    """
    A PennyLane variational quantum circuit wrapped as a PyTorch nn.Module.
    Uses AngleEmbedding + StronglyEntanglingLayers.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (= input feature dimension to the circuit)
    n_layers : int
        Number of variational entangling layers
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        if PENNYLANE_AVAILABLE:
            self.dev = qml.device("default.qubit", wires=n_qubits)
            weight_shapes = {"weights": (n_layers, n_qubits, 3)}

            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def circuit(inputs, weights):
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        else:
            # Fallback: classical linear layer with same I/O shape
            self.qlayer = nn.Linear(n_qubits, n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch, n_qubits)
            Input features normalized to [-π, π]

        Returns
        -------
        torch.Tensor of shape (batch, n_qubits)
            Expectation values of PauliZ on each qubit
        """
        # Normalize to [-pi, pi] for angle embedding
        x = torch.tanh(x) * np.pi
        return self.qlayer(x)


class HybridQuantumClassifier(nn.Module):
    """
    Full hybrid classifier: linear projection → VQC → linear head.

    Parameters
    ----------
    input_dim : int
        Dimension of input GAT embeddings
    n_qubits : int
        Number of qubits in the VQC
    n_layers : int
        Variational layers
    n_classes : int
        Output classes (3 for CN/MCI/AD)
    """

    def __init__(self, input_dim: int = 64, n_qubits: int = 4,
                 n_layers: int = 3, n_classes: int = 3):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(input_dim, n_qubits),
            nn.Tanh(),
        )
        self.vqc = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers)
        self.post = nn.Linear(n_qubits, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = self.vqc(x)
        return self.post(x)
