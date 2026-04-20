"""
Reservoir Computing Module
==========================

Implements reservoir computing paradigms including:
- Echo State Networks (ESN)
- Liquid State Machines (LSM)

Reservoir computing leverages the dynamic properties of a fixed 
recurrent network (the reservoir) for computation.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
import numpy as np

from cogniton.core.config import ReservoirConfig
from cogniton.neurons.lif import LIFNeuronGroup, LIFConfig


@dataclass
class ReservoirConfig:
    """
    Reservoir computing configuration.
    
    Attributes:
        num_reservoir: Number of reservoir neurons
        num_inputs: Number of input dimensions
        num_outputs: Number of output dimensions
        spectral_radius: Spectral radius of internal weights
        input_scaling: Input connection scaling
        connectivity: Internal connectivity probability
        leak_rate: Membrane leak rate (ESN)
        regularization: Regularization for readout training
    """
    num_reservoir: int = 400
    num_inputs: int = 10
    num_outputs: int = 5
    spectral_radius: float = 0.9
    input_scaling: float = 1.0
    connectivity: float = 0.1
    leak_rate: float = 0.3
    regularization: float = 1e-6
    
    def __post_init__(self):
        """Validate configuration."""
        if self.num_reservoir < 1:
            raise ValueError("num_reservoir must be positive")
        if not 0 < self.spectral_radius <= 2.0:
            raise ValueError("spectral_radius must be in (0, 2]")


class ReservoirComputing:
    """
    Base class for reservoir computing models.
    
    Provides common functionality for ESN and LSM implementations.
    """
    
    def __init__(self, config: Optional[ReservoirConfig] = None):
        """
        Initialize reservoir computer.
        
        Args:
            config: Reservoir configuration
        """
        self.config = config or ReservoirConfig()
        
        # State
        self.reservoir_state: np.ndarray = np.zeros(self.config.num_reservoir)
        self.input_state: np.ndarray = np.zeros(self.config.num_inputs)
        self.output_state: np.ndarray = np.zeros(self.config.num_outputs)
        
        # Weights
        self.W_in: Optional[np.ndarray] = None
        self.W_res: Optional[np.ndarray] = None
        self.W_out: Optional[np.ndarray] = None
        
        # Training data storage
        self.states_collected: List[np.ndarray] = []
        self.outputs_collected: List[np.ndarray] = []
        
        # Initialize
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize weight matrices."""
        raise NotImplementedError
    
    def _spectral_radius_normalization(self, W: np.ndarray) -> np.ndarray:
        """
        Normalize matrix to have given spectral radius.
        
        Args:
            W: Weight matrix
            
        Returns:
            Normalized matrix
        """
        eigenvalues = np.linalg.eigvals(W)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        
        if max_eigenvalue > 0:
            return W * (self.config.spectral_radius / max_eigenvalue)
        return W
    
    def set_input(self, input_vector: np.ndarray) -> None:
        """
        Set input state.
        
        Args:
            input_vector: Input vector
        """
        if len(input_vector) != self.config.num_inputs:
            raise ValueError(f"Input size {len(input_vector)} != {self.config.num_inputs}")
        self.input_state = np.asarray(input_vector)
    
    def get_reservoir_state(self) -> np.ndarray:
        """Get current reservoir state."""
        return self.reservoir_state.copy()
    
    def get_output(self) -> np.ndarray:
        """Get current output."""
        return self.output_state.copy()
    
    def reset_state(self) -> None:
        """Reset reservoir and output states."""
        self.reservoir_state = np.zeros(self.config.num_reservoir)
        self.output_state = np.zeros(self.config.num_outputs)
    
    def clear_collected_states(self) -> None:
        """Clear collected states for training."""
        self.states_collected.clear()
        self.outputs_collected.clear()


class EchoStateNetwork(ReservoirComputing):
    """
    Echo State Network (ESN) implementation.
    
    ESN uses a large, randomly connected recurrent network (reservoir)
    with linear readout trained via regression.
    
    Mathematical Model:
    -------------------
    x(t+1) = (1 - α) * x(t) + α * tanh(W_res * x(t) + W_in * u(t))
    y(t) = W_out * [x(t); u(t)]
    
    Where:
    - x: Reservoir state
    - u: Input
    - y: Output
    - α: Leak rate
    - W_res: Reservoir weight matrix
    - W_in: Input weight matrix
    - W_out: Output weight matrix (trained)
    
    References:
    -----------
    Jaeger, H. (2001). The "echo state" approach to analysing and 
        training recurrent neural networks.
    """
    
    def __init__(self, config: Optional[ReservoirConfig] = None,
                 seed: Optional[int] = None):
        """
        Initialize ESN.
        
        Args:
            config: ESN configuration
            seed: Random seed
        """
        self.rng = np.random.RandomState(seed)
        super().__init__(config)
    
    def _initialize_weights(self) -> None:
        """Initialize ESN weight matrices."""
        n = self.config.num_reservoir
        n_in = self.config.num_inputs
        
        # Input weights
        self.W_in = self.rng.randn(n, n_in) * self.config.input_scaling
        
        # Reservoir weights
        W_res = self.rng.randn(n, n) * self.config.connectivity
        
        # Make sparse
        mask = self.rng.rand(n, n) < self.config.connectivity
        W_res = W_res * mask
        
        # Normalize spectral radius
        self.W_res = self._spectral_radius_normalization(W_res)
        
        # Output weights (initialized to zeros, trained later)
        self.W_out = np.zeros((self.config.num_outputs, 
                              n + n_in))  # Includes input
    
    def update(self, input_vector: Optional[np.ndarray] = None,
               store_state: bool = True) -> np.ndarray:
        """
        Update reservoir state.
        
        Args:
            input_vector: Input vector (uses current if None)
            store_state: Whether to store for training
            
        Returns:
            New reservoir state
        """
        if input_vector is not None:
            self.input_state = np.asarray(input_vector)
        
        # Compute reservoir input
        u = self.input_state
        x = self.reservoir_state
        
        # Update reservoir (leaky integrator)
        x_new = (1 - self.config.leak_rate) * x + \
                self.config.leak_rate * np.tanh(
                    self.W_res @ x + self.W_in @ u
                )
        
        self.reservoir_state = x_new
        
        # Compute output
        if self.W_out is not None:
            # Concatenate reservoir and input states
            state_extended = np.concatenate([x_new, u])
            self.output_state = self.W_out @ state_extended
        
        # Store state for training
        if store_state:
            self.states_collected.append(x_new.copy())
            self.outputs_collected.append(self.input_state.copy())
        
        return x_new
    
    def run(self, input_sequence: np.ndarray,
           burn_in: int = 50) -> np.ndarray:
        """
        Run ESN on input sequence.
        
        Args:
            input_sequence: Input time series (time_steps x input_dim)
            burn_in: Number of initial steps to discard
            
        Returns:
            Output time series
        """
        n_steps = len(input_sequence)
        outputs = np.zeros((n_steps, self.config.num_outputs))
        
        self.reset_state()
        
        for t in range(n_steps):
            self.update(input_sequence[t], store_state=True)
            outputs[t] = self.output_state
        
        # Discard burn-in period
        if burn_in > 0 and len(self.states_collected) > burn_in:
            self.states_collected = self.states_collected[burn_in:]
            self.outputs_collected = self.outputs_collected[burn_in:]
        
        return outputs
    
    def train_readout(self, target_outputs: np.ndarray,
                     method: str = "ridge") -> None:
        """
        Train output weights using collected states.
        
        Args:
            target_outputs: Desired outputs (time_steps x output_dim)
            method: Training method ('ridge', 'lstsq', 'pseudo_inv')
        """
        if not self.states_collected:
            raise ValueError("No states collected. Run update() first.")
        
        # Build state matrix (include input)
        n_samples = len(self.states_collected)
        X = np.zeros((n_samples, self.config.num_reservoir + self.config.num_inputs))
        
        for i, (state, inp) in enumerate(zip(self.states_collected, 
                                             self.outputs_collected)):
            X[i, :self.config.num_reservoir] = state
            X[i, self.config.num_reservoir:] = inp
        
        Y = target_outputs
        
        # Train using selected method
        if method == "ridge":
            # Ridge regression: W_out = Y * X^T * (X * X^T + λI)^-1
            lambda_reg = self.config.regularization
            self.W_out = Y.T @ X @ np.linalg.inv(X.T @ X + lambda_reg * np.eye(X.shape[1]))
        
        elif method == "lstsq":
            self.W_out, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            self.W_out = self.W_out.T
        
        elif method == "pseudo_inv":
            self.W_out = Y.T @ np.linalg.pinv(X)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def predict(self, input_sequence: np.ndarray) -> np.ndarray:
        """
        Predict outputs for input sequence.
        
        Args:
            input_sequence: Input time series
            
        Returns:
            Predicted outputs
        """
        self.reset_state()
        return self.run(input_sequence)
    
    def __repr__(self) -> str:
        return (f"EchoStateNetwork(N={self.config.num_reservoir}, "
                f"in={self.config.num_inputs}, "
                f"out={self.config.num_outputs})")


class LiquidStateMachine(ReservoirComputing):
    """
    Liquid State Machine (LSM) implementation.
    
    LSM uses a spiking neural network (liquid) for computation.
    The liquid is typically modeled with LIF neurons and STDP.
    
    Mathematical Model:
    -------------------
    τ * dx_i/dt = -x_i + Σ_j W_ij * y_j + Σ_k W_ik^in * u_k
    
    where x is membrane potential, y is output spike train,
    and W are synaptic weights.
    
    Properties:
    - Separation: Similar inputs -> similar states
    - Convergence: Different inputs -> distinct states
    - Memory: Temporal information preserved
    
    References:
    -----------
    Maass, W., Natschlager, T., & Markram, H. (2002). Real-time computing 
        without stable states: A new framework for neural computation.
    """
    
    def __init__(self, config: Optional[ReservoirConfig] = None,
                 lif_config: Optional["LIFConfig"] = None,
                 seed: Optional[int] = None):
        """
        Initialize LSM.
        
        Args:
            config: LSM configuration
            lif_config: LIF neuron parameters
            seed: Random seed
        """
        self.rng = np.random.RandomState(seed)
        self.lif_config = lif_config or LIFConfig()
        
        super().__init__(config)
        
        # Create liquid (spiking network)
        self.liquid = LIFNeuronGroup(
            group_id=0,
            n=self.config.num_reservoir,
            config=self.lif_config
        )
        
        # Input synapses to liquid
        self.input_synapses: List[Tuple[int, float]] = []  # (neuron_idx, weight)
        
        # Initialize
        self._initialize_liquid()
    
    def _initialize_liquid(self) -> None:
        """Initialize liquid connectivity."""
        n = self.config.num_reservoir
        n_in = self.config.num_inputs
        
        # Random input connections
        for _ in range(int(n * 0.1)):  # 10% connectivity
            neuron_idx = self.rng.randint(0, n)
            weight = self.rng.randn() * self.config.input_scaling
            self.input_synapses.append((neuron_idx, weight))
        
        # Initialize reservoir weights
        W_res = self.rng.randn(n, n) * self.config.connectivity
        
        # Make sparse
        mask = self.rng.rand(n, n) < self.config.connectivity
        W_res = W_res * mask
        
        # Store as dense for now (could use sparse matrix)
        self.W_res = self._spectral_radius_normalization(W_res)
        
        # Output weights
        self.W_out = np.zeros((self.config.num_outputs, n + n_in))
    
    def inject_input(self, input_vector: np.ndarray, 
                    current: float = 1.0) -> None:
        """
        Inject current based on input vector.
        
        Args:
            input_vector: Input values
            current: Scaling factor
        """
        self.input_state = np.asarray(input_vector)
        
        for neuron_idx, weight in self.input_synapses:
            self.liquid.neurons[neuron_idx].inject_current(
                np.dot(input_vector, input_vector) * weight * current
            )
    
    def update(self, input_vector: Optional[np.ndarray] = None,
               dt: float = 1e-4,
               store_state: bool = True) -> np.ndarray:
        """
        Update liquid state.
        
        Args:
            input_vector: Input vector
            dt: Integration timestep
            store_state: Store state for training
            
        Returns:
            New reservoir state
        """
        t = len(self.states_collected) * dt
        
        # Inject input
        if input_vector is not None:
            self.inject_input(input_vector)
        
        # Update liquid
        spiked = self.liquid.update(t, dt)
        
        # Get reservoir state from firing rates
        self.reservoir_state = self.liquid.get_firing_rates()
        
        # Compute output
        if self.W_out is not None:
            state_extended = np.concatenate([self.reservoir_state, self.input_state])
            self.output_state = self.W_out @ state_extended
        
        # Store for training
        if store_state:
            self.states_collected.append(self.reservoir_state.copy())
            self.outputs_collected.append(self.input_state.copy())
        
        return self.reservoir_state
    
    def run(self, input_sequence: np.ndarray,
           dt: float = 1e-4,
           burn_in: int = 50) -> np.ndarray:
        """
        Run LSM on input sequence.
        
        Args:
            input_sequence: Input time series
            dt: Integration timestep
            burn_in: Burn-in steps
            
        Returns:
            Output time series
        """
        n_steps = len(input_sequence)
        outputs = np.zeros((n_steps, self.config.num_outputs))
        
        self.reset_state()
        self.liquid.reset()
        
        for t in range(n_steps):
            self.update(input_sequence[t], dt=dt, store_state=True)
            outputs[t] = self.output_state
        
        # Discard burn-in
        if burn_in > 0 and len(self.states_collected) > burn_in:
            self.states_collected = self.states_collected[burn_in:]
            self.outputs_collected = self.outputs_collected[burn_in:]
        
        return outputs
    
    def train_readout(self, target_outputs: np.ndarray,
                     method: str = "ridge") -> None:
        """
        Train output weights.
        
        Args:
            target_outputs: Target outputs
            method: Training method
        """
        EchoStateNetwork.train_readout(self, target_outputs, method)
    
    def get_liquid_spikes(self) -> List[Tuple[float, int]]:
        """Get spike times from liquid."""
        return self.liquid.spike_raster
    
    def get_liquid_raster(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get spike raster data."""
        spikes = self.liquid.spike_raster
        if not spikes:
            return np.array([]), np.array([])
        times = np.array([s[0] for s in spikes])
        ids = np.array([s[1] for s in spikes])
        return times, ids
    
    def __repr__(self) -> str:
        return (f"LiquidStateMachine(N={self.config.num_reservoir}, "
                f"spikes={len(self.liquid.spike_raster)})")


def create_esn(num_reservoir: int = 400,
              num_inputs: int = 10,
              num_outputs: int = 5,
              spectral_radius: float = 0.9,
              seed: Optional[int] = None) -> EchoStateNetwork:
    """
    Create configured Echo State Network.
    
    Args:
        num_reservoir: Number of reservoir neurons
        num_inputs: Input dimensions
        num_outputs: Output dimensions
        spectral_radius: Spectral radius
        seed: Random seed
        
    Returns:
        Configured ESN
    """
    config = ReservoirConfig(
        num_reservoir=num_reservoir,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        spectral_radius=spectral_radius,
    )
    return EchoStateNetwork(config, seed)


def create_lsm(num_reservoir: int = 400,
              num_inputs: int = 10,
              num_outputs: int = 5,
              seed: Optional[int] = None) -> LiquidStateMachine:
    """
    Create configured Liquid State Machine.
    
    Args:
        num_reservoir: Number of liquid neurons
        num_inputs: Input dimensions
        num_outputs: Output dimensions
        seed: Random seed
        
    Returns:
        Configured LSM
    """
    config = ReservoirConfig(
        num_reservoir=num_reservoir,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
    )
    return LiquidStateMachine(config, seed=seed)
