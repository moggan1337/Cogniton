"""
Leaky Integrate-and-Fire (LIF) Neuron Model
============================================

Implementation of the LIF neuron model with event-driven updates
and comprehensive dynamics simulation.

The LIF model is defined by:
    τ_m * dv/dt = -(v - v_rest) + R_m * i_syn(t) + R_m * i_offset
    
With spike generation when v >= v_thresh:
    v <- v_reset
    Spike emitted
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
import numpy as np

from cogniton.core.config import LIFConfig
from cogniton.core.event import Event, EventType


class LIFNeuron:
    """
    Leaky Integrate-and-Fire (LIF) neuron.
    
    The LIF model captures essential neuronal dynamics while remaining
    computationally efficient. It's widely used in neuromorphic computing
    due to its event-driven nature and biological plausibility.
    
    Mathematical Model:
    -------------------
    τ_m * dv/dt = -(v - v_rest) + R_m * I(t)
    
    Where:
    - τ_m: Membrane time constant
    - v: Membrane potential
    - v_rest: Resting potential
    - R_m: Membrane resistance
    - I(t): Total input current
    
    Spike generation:
    - When v >= v_thresh: emit spike, reset to v_reset
    - Refractory period prevents spiking during recovery
    
    Attributes:
        neuron_id: Unique identifier
        config: LIF neuron parameters
        v: Current membrane potential (mV)
        i_syn: Synaptic current (nA)
        refractory_remaining: Time in refractory period (s)
        spike_times: List of spike times
        last_update: Last update time
    """
    
    def __init__(self, neuron_id: int, config: Optional[LIFConfig] = None,
                 seed: Optional[int] = None):
        """
        Initialize LIF neuron.
        
        Args:
            neuron_id: Unique neuron identifier
            config: LIF parameters (uses default if None)
            seed: Random seed for stochastic behavior
        """
        self.neuron_id = neuron_id
        self.config = config or LIFConfig()
        
        # State variables
        self.v = self.config.v_init
        self.i_syn = 0.0
        self.refractory_remaining = 0.0
        
        # History tracking
        self.spike_times: List[float] = []
        self.v_history: List[Tuple[float, float]] = []  # (time, v)
        self.i_history: List[Tuple[float, float]] = []  # (time, i_syn)
        
        # Timing
        self.last_update: float = 0.0
        self.birth_time: float = 0.0
        
        # Random number generator
        self._rng = np.random.RandomState(seed)
        
        # Callback for spike events
        self._spike_callback: Optional[Callable] = None
    
    @property
    def tau_mem(self) -> float:
        """Membrane time constant (seconds)."""
        return self.config.tau_mem
    
    @property
    def tau_syn(self) -> float:
        """Synaptic time constant (seconds)."""
        return self.config.tau_syn
    
    @property
    def v_rest(self) -> float:
        """Resting potential (mV)."""
        return self.config.v_rest
    
    @property
    def v_thresh(self) -> float:
        """Spike threshold (mV)."""
        return self.config.v_thresh
    
    @property
    def v_reset(self) -> float:
        """Reset potential (mV)."""
        return self.config.v_reset
    
    @property
    def is_refractory(self) -> bool:
        """Whether neuron is in refractory period."""
        return self.refractory_remaining > 0
    
    @property
    def membrane_time_constant_ms(self) -> float:
        """Membrane time constant in milliseconds."""
        return self.tau_mem * 1000
    
    def set_spike_callback(self, callback: Callable[["LIFNeuron", float], None]) -> None:
        """
        Set callback for spike events.
        
        Args:
            callback: Function(neuron, spike_time)
        """
        self._spike_callback = callback
    
    def receive_spike(self, time: float, weight: float) -> None:
        """
        Receive synaptic input from spike.
        
        Args:
            time: Arrival time of spike
            weight: Synaptic weight
        """
        # Add to synaptic current (exponential decay model)
        self.i_syn += weight
        
        # Record current injection
        self.i_history.append((time, self.i_syn))
    
    def inject_current(self, current: float) -> None:
        """
        Inject external current.
        
        Args:
            current: Current in nA
        """
        self.i_syn += current
    
    def update(self, t: float, dt: Optional[float] = None) -> bool:
        """
        Update neuron state using exponential Euler integration.
        
        Args:
            t: Current simulation time
            dt: Integration timestep (uses adaptive if None)
            
        Returns:
            True if spike occurred
        """
        if dt is None:
            dt = t - self.last_update
        
        self.last_update = t
        
        # Handle refractory period
        if self.refractory_remaining > 0:
            self.refractory_remaining -= dt
            # During refractory: hold at reset, decay syn current
            self.i_syn *= np.exp(-dt / self.tau_syn)
            self.v_history.append((t, self.v))
            return False
        
        # Membrane potential update (exponential Euler)
        # dv/dt = (-(v - v_rest) + R_m * i_syn) / tau_mem
        
        # Compute decay factor
        exp_mem = np.exp(-dt / self.tau_mem)
        exp_syn = np.exp(-dt / self.tau_syn)
        
        # Update synaptic current
        i_syn_new = self.i_syn * exp_syn
        
        # Update membrane potential
        v_inf = self.v_rest + self.config.r_mem * i_syn_new + self.config.i_offset
        self.v = self.v_rest + (self.v - v_inf) * exp_mem
        
        self.i_syn = i_syn_new
        
        # Record state
        self.v_history.append((t, self.v))
        
        # Check for spike
        if self.v >= self.v_thresh:
            return self._generate_spike(t)
        
        return False
    
    def _generate_spike(self, t: float) -> bool:
        """
        Generate spike and reset neuron.
        
        Args:
            t: Spike time
            
        Returns:
            True (spike occurred)
        """
        # Record spike
        self.spike_times.append(t)
        
        # Reset membrane potential
        self.v = self.v_reset
        
        # Enter refractory period
        self.refractory_remaining = self.config.refract_time
        
        # Call spike callback if registered
        if self._spike_callback:
            self._spike_callback(self, t)
        
        return True
    
    def reset_state(self) -> None:
        """Reset neuron to initial state."""
        self.v = self.config.v_init
        self.i_syn = 0.0
        self.refractory_remaining = 0.0
        self.spike_times.clear()
        self.v_history.clear()
        self.i_history.clear()
        self.last_update = 0.0
    
    def get_firing_rate(self, t_start: float = 0.0, t_end: Optional[float] = None) -> float:
        """
        Calculate firing rate over time window.
        
        Args:
            t_start: Start time
            t_end: End time (defaults to last spike or current)
            
        Returns:
            Firing rate in Hz
        """
        if t_end is None:
            t_end = self.spike_times[-1] if self.spike_times else self.last_update
        
        duration = t_end - t_start
        if duration <= 0:
            return 0.0
        
        spikes_in_window = sum(1 for st in self.spike_times if t_start <= st <= t_end)
        return spikes_in_window / duration
    
    def get_average_voltage(self, t_start: float = 0.0, 
                            t_end: Optional[float] = None) -> float:
        """
        Calculate average membrane potential.
        
        Args:
            t_start: Start time
            t_end: End time
            
        Returns:
            Average voltage in mV
        """
        if not self.v_history:
            return self.v_rest
        
        if t_end is None:
            t_end = self.v_history[-1][0]
        
        relevant = [(t, v) for t, v in self.v_history if t_start <= t <= t_end]
        
        if not relevant:
            return self.v_rest
        
        return np.mean([v for _, v in relevant])
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize neuron state to dictionary."""
        return {
            "neuron_id": self.neuron_id,
            "v": self.v,
            "i_syn": self.i_syn,
            "refractory_remaining": self.refractory_remaining,
            "spike_times": self.spike_times.copy(),
            "v_history": self.v_history.copy(),
            "config": {
                "tau_mem": self.config.tau_mem,
                "tau_syn": self.config.tau_syn,
                "v_rest": self.config.v_rest,
                "v_thresh": self.config.v_thresh,
                "v_reset": self.config.v_reset,
                "r_mem": self.config.r_mem,
                "i_offset": self.config.i_offset,
                "refract_time": self.config.refract_time,
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LIFNeuron":
        """Deserialize neuron from dictionary."""
        config = LIFConfig(**data["config"])
        neuron = cls(data["neuron_id"], config)
        neuron.v = data["v"]
        neuron.i_syn = data["i_syn"]
        neuron.refractory_remaining = data["refractory_remaining"]
        neuron.spike_times = data["spike_times"]
        neuron.v_history = data["v_history"]
        return neuron
    
    def __repr__(self) -> str:
        return (f"LIFNeuron(id={self.neuron_id}, v={self.v:.2f}mV, "
                f"spikes={len(self.spike_times)}, "
                f"refractory={self.is_refractory})")


class LIFNeuronGroup:
    """
    Group of LIF neurons with vectorized operations.
    
    Provides efficient simulation of large neuron populations
    using NumPy array operations.
    
    Attributes:
        group_id: Group identifier
        neurons: List of individual neurons
        config: Shared configuration
        n: Number of neurons
    """
    
    def __init__(self, group_id: int, n: int, 
                 config: Optional[LIFConfig] = None,
                 seeds: Optional[List[int]] = None):
        """
        Initialize neuron group.
        
        Args:
            group_id: Group identifier
            n: Number of neurons
            config: Shared LIF configuration
            seeds: Random seeds for each neuron
        """
        self.group_id = group_id
        self.config = config or LIFConfig()
        self.n = n
        
        # Create neurons
        self.neurons: List[LIFNeuron] = []
        for i in range(n):
            seed = seeds[i] if seeds is not None else None
            neuron = LIFNeuron(neuron_id=i, config=self.config, seed=seed)
            self.neurons.append(neuron)
        
        # Vectorized state arrays
        self._init_arrays()
    
    def _init_arrays(self) -> None:
        """Initialize NumPy state arrays."""
        self.v = np.full(self.n, self.config.v_init)  # Membrane potentials
        self.i_syn = np.zeros(self.n)  # Synaptic currents
        
        # Refractory state
        self.refractory = np.zeros(self.n, dtype=bool)
        self.refractory_time = np.zeros(self.n)
        
        # Spike tracking
        self.spike_count = np.zeros(self.n, dtype=int)
        self.last_spike_time = np.full(self.n, -np.inf)
        
        # History
        self.spike_times: List[Tuple[float, int]] = []
    
    @property
    def spike_raster(self) -> List[Tuple[float, int]]:
        """Get spike raster data (time, neuron_id pairs)."""
        return self.spike_times
    
    def receive_spikes(self, times: np.ndarray, source_ids: np.ndarray,
                       weights: np.ndarray) -> None:
        """
        Process incoming spikes (vectorized).
        
        Args:
            times: Spike times
            source_ids: Target neuron indices
            weights: Synaptic weights
        """
        for i, (t, sid, w) in enumerate(zip(times, source_ids, weights)):
            if 0 <= sid < self.n:
                self.i_syn[sid] += w
    
    def inject_current(self, current: np.ndarray) -> None:
        """
        Inject current into all neurons.
        
        Args:
            current: Current array (nA), shape (n,) or scalar
        """
        self.i_syn += np.asarray(current)
    
    def update(self, t: float, dt: float) -> np.ndarray:
        """
        Update all neurons (vectorized).
        
        Args:
            t: Current time
            dt: Integration timestep
            
        Returns:
            Boolean array of neurons that spiked
        """
        # Update refractory timers
        self.refractory_time = np.maximum(0, self.refractory_time - dt)
        self.refractory = self.refractory_time > 0
        
        # Compute exponential decay factors
        exp_mem = np.exp(-dt / self.config.tau_mem)
        exp_syn = np.exp(-dt / self.config.tau_syn)
        
        # Update synaptic currents (only non-refractory neurons)
        mask_active = ~self.refractory
        self.i_syn[mask_active] *= exp_syn
        
        # Compute steady-state voltage
        v_inf = (self.config.v_rest + 
                 self.config.r_mem * self.i_syn + 
                 self.config.i_offset)
        
        # Update membrane potentials
        self.v[mask_active] = (self.config.v_rest + 
                               (self.v[mask_active] - v_inf[mask_active]) * exp_mem)
        
        # Check for spikes
        spiked = (self.v >= self.config.v_thresh) & mask_active
        
        # Process spikes
        spike_indices = np.where(spiked)[0]
        
        for idx in spike_indices:
            # Record spike
            self.spike_count[idx] += 1
            self.last_spike_time[idx] = t
            self.spike_times.append((t, self.group_id * 1000 + idx))
            
            # Reset voltage
            self.v[idx] = self.config.v_reset
            
            # Enter refractory period
            self.refractory_time[idx] = self.config.refract_time
        
        return spiked
    
    def get_firing_rates(self, t_start: float = 0.0, 
                         t_end: Optional[float] = None) -> np.ndarray:
        """
        Calculate firing rates for all neurons.
        
        Args:
            t_start: Start time
            t_end: End time
            
        Returns:
            Firing rate array (Hz)
        """
        if t_end is None:
            t_end = max(self.last_spike_time.max() if self.n > 0 else 0, t_start + 1e-6)
        
        duration = t_end - t_start
        return self.spike_count / duration
    
    def get_population_firing_rate(self, t: float, window: float = 0.1) -> float:
        """
        Calculate population firing rate.
        
        Args:
            t: Current time
            window: Time window (seconds)
            
        Returns:
            Population rate in Hz
        """
        t_start = max(0, t - window)
        spikes_in_window = sum(1 for st, _ in self.spike_times 
                               if t_start <= st <= t)
        return spikes_in_window / (window * self.n)
    
    def reset(self) -> None:
        """Reset all neurons to initial state."""
        self._init_arrays()
        for neuron in self.neurons:
            neuron.reset_state()
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """Get current state as dictionary of arrays."""
        return {
            "v": self.v.copy(),
            "i_syn": self.i_syn.copy(),
            "refractory": self.refractory.copy(),
            "refractory_time": self.refractory_time.copy(),
            "spike_count": self.spike_count.copy(),
        }
    
    def set_state(self, state: Dict[str, np.ndarray]) -> None:
        """Set state from dictionary of arrays."""
        self.v = state["v"].copy()
        self.i_syn = state["i_syn"].copy()
        self.refractory = state["refractory"].copy()
        self.refractory_time = state["refractory_time"].copy()
        self.spike_count = state["spike_count"].copy()
    
    def __len__(self) -> int:
        """Number of neurons in group."""
        return self.n
    
    def __getitem__(self, index: int) -> LIFNeuron:
        """Get neuron by index."""
        return self.neurons[index]
    
    def __repr__(self) -> str:
        total_spikes = sum(len(n.spike_times) for n in self.neurons)
        return (f"LIFNeuronGroup(id={self.group_id}, n={self.n}, "
                f"total_spikes={total_spikes})")
