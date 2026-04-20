"""
Synapse Module
==============

Implements synaptic connections with various models and plasticity mechanisms.

Synaptic Models:
- Static synapses: Fixed weight
- Dynamic synapses: Short-term plasticity (facilitation/depression)
- STDP synapses: Spike-timing-dependent plasticity
- conductance-based synapses: Voltage-dependent conductances
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
from enum import Enum
import numpy as np


class ConnectionType(Enum):
    """Synaptic connection types."""
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    UNKNOWN = "unknown"


class SynapseModel(Enum):
    """Synaptic model types."""
    STATIC = "static"
    STATIC_CONDUCTANCE = "static_conductance"
    STDP = "stdp"
    SHORT_TERM_PLASTICITY = "short_term"
    HOMEOSTATIC = "homeostatic"


@dataclass
class Synapse:
    """
    Synaptic connection between two neurons.
    
    Attributes:
        synapse_id: Unique identifier
        pre_neuron_id: Presynaptic neuron ID
        post_neuron_id: Postsynaptic neuron ID
        weight: Synaptic weight
        delay: Axonal/propagation delay (seconds)
        connection_type: Excitatory or inhibitory
        model: Synaptic model type
        last_spike_time: Time of last presynaptic spike
    """
    synapse_id: int
    pre_neuron_id: int
    post_neuron_id: int
    weight: float = 0.5
    delay: float = 5e-3  # 5 ms default delay
    connection_type: ConnectionType = ConnectionType.EXCITATORY
    model: SynapseModel = SynapseModel.STATIC
    
    # State variables
    last_spike_time: float = -np.inf
    last_update_time: float = 0.0
    
    # Trace for STDP
    trace: float = 0.0
    
    # Short-term plasticity state
    available: float = 1.0  # Fraction of vesicles available
    use_probability: float = 0.0
    
    # Callbacks
    _weight_change_callback: Optional[Callable] = None
    
    @property
    def is_excitatory(self) -> bool:
        """Check if synapse is excitatory."""
        return self.connection_type == ConnectionType.EXCITATORY
    
    @property
    def is_inhibitory(self) -> bool:
        """Check if synapse is inhibitory."""
        return self.connection_type == ConnectionType.INHIBITORY
    
    def get_effective_weight(self) -> float:
        """
        Get weight with any modulatory factors.
        
        Returns:
            Effective synaptic weight
        """
        if self.is_inhibitory:
            return -abs(self.weight)
        return abs(self.weight)
    
    def pre_spike(self, time: float) -> float:
        """
        Handle presynaptic spike.
        
        Args:
            time: Spike time
            
        Returns:
            Weight to deliver to postsynaptic neuron
        """
        self.last_spike_time = time
        
        # Apply short-term plasticity if enabled
        if self.model == SynapseModel.SHORT_TERM_PLASTICITY:
            w = self.weight * self.available
            # Depression: reduce available vesicles
            self.available *= (1.0 - self.use_probability)
        else:
            w = self.get_effective_weight()
        
        return w
    
    def post_spike(self, time: float) -> None:
        """
        Handle postsynaptic spike (for STDP).
        
        Args:
            time: Spike time
        """
        # Update trace
        self.trace = 1.0
    
    def update_trace(self, time: float, tau: float = 20e-3) -> float:
        """
        Decay synaptic trace.
        
        Args:
            time: Current time
            tau: Trace time constant
            
        Returns:
            Current trace value
        """
        dt = time - self.last_update_time
        self.trace *= np.exp(-dt / tau)
        self.last_update_time = time
        return self.trace
    
    def apply_stdp_update(self, delta_t: float, tau: float = 20e-3,
                          a_plus: float = 0.01, a_minus: float = 0.012,
                          w_max: float = 1.0, w_min: float = 0.0) -> float:
        """
        Apply STDP weight update.
        
        Args:
            delta_t: Time difference (post_time - pre_time)
            tau: STDP time constant
            a_plus: Learning rate for potentiation
            a_minus: Learning rate for depression
            w_max: Maximum weight
            w_min: Minimum weight
            
        Returns:
            Weight change
        """
        if delta_t > 0:
            # Pre before post: potentiation
            dw = a_plus * np.exp(-delta_t / tau)
            self.weight = min(self.weight + dw, w_max)
        else:
            # Post before pre: depression
            dw = -a_minus * np.exp(delta_t / tau)
            self.weight = max(self.weight + dw, w_min)
        
        return self.weight
    
    def update_short_term_plasticity(self, time: float, 
                                      tau_facil: float = 100e-3,
                                      tau_depr: float = 100e-3,
                                      u: float = 0.2) -> None:
        """
        Update short-term plasticity variables.
        
        Args:
            time: Current time
            tau_facil: Facilitation time constant
            tau_depr: Depression time constant
            u: Use probability increment
        """
        dt = time - self.last_update_time
        
        # Recovery of available vesicles (depression)
        self.available += (1.0 - self.available) * (dt / tau_depr)
        
        # Recovery of use probability (facilitation)
        self.use_probability *= np.exp(-dt / tau_facil)
        
        self.last_update_time = time
    
    def set_weight(self, weight: float) -> None:
        """Set synaptic weight."""
        self.weight = weight
    
    def get_weight(self) -> float:
        """Get current weight."""
        return self.weight
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize synapse to dictionary."""
        return {
            "synapse_id": self.synapse_id,
            "pre_neuron_id": self.pre_neuron_id,
            "post_neuron_id": self.post_neuron_id,
            "weight": self.weight,
            "delay": self.delay,
            "connection_type": self.connection_type.value,
            "model": self.model.value,
            "trace": self.trace,
            "last_spike_time": self.last_spike_time,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Synapse":
        """Deserialize synapse from dictionary."""
        return cls(
            synapse_id=data["synapse_id"],
            pre_neuron_id=data["pre_neuron_id"],
            post_neuron_id=data["post_neuron_id"],
            weight=data["weight"],
            delay=data["delay"],
            connection_type=ConnectionType(data["connection_type"]),
            model=SynapseModel(data["model"]),
            trace=data.get("trace", 0.0),
            last_spike_time=data.get("last_spike_time", -np.inf),
        )
    
    def __repr__(self) -> str:
        return (f"Synapse(id={self.synapse_id}, {self.pre_neuron_id}->{self.post_neuron_id}, "
                f"w={self.weight:.3f}, delay={self.delay*1000:.1f}ms)")


class SynapseGroup:
    """
    Group of synapses with vectorized operations.
    
    Manages synaptic connectivity and provides efficient updates
    for large networks.
    
    Attributes:
        group_id: Group identifier
        synapses: List of synapses
        n_synapses: Number of synapses
        connectivity_matrix: Sparse connectivity representation
    """
    
    def __init__(self, group_id: int):
        """
        Initialize synapse group.
        
        Args:
            group_id: Group identifier
        """
        self.group_id = group_id
        self.synapses: List[Synapse] = []
        self.n_synapses: int = 0
        
        # Connectivity structures
        self._pre_to_post: Dict[int, List[int]] = {}  # pre_id -> [syn_idx, ...]
        self._post_to_pre: Dict[int, List[int]] = {}  # post_id -> [syn_idx, ...]
        self._synapse_lookup: Dict[Tuple[int, int], int] = {}  # (pre, post) -> syn_idx
    
    def add_synapse(self, synapse: Synapse) -> None:
        """
        Add synapse to group.
        
        Args:
            synapse: Synapse to add
        """
        idx = len(self.synapses)
        self.synapses.append(synapse)
        self.n_synapses += 1
        
        # Update connectivity mappings
        pre = synapse.pre_neuron_id
        post = synapse.post_neuron_id
        
        if pre not in self._pre_to_post:
            self._pre_to_post[pre] = []
        self._pre_to_post[pre].append(idx)
        
        if post not in self._post_to_pre:
            self._post_to_pre[post] = []
        self._post_to_pre[post].append(idx)
        
        self._synapse_lookup[(pre, post)] = idx
    
    def get_synapse(self, pre_id: int, post_id: int) -> Optional[Synapse]:
        """
        Get synapse by pre/post IDs.
        
        Args:
            pre_id: Presynaptic neuron ID
            post_id: Postsynaptic neuron ID
            
        Returns:
            Synapse if exists, None otherwise
        """
        idx = self._synapse_lookup.get((pre_id, post_id))
        if idx is not None:
            return self.synapses[idx]
        return None
    
    def get_outgoing_synapses(self, pre_id: int) -> List[Synapse]:
        """
        Get all synapses originating from pre_id.
        
        Args:
            pre_id: Presynaptic neuron ID
            
        Returns:
            List of synapses
        """
        indices = self._pre_to_post.get(pre_id, [])
        return [self.synapses[i] for i in indices]
    
    def get_incoming_synapses(self, post_id: int) -> List[Synapse]:
        """
        Get all synapses targeting post_id.
        
        Args:
            post_id: Postsynaptic neuron ID
            
        Returns:
            List of synapses
        """
        indices = self._post_to_pre.get(post_id, [])
        return [self.synapses[i] for i in indices]
    
    def get_weights(self) -> np.ndarray:
        """Get all weights as array."""
        return np.array([s.weight for s in self.synapses])
    
    def set_weights(self, weights: np.ndarray) -> None:
        """Set all weights from array."""
        for i, w in enumerate(weights):
            self.synapses[i].weight = w
    
    def get_weight_matrix(self, n_neurons: int) -> np.ndarray:
        """
        Get dense weight matrix.
        
        Args:
            n_neurons: Number of neurons
            
        Returns:
            Weight matrix (n_neurons x n_neurons)
        """
        W = np.zeros((n_neurons, n_neurons))
        for s in self.synapses:
            W[s.pre_neuron_id, s.post_neuron_id] = s.weight
        return W
    
    def get_delay_matrix(self, n_neurons: int) -> np.ndarray:
        """
        Get dense delay matrix.
        
        Args:
            n_neurons: Number of neurons
            
        Returns:
            Delay matrix (n_neurons x n_neurons)
        """
        D = np.full((n_neurons, n_neurons), np.inf)
        for s in self.synapses:
            D[s.pre_neuron_id, s.post_neuron_id] = s.delay
        return D
    
    def apply_function_to_weights(self, func: Callable[[float], float]) -> None:
        """
        Apply function to all weights.
        
        Args:
            func: Function to apply to each weight
        """
        for s in self.synapses:
            s.weight = func(s.weight)
    
    def clamp_weights(self, w_min: float, w_max: float) -> None:
        """Clamp all weights to range."""
        for s in self.synapses:
            s.weight = max(w_min, min(w_max, s.weight))
    
    def get_statistics(self) -> Dict[str, float]:
        """Get weight statistics."""
        weights = self.get_weights()
        return {
            "mean": np.mean(weights),
            "std": np.std(weights),
            "min": np.min(weights),
            "max": np.max(weights),
            "median": np.median(weights),
        }
    
    def reset(self) -> None:
        """Reset all synapses."""
        for s in self.synapses:
            s.trace = 0.0
            s.last_spike_time = -np.inf
            s.last_update_time = 0.0
    
    def __len__(self) -> int:
        return self.n_synapses
    
    def __repr__(self) -> str:
        return f"SynapseGroup(id={self.group_id}, n={self.n_synapses})"


def create_random_connectivity(n_pre: int, n_post: int, 
                               probability: float,
                               weight_mean: float = 0.5,
                               weight_std: float = 0.1,
                               delay_mean: float = 5e-3,
                               delay_std: float = 2e-3,
                               seed: Optional[int] = None,
                               exc_ratio: float = 0.8) -> SynapseGroup:
    """
    Create random connectivity.
    
    Args:
        n_pre: Number of presynaptic neurons
        n_post: Number of postsynaptic neurons
        probability: Connection probability
        weight_mean: Mean weight
        weight_std: Weight standard deviation
        delay_mean: Mean delay
        delay_std: Delay standard deviation
        seed: Random seed
        exc_ratio: Excitatory connection ratio
        
    Returns:
        SynapseGroup with random connectivity
    """
    rng = np.random.RandomState(seed)
    group = SynapseGroup(group_id=0)
    
    for i in range(n_pre):
        for j in range(n_post):
            if rng.rand() < probability:
                # Determine connection type
                is_excitatory = rng.rand() < exc_ratio
                conn_type = (ConnectionType.EXCITATORY if is_excitatory 
                            else ConnectionType.INHIBITORY)
                
                # Sample weight
                w = rng.randn() * weight_std + weight_mean
                if not is_excitatory:
                    w = -abs(w)
                
                # Sample delay
                d = abs(rng.randn() * delay_std + delay_mean)
                
                synapse = Synapse(
                    synapse_id=len(group.synapses),
                    pre_neuron_id=i,
                    post_neuron_id=j,
                    weight=w,
                    delay=d,
                    connection_type=conn_type,
                )
                group.add_synapse(synapse)
    
    return group


def create_connectivity_matrix(n_pre: int, n_post: int,
                               weights: np.ndarray,
                               delays: Optional[np.ndarray] = None,
                               exc_ratio: float = 0.8) -> SynapseGroup:
    """
    Create connectivity from weight matrix.
    
    Args:
        n_pre: Number of presynaptic neurons
        n_post: Number of postsynaptic neurons
        weights: Weight matrix (n_pre x n_post)
        delays: Optional delay matrix
        exc_ratio: Ratio for determining connection type
        
    Returns:
        SynapseGroup
    """
    group = SynapseGroup(group_id=0)
    
    for i in range(n_pre):
        for j in range(n_post):
            w = weights[i, j]
            if abs(w) > 1e-9:  # Non-zero weight
                d = delays[i, j] if delays is not None else 5e-3
                is_exc = w > 0
                
                synapse = Synapse(
                    synapse_id=len(group.synapses),
                    pre_neuron_id=i,
                    post_neuron_id=j,
                    weight=w,
                    delay=d,
                    connection_type=(ConnectionType.EXCITATORY if is_exc 
                                    else ConnectionType.INHIBITORY),
                )
                group.add_synapse(synapse)
    
    return group
