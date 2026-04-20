"""
Hardware-in-the-Loop Module
============================

Interfaces for neuromorphic hardware platforms including:
- Intel Loihi (NPS and Loihi 2)
- IBM TrueNorth
- Generic hardware abstraction

This module provides both simulation backends and hardware interfaces
for real neuromorphic chips.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Callable
from enum import Enum
import numpy as np


class HardwareBackend(Enum):
    """Available hardware backends."""
    CPU = "cpu"
    GPU = "gpu"
    INTEL_LOIHI = "intel_loihi"
    INTEL_LOIHI_2 = "intel_loihi_2"
    IBM_TRUENORTH = "ibm_truenorth"
    SPIKE_NEURO = "spike_neuro"


class HardwareInterface(ABC):
    """
    Abstract base class for hardware interfaces.
    
    Defines the interface that all neuromorphic hardware backends
    must implement for compatibility with Cogniton.
    """
    
    def __init__(self, backend: HardwareBackend):
        """
        Initialize hardware interface.
        
        Args:
            backend: Hardware backend type
        """
        self.backend = backend
        self._connected = False
        self._config: Dict[str, Any] = {}
    
    @abstractmethod
    def connect(self, host: str = "localhost", port: int = 5000) -> bool:
        """
        Connect to hardware.
        
        Args:
            host: Host address
            port: Port number
            
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from hardware."""
        pass
    
    @abstractmethod
    def load_network(self, network_config: Dict[str, Any]) -> str:
        """
        Load network configuration onto hardware.
        
        Args:
            network_config: Network configuration dictionary
            
        Returns:
            Network ID on hardware
        """
        pass
    
    @abstractmethod
    def send_spikes(self, spikes: List[Tuple[int, float]]) -> None:
        """
        Send spike events to hardware.
        
        Args:
            spikes: List of (neuron_id, time) tuples
        """
        pass
    
    @abstractmethod
    def receive_spikes(self, timeout: float = 1.0) -> List[Tuple[int, float]]:
        """
        Receive spike events from hardware.
        
        Args:
            timeout: Reception timeout
            
        Returns:
            List of (neuron_id, time) tuples
        """
        pass
    
    @abstractmethod
    def set_parameter(self, name: str, value: Any) -> None:
        """
        Set hardware parameter.
        
        Args:
            name: Parameter name
            value: Parameter value
        """
        pass
    
    @abstractmethod
    def get_parameter(self, name: str) -> Any:
        """
        Get hardware parameter.
        
        Args:
            name: Parameter name
            
        Returns:
            Parameter value
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset hardware state."""
        pass
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to hardware."""
        return self._connected
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(backend={self.backend.value}, connected={self._connected})"


class CPUBackend(HardwareInterface):
    """
    CPU-based simulation backend.
    
    Provides optimized CPU simulation for networks without hardware.
    """
    
    def __init__(self, num_threads: int = 4):
        """
        Initialize CPU backend.
        
        Args:
            num_threads: Number of OpenMP threads
        """
        super().__init__(HardwareBackend.CPU)
        self.num_threads = num_threads
        self._numba_available = False
        
        # Try to import numba for JIT compilation
        try:
            from numba import jit
            self._jit = jit
            self._numba_available = True
        except ImportError:
            self._jit = lambda x: x  # No-op decorator
        
        # State
        self._neurons: Dict[int, Dict[str, float]] = {}
        self._synapses: List[Dict[str, Any]] = []
        self._spike_log: List[Tuple[int, float]] = []
    
    def connect(self, host: str = "localhost", port: int = 5000) -> bool:
        """CPU backend is always connected (no external hardware)."""
        self._connected = True
        return True
    
    def disconnect(self) -> None:
        """CPU backend disconnect (no-op)."""
        self._connected = False
    
    def load_network(self, network_config: Dict[str, Any]) -> str:
        """Load network onto CPU."""
        network_id = f"cpu_network_{id(self)}"
        
        # Initialize neurons
        n_neurons = network_config.get("num_neurons", 100)
        for i in range(n_neurons):
            self._neurons[i] = {
                "v": network_config.get("v_init", -70.0),
                "i_syn": 0.0,
                "v_thresh": network_config.get("v_thresh", -55.0),
                "v_reset": network_config.get("v_reset", -75.0),
                "tau_mem": network_config.get("tau_mem", 20e-3),
            }
        
        # Initialize synapses
        self._synapses = network_config.get("synapses", [])
        
        return network_id
    
    def send_spikes(self, spikes: List[Tuple[int, float]]) -> None:
        """Process incoming spikes."""
        for neuron_id, time in spikes:
            if neuron_id in self._neurons:
                self._spike_log.append((neuron_id, time))
    
    def receive_spikes(self, timeout: float = 1.0) -> List[Tuple[int, float]]:
        """Get spikes (CPU returns locally generated spikes)."""
        spikes = self._spike_log.copy()
        self._spike_log.clear()
        return spikes
    
    def set_parameter(self, name: str, value: Any) -> None:
        """Set backend parameter."""
        self._config[name] = value
        if name == "num_threads":
            self.num_threads = value
    
    def get_parameter(self, name: str) -> Any:
        """Get backend parameter."""
        return self._config.get(name)
    
    def reset(self) -> None:
        """Reset CPU simulation state."""
        for neuron in self._neurons.values():
            neuron["v"] = -70.0
            neuron["i_syn"] = 0.0
        self._spike_log.clear()
    
    def step(self, dt: float) -> List[Tuple[int, float]]:
        """
        Execute simulation step.
        
        Args:
            dt: Timestep
            
        Returns:
            List of (neuron_id, time) spikes
        """
        spikes = []
        
        for neuron_id, neuron in self._neurons.items():
            # Update neuron (simplified LIF)
            v = neuron["v"]
            i_syn = neuron["i_syn"]
            tau = neuron["tau_mem"]
            
            # Decay
            v = v * np.exp(-dt / tau) + i_syn * (1 - np.exp(-dt / tau))
            i_syn *= np.exp(-dt / tau)
            
            # Check threshold
            if v >= neuron["v_thresh"]:
                spikes.append((neuron_id, 0.0))
                v = neuron["v_reset"]
            
            neuron["v"] = v
            neuron["i_syn"] = i_syn
        
        return spikes
    
    def __repr__(self) -> str:
        return f"CPUBackend(threads={self.num_threads}, neurons={len(self._neurons)})"


class IntelLoihiInterface(HardwareInterface):
    """
    Interface for Intel Loihi neuromorphic chip.
    
    Supports Loihi 1 (NPS) and Loihi 2 architectures.
    
    Note: Requires nxsdk to be installed and Loihi hardware/access.
    """
    
    def __init__(self, version: int = 2):
        """
        Initialize Loihi interface.
        
        Args:
            version: Loihi version (1 or 2)
        """
        super().__init__(
            HardwareBackend.INTEL_LOIHI_2 if version == 2 
            else HardwareBackend.INTEL_LOIHI
        )
        self.version = version
        self._network_id: Optional[str] = None
        self._emulator = False
        
        # Try to import nxsdk
        self._nxsdk_available = False
        try:
            import nxsdk
            self._nxsdk_available = True
        except ImportError:
            pass
    
    def connect(self, host: str = "localhost", port: int = 5000) -> bool:
        """Connect to Loihi hardware or emulator."""
        if not self._nxsdk_available:
            print("Warning: nxsdk not available. Using emulator mode.")
            self._emulator = True
        
        self._connected = True
        return True
    
    def disconnect(self) -> None:
        """Disconnect from Loihi."""
        self._network_id = None
        self._connected = False
    
    def load_network(self, network_config: Dict[str, Any]) -> str:
        """Load network onto Loihi."""
        if self._emulator:
            # Use CPU emulation
            network_id = "emulated_network"
        else:
            # Load onto actual Loihi
            network_id = self._create_loihi_network(network_config)
        
        self._network_id = network_id
        return network_id
    
    def _create_loihi_network(self, config: Dict[str, Any]) -> str:
        """Create Loihi network graph."""
        # This would use nxsdk API to create network
        # Placeholder implementation
        return f"loihi_network_{config.get('id', 0)}"
    
    def send_spikes(self, spikes: List[Tuple[int, float]]) -> None:
        """Send spikes to Loihi."""
        if not self._connected:
            raise RuntimeError("Not connected to Loihi")
        
        # Convert to Loihi spike format and send
        # Implementation depends on nxsdk version
    
    def receive_spikes(self, timeout: float = 1.0) -> List[Tuple[int, float]]:
        """Receive spikes from Loihi."""
        if not self._connected:
            raise RuntimeError("Not connected to Loihi")
        
        # Get spikes from Loihi
        # Implementation depends on nxsdk version
        return []
    
    def set_parameter(self, name: str, value: Any) -> None:
        """Set Loihi parameter."""
        self._config[name] = value
    
    def get_parameter(self, name: str) -> Any:
        """Get Loihi parameter."""
        return self._config.get(name)
    
    def reset(self) -> None:
        """Reset Loihi state."""
        if self._network_id:
            # Reset network state on Loihi
            pass
    
    def set_core_config(self, cores: int, neurons_per_core: int = 1024) -> None:
        """
        Configure core allocation.
        
        Args:
            cores: Number of cores
            neurons_per_core: Neurons per core
        """
        self._config["cores"] = cores
        self._config["neurons_per_core"] = neurons_per_core
    
    def enable_profiling(self, enabled: bool = True) -> None:
        """Enable/disable hardware profiling."""
        self._config["profiling"] = enabled
    
    def __repr__(self) -> str:
        return f"IntelLoihiInterface(version={self.version}, connected={self._connected})"


class IBMTrueNorthInterface(HardwareInterface):
    """
    Interface for IBM TrueNorth neuromorphic chip.
    
    Note: Requires IBM's corelet programming environment.
    """
    
    def __init__(self):
        """Initialize TrueNorth interface."""
        super().__init__(HardwareBackend.IBM_TRUENORTH)
        self._network_id: Optional[str] = None
        
        # Try to import IBM's libraries
        self._truenorth_available = False
        try:
            # IBM's TrueNorth API would be imported here
            self._truenorth_available = False  # Placeholder
        except ImportError:
            pass
    
    def connect(self, host: str = "localhost", port: int = 5000) -> bool:
        """Connect to TrueNorth."""
        if not self._truenorth_available:
            print("Warning: TrueNorth SDK not available.")
        
        self._connected = True
        return True
    
    def disconnect(self) -> None:
        """Disconnect from TrueNorth."""
        self._network_id = None
        self._connected = False
    
    def load_network(self, network_config: Dict[str, Any]) -> str:
        """Load network onto TrueNorth."""
        network_id = f"truenorth_{id(self)}"
        self._network_id = network_id
        return network_id
    
    def send_spikes(self, spikes: List[Tuple[int, float]]) -> None:
        """Send spikes to TrueNorth."""
        pass
    
    def receive_spikes(self, timeout: float = 1.0) -> List[Tuple[int, float]]:
        """Receive spikes from TrueNorth."""
        return []
    
    def set_parameter(self, name: str, value: Any) -> None:
        """Set TrueNorth parameter."""
        self._config[name] = value
    
    def get_parameter(self, name: str) -> Any:
        """Get TrueNorth parameter."""
        return self._config.get(name)
    
    def reset(self) -> None:
        """Reset TrueNorth state."""
        pass
    
    def configure_cores(self, num_cores: int = 4096) -> None:
        """
        Configure TrueNorth cores.
        
        Args:
            num_cores: Number of cores to use
        """
        self._config["num_cores"] = num_cores
    
    def __repr__(self) -> str:
        return f"IBMTrueNorthInterface(connected={self._connected})"


def create_hardware_interface(backend: HardwareBackend,
                             **kwargs) -> HardwareInterface:
    """
    Factory function to create hardware interfaces.
    
    Args:
        backend: Hardware backend type
        **kwargs: Additional arguments for interface
        
    Returns:
        Hardware interface instance
    """
    if backend == HardwareBackend.CPU:
        return CPUBackend(num_threads=kwargs.get("num_threads", 4))
    
    elif backend == HardwareBackend.INTEL_LOIHI:
        return IntelLoihiInterface(version=1)
    
    elif backend == HardwareBackend.INTEL_LOIHI_2:
        return IntelLoihiInterface(version=2)
    
    elif backend == HardwareBackend.IBM_TRUENORTH:
        return IBMTrueNorthInterface()
    
    else:
        raise ValueError(f"Unknown backend: {backend}")


@dataclass
class HardwareSpec:
    """
    Specifications for neuromorphic hardware.
    
    Provides hardware capability information.
    """
    name: str
    cores: int
    neurons_per_core: int
    synapses_per_neuron: int
    max_weight_bits: int
    on_chip_learning: bool
    real_time_factors: Dict[str, float]
    
    @property
    def total_neurons(self) -> int:
        """Total neurons available."""
        return self.cores * self.neurons_per_core
    
    @property
    def total_synapses(self) -> int:
        """Total synapses (approximate)."""
        return self.total_neurons * self.synapses_per_neuron


# Hardware specifications
LOIHI_SPEC = HardwareSpec(
    name="Intel Loihi 2",
    cores=128,
    neurons_per_core=1024,
    synapses_per_neuron=4096,
    max_weight_bits=8,
    on_chip_learning=True,
    real_time_factors={"sparse": 100, "dense": 10},
)

TRUENORTH_SPEC = HardwareSpec(
    name="IBM TrueNorth",
    cores=4096,
    neurons_per_core=256,
    synapses_per_neuron=256,
    max_weight_bits=4,
    on_chip_learning=False,
    real_time_factors={"sparse": 1000, "dense": 100},
)
