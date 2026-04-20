"""
Data Classes for Cogniton
=========================

Data structures for storing simulation results.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import json


@dataclass
class SpikeData:
    """
    Container for spike data.
    
    Attributes:
        times: Spike times (seconds)
        neuron_ids: Neuron IDs for each spike
        t_start: Recording start time
        t_end: Recording end time
        metadata: Additional metadata
    """
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    neuron_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    t_start: float = 0.0
    t_end: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        """Number of spikes."""
        return len(self.times)
    
    def add_spike(self, time: float, neuron_id: int) -> None:
        """Add a spike."""
        self.times = np.append(self.times, time)
        self.neuron_ids = np.append(self.neuron_ids, neuron_id)
    
    def get_raster(self) -> List[Tuple[float, int]]:
        """Get as list of (time, neuron_id) tuples."""
        return list(zip(self.times.tolist(), self.neuron_ids.tolist()))
    
    def get_spikes_for_neuron(self, neuron_id: int) -> np.ndarray:
        """Get spike times for specific neuron."""
        return self.times[self.neuron_ids == neuron_id]
    
    def get_firing_rate(self, neuron_id: int) -> float:
        """Get average firing rate for neuron."""
        spikes = self.get_spikes_for_neuron(neuron_id)
        duration = self.t_end - self.t_start
        if duration > 0:
            return len(spikes) / duration
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "times": self.times.tolist(),
            "neuron_ids": self.neuron_ids.tolist(),
            "t_start": self.t_start,
            "t_end": self.t_end,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpikeData":
        """Create from dictionary."""
        return cls(
            times=np.array(data["times"]),
            neuron_ids=np.array(data["neuron_ids"], dtype=int),
            t_start=data.get("t_start", 0.0),
            t_end=data.get("t_end", 0.0),
            metadata=data.get("metadata", {}),
        )
    
    def save(self, filepath: str) -> None:
        """Save to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f)
    
    @classmethod
    def load(cls, filepath: str) -> "SpikeData":
        """Load from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class VoltageData:
    """
    Container for membrane potential data.
    
    Attributes:
        time: Time points (seconds)
        voltage: Voltage values (mV)
        neuron_id: Neuron ID
        metadata: Additional metadata
    """
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    voltage: np.ndarray = field(default_factory=lambda: np.array([]))
    neuron_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        """Number of data points."""
        return len(self.time)
    
    def add_sample(self, t: float, v: float) -> None:
        """Add voltage sample."""
        self.time = np.append(self.time, t)
        self.voltage = np.append(self.voltage, v)
    
    def get_voltage_at(self, t: float, interp: bool = True) -> float:
        """Get voltage at specific time."""
        if not interp or len(self.time) < 2:
            idx = np.argmin(np.abs(self.time - t))
            return self.voltage[idx]
        
        return np.interp(t, self.time, self.voltage)
    
    def get_spike_times(self, threshold: float = 0.0) -> np.ndarray:
        """Detect spike times from voltage trace."""
        above = self.voltage > threshold
        crossings = np.diff(above.astype(int))
        rise_indices = np.where(crossings == 1)[0]
        return self.time[rise_indices]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "time": self.time.tolist(),
            "voltage": self.voltage.tolist(),
            "neuron_id": self.neuron_id,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoltageData":
        """Create from dictionary."""
        return cls(
            time=np.array(data["time"]),
            voltage=np.array(data["voltage"]),
            neuron_id=data.get("neuron_id", 0),
            metadata=data.get("metadata", {}),
        )
    
    def save(self, filepath: str) -> None:
        """Save to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f)
    
    @classmethod
    def load(cls, filepath: str) -> "VoltageData":
        """Load from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class WeightData:
    """
    Container for synaptic weight data.
    
    Attributes:
        pre_ids: Presynaptic neuron IDs
        post_ids: Postsynaptic neuron IDs
        weights: Synaptic weights
        times: Time points for each weight (optional)
        metadata: Additional metadata
    """
    pre_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    post_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    times: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        """Number of synapses."""
        return len(self.weights)
    
    def add_synapse(self, pre_id: int, post_id: int, weight: float,
                   time: Optional[float] = None) -> None:
        """Add synapse."""
        self.pre_ids = np.append(self.pre_ids, pre_id)
        self.post_ids = np.append(self.post_ids, post_id)
        self.weights = np.append(self.weights, weight)
        
        if time is not None:
            if self.times is None:
                self.times = np.array([])
            self.times = np.append(self.times, time)
    
    def update_weight(self, pre_id: int, post_id: int, new_weight: float,
                     time: Optional[float] = None) -> None:
        """Update weight for existing synapse."""
        mask = (self.pre_ids == pre_id) & (self.post_ids == post_id)
        if np.any(mask):
            idx = np.where(mask)[0][0]
            self.weights[idx] = new_weight
            if time is not None and self.times is not None:
                self.times[idx] = time
    
    def get_weight_matrix(self, n_pre: int, n_post: int) -> np.ndarray:
        """Get dense weight matrix."""
        W = np.zeros((n_pre, n_post))
        for i in range(len(self)):
            W[self.pre_ids[i], self.post_ids[i]] = self.weights[i]
        return W
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "pre_ids": self.pre_ids.tolist(),
            "post_ids": self.post_ids.tolist(),
            "weights": self.weights.tolist(),
            "metadata": self.metadata,
        }
        if self.times is not None:
            data["times"] = self.times.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeightData":
        """Create from dictionary."""
        return cls(
            pre_ids=np.array(data["pre_ids"], dtype=int),
            post_ids=np.array(data["post_ids"], dtype=int),
            weights=np.array(data["weights"]),
            times=np.array(data.get("times", [])) if "times" in data else None,
            metadata=data.get("metadata", {}),
        )
    
    def save(self, filepath: str) -> None:
        """Save to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f)
    
    @classmethod
    def load(cls, filepath: str) -> "WeightData":
        """Load from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class SimulationResult:
    """
    Complete simulation result container.
    
    Attributes:
        name: Simulation name
        spike_data: Spike recordings
        voltage_data: Voltage recordings
        weight_data: Synaptic weight history
        config: Simulation configuration
        statistics: Computed statistics
        metadata: Additional metadata
    """
    name: str
    spike_data: Optional[SpikeData] = None
    voltage_data: Optional[VoltageData] = None
    weight_data: Optional[WeightData] = None
    config: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "spike_data": self.spike_data.to_dict() if self.spike_data else None,
            "voltage_data": self.voltage_data.to_dict() if self.voltage_data else None,
            "weight_data": self.weight_data.to_dict() if self.weight_data else None,
            "config": self.config,
            "statistics": self.statistics,
            "metadata": self.metadata,
        }
    
    def save(self, filepath: str) -> None:
        """Save to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "SimulationResult":
        """Load from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        result = cls(name=data["name"])
        result.config = data.get("config", {})
        result.statistics = data.get("statistics", {})
        result.metadata = data.get("metadata", {})
        
        if data.get("spike_data"):
            result.spike_data = SpikeData.from_dict(data["spike_data"])
        if data.get("voltage_data"):
            result.voltage_data = VoltageData.from_dict(data["voltage_data"])
        if data.get("weight_data"):
            result.weight_data = WeightData.from_dict(data["weight_data"])
        
        return result
