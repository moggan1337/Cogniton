"""
Simulation Configuration Module
===============================

Defines configuration classes for neuromorphic simulations.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import numpy as np


class SimulationMode(Enum):
    """Simulation mode types."""
    EVENT_DRIVEN = "event_driven"
    FIXED_TIMESTEP = "fixed_timestep"
    HYBRID = "hybrid"
    REAL_TIME = "real_time"


class HardwareBackend(Enum):
    """Hardware backend options."""
    CPU = "cpu"
    GPU = "gpu"
    INTEL_LOIHI = "intel_loihi"
    IBM_TRUENORTH = "ibm_truenorth"
    INTEL_LOIHI_2 = "intel_loihi_2"
    SPIKE_NEURO = "spike_neuro"


@dataclass
class SimulationConfig:
    """
    Main simulation configuration.
    
    Attributes:
        mode: Simulation mode (event-driven, fixed timestep, etc.)
        dt: Fixed timestep for hybrid/fixed modes (seconds)
        t_max: Maximum simulation time (seconds)
        seed: Random seed for reproducibility
        hardware: Hardware backend target
        precision: Numerical precision ('float32', 'float64')
        record_spikes: Whether to record spike times
        record_voltage: Whether to record membrane potentials
        record_weights: Whether to record synaptic weights
        parallel: Enable parallel processing
        num_threads: Number of threads for parallel processing
    """
    mode: SimulationMode = SimulationMode.EVENT_DRIVEN
    dt: float = 1e-4  # 0.1 ms default timestep
    t_max: float = 1.0  # 1 second default
    seed: Optional[int] = None
    hardware: HardwareBackend = HardwareBackend.CPU
    precision: str = "float64"
    record_spikes: bool = True
    record_voltage: bool = True
    record_weights: bool = False
    parallel: bool = False
    num_threads: int = 4
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.t_max <= 0:
            raise ValueError("t_max must be positive")
        if self.precision not in ["float32", "float64"]:
            raise ValueError("precision must be 'float32' or 'float64'")
        if self.num_threads < 1:
            raise ValueError("num_threads must be at least 1")


@dataclass
class NetworkConfig:
    """
    Spiking neural network configuration.
    
    Attributes:
        num_neurons: Total number of neurons
        num_inputs: Number of input neurons
        num_outputs: Number of output neurons
        input_rate: Poisson input firing rate (Hz)
        connection_probability: Probability of synaptic connection
        initial_weight_mean: Mean initial synaptic weight
        initial_weight_std: Standard deviation of initial weights
        delay_mean: Mean axonal delay (seconds)
        delay_std: Standard deviation of axonal delay
        excitatory_ratio: Ratio of excitatory neurons (0-1)
    """
    num_neurons: int = 1000
    num_inputs: int = 100
    num_outputs: int = 10
    input_rate: float = 10.0  # Hz
    connection_probability: float = 0.1
    initial_weight_mean: float = 0.5
    initial_weight_std: float = 0.1
    delay_mean: float = 5e-3  # 5 ms
    delay_std: float = 2e-3   # 2 ms
    excitatory_ratio: float = 0.8
    
    def __post_init__(self):
        """Validate network configuration."""
        if self.num_neurons < 1:
            raise ValueError("num_neurons must be at least 1")
        if self.num_inputs < 0 or self.num_inputs > self.num_neurons:
            raise ValueError("num_inputs must be between 0 and num_neurons")
        if self.num_outputs < 0 or self.num_outputs > self.num_neurons:
            raise ValueError("num_outputs must be between 0 and num_neurons")
        if not 0 <= self.excitatory_ratio <= 1:
            raise ValueError("excitatory_ratio must be between 0 and 1")


@dataclass
class LIFConfig:
    """
    Leaky Integrate-and-Fire neuron configuration.
    
    Attributes:
        tau_mem: Membrane time constant (seconds)
        tau_syn: Synaptic time constant (seconds)
        v_rest: Resting membrane potential (mV)
        v_thresh: Spike threshold (mV)
        v_reset: Reset membrane potential after spike (mV)
        v_init: Initial membrane potential (mV)
        r_mem: Membrane resistance (MOhm)
        i_offset: Constant offset current (nA)
        refract_time: Refractory period (seconds)
    """
    tau_mem: float = 20e-3   # 20 ms
    tau_syn: float = 10e-3    # 10 ms
    v_rest: float = -70.0    # mV
    v_thresh: float = -55.0  # mV
    v_reset: float = -75.0   # mV
    v_init: float = -70.0    # mV
    r_mem: float = 10.0      # MOhm
    i_offset: float = 0.0    # nA
    refract_time: float = 2e-3  # 2 ms refractory period
    
    def __post_init__(self):
        """Validate LIF configuration."""
        if self.tau_mem <= 0:
            raise ValueError("tau_mem must be positive")
        if self.tau_syn <= 0:
            raise ValueError("tau_syn must be positive")
        if self.v_thresh <= self.v_rest:
            raise ValueError("v_thresh must be greater than v_rest")
        if self.v_reset >= self.v_thresh:
            raise ValueError("v_reset must be less than v_thresh")


@dataclass
class HHConfig:
    """
    Hodgkin-Huxley neuron configuration.
    
    Attributes:
        g_na: Sodium channel maximum conductance (mS/cm²)
        g_k: Potassium channel maximum conductance (mS/cm²)
        g_l: Leak channel conductance (mS/cm²)
        e_na: Sodium reversal potential (mV)
        e_k: Potassium reversal potential (mV)
        e_l: Leak reversal potential (mV)
        c_m: Membrane capacitance (μF/cm²)
        v_init: Initial membrane potential (mV)
        temperature: Operating temperature (°C)
        q10: Temperature scaling factor
    """
    g_na: float = 120.0    # mS/cm²
    g_k: float = 36.0     # mS/cm²
    g_l: float = 0.3      # mS/cm²
    e_na: float = 50.0    # mV
    e_k: float = -77.0    # mV
    e_l: float = -54.387  # mV
    c_m: float = 1.0      # μF/cm²
    v_init: float = -65.0 # mV
    temperature: float = 6.3  # °C (standard HH temperature)
    q10: float = 3.0
        
    def __post_init__(self):
        """Validate Hodgkin-Huxley configuration."""
        if self.g_na <= 0 or self.g_k <= 0 or self.g_l <= 0:
            raise ValueError("Conductances must be positive")
        if self.c_m <= 0:
            raise ValueError("c_m must be positive")


@dataclass
class STDPConfig:
    """
    Spike-Timing-Dependent Plasticity configuration.
    
    Attributes:
        tau_plus: Time constant for potentiation (seconds)
        tau_minus: Time constant for depression (seconds)
        a_plus: Learning rate for potentiation
        a_minus: Learning rate for depression
        w_max: Maximum synaptic weight
        w_min: Minimum synaptic weight
        w_init: Initial synaptic weight
        use_soft_bounds: Use soft bounds (True) or hard bounds (False)
        mu: Weight dependence exponent (0 = additive, 1 = multiplicative)
    """
    tau_plus: float = 20e-3   # 20 ms
    tau_minus: float = 20e-3  # 20 ms
    a_plus: float = 0.01
    a_minus: float = 0.012
    w_max: float = 1.0
    w_min: float = 0.0
    w_init: float = 0.5
    use_soft_bounds: bool = True
    mu: float = 0.0  # Additive STDP by default
    
    def __post_init__(self):
        """Validate STDP configuration."""
        if self.tau_plus <= 0 or self.tau_minus <= 0:
            raise ValueError("Tau values must be positive")
        if self.a_plus <= 0 or self.a_minus <= 0:
            raise ValueError("Learning rates must be positive")
        if self.w_min < 0:
            raise ValueError("w_min cannot be negative")
        if self.w_max <= self.w_min:
            raise ValueError("w_max must be greater than w_min")


@dataclass 
class ReservoirConfig:
    """
    Reservoir computing configuration.
    
    Attributes:
        num_reservoir: Number of reservoir neurons
        num_inputs: Number of input dimensions
        num_outputs: Number of output dimensions
        spectral_radius: Spectral radius of reservoir weight matrix
        input_scaling: Input connection scaling factor
        connectivity: Internal connectivity probability
        leak_rate: Membrane leak rate for echo state network
        regularization: Regularization parameter for readout training
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
        """Validate reservoir configuration."""
        if self.num_reservoir < 1:
            raise ValueError("num_reservoir must be at least 1")
        if self.spectral_radius <= 0:
            raise ValueError("spectral_radius must be positive")
        if not 0 < self.connectivity <= 1:
            raise ValueError("connectivity must be in (0, 1]")


@dataclass
class HardwareConfig:
    """
    Hardware-in-the-loop configuration.
    
    Attributes:
        backend: Target hardware backend
        host: Host address for hardware communication
        port: Port for hardware communication
        chip_id: Specific chip identifier
        num_cores: Number of cores to use
        memory_limit: Memory limit in bytes
        latency_compensation: Enable latency compensation
        async_mode: Use asynchronous communication
    """
    backend: HardwareBackend = HardwareBackend.CPU
    host: str = "localhost"
    port: int = 5000
    chip_id: int = 0
    num_cores: int = 128
    memory_limit: Optional[int] = None
    latency_compensation: bool = True
    async_mode: bool = False
    
    def __post_init__(self):
        """Validate hardware configuration."""
        if self.port < 1 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.num_cores < 1:
            raise ValueError("num_cores must be at least 1")


@dataclass
class MonitoringConfig:
    """
    Monitoring and logging configuration.
    
    Attributes:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
        metrics_interval: Interval for computing metrics (seconds)
        plot_interval: Interval for updating plots (seconds)
        save_state_interval: Interval for saving state (seconds)
        checkpoint_dir: Directory for checkpoints
    """
    log_level: str = "INFO"
    log_file: Optional[str] = None
    metrics_interval: float = 0.1
    plot_interval: float = 1.0
    save_state_interval: float = 10.0
    checkpoint_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validate monitoring configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")


class ConfigBuilder:
    """
    Builder class for constructing simulation configurations.
    
    Example:
        config = (ConfigBuilder()
                 .with_simulation(dt=1e-4, t_max=10.0)
                 .with_network(num_neurons=1000)
                 .with_lif(tau_mem=20e-3)
                 .with_stdp(tau_plus=20e-3)
                 .build())
    """
    
    def __init__(self):
        self._sim_config = SimulationConfig()
        self._net_config = NetworkConfig()
        self._lif_config = LIFConfig()
        self._hh_config = HHConfig()
        self._stdp_config = STDPConfig()
        self._reservoir_config = ReservoirConfig()
        self._hardware_config = HardwareConfig()
        self._monitoring_config = MonitoringConfig()
    
    def with_simulation(self, **kwargs) -> "ConfigBuilder":
        """Update simulation configuration."""
        for key, value in kwargs.items():
            if hasattr(self._sim_config, key):
                setattr(self._sim_config, key, value)
        return self
    
    def with_network(self, **kwargs) -> "ConfigBuilder":
        """Update network configuration."""
        for key, value in kwargs.items():
            if hasattr(self._net_config, key):
                setattr(self._net_config, key, value)
        return self
    
    def with_lif(self, **kwargs) -> "ConfigBuilder":
        """Update LIF neuron configuration."""
        for key, value in kwargs.items():
            if hasattr(self._lif_config, key):
                setattr(self._lif_config, key, value)
        return self
    
    def with_hh(self, **kwargs) -> "ConfigBuilder":
        """Update Hodgkin-Huxley configuration."""
        for key, value in kwargs.items():
            if hasattr(self._hh_config, key):
                setattr(self._hh_config, key, value)
        return self
    
    def with_stdp(self, **kwargs) -> "ConfigBuilder":
        """Update STDP configuration."""
        for key, value in kwargs.items():
            if hasattr(self._stdp_config, key):
                setattr(self._stdp_config, key, value)
        return self
    
    def with_reservoir(self, **kwargs) -> "ConfigBuilder":
        """Update reservoir configuration."""
        for key, value in kwargs.items():
            if hasattr(self._reservoir_config, key):
                setattr(self._reservoir_config, key, value)
        return self
    
    def with_hardware(self, **kwargs) -> "ConfigBuilder":
        """Update hardware configuration."""
        for key, value in kwargs.items():
            if hasattr(self._hardware_config, key):
                setattr(self._hardware_config, key, value)
        return self
    
    def with_monitoring(self, **kwargs) -> "ConfigBuilder":
        """Update monitoring configuration."""
        for key, value in kwargs.items():
            if hasattr(self._monitoring_config, key):
                setattr(self._monitoring_config, key, value)
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build and return all configurations."""
        return {
            "simulation": self._sim_config,
            "network": self._net_config,
            "lif": self._lif_config,
            "hh": self._hh_config,
            "stdp": self._stdp_config,
            "reservoir": self._reservoir_config,
            "hardware": self._hardware_config,
            "monitoring": self._monitoring_config,
        }
