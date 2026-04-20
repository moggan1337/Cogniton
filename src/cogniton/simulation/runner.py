"""
Simulation Runner Module
========================

High-level simulation control and benchmarking utilities.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
import time
import numpy as np

from cogniton.core.config import SimulationConfig, NetworkConfig, LIFConfig, SimulationMode
from cogniton.core.time import SimulationTime
from cogniton.network.network import SpikingNeuralNetwork
from cogniton.neurons.lif import LIFNeuronGroup
from cogniton.reservoir.reservoir import EchoStateNetwork, LiquidStateMachine, ReservoirConfig


@dataclass
class BenchmarkResult:
    """
    Results from a simulation benchmark.
    
    Attributes:
        name: Benchmark name
        simulation_time: Simulated time (seconds)
        wall_time: Wall clock time (seconds)
        real_time_factor: Simulation speed ratio
        spikes_per_second: Average spike rate
        neurons: Number of neurons
        synapses: Number of synapses
        memory_usage: Estimated memory usage (bytes)
    """
    name: str
    simulation_time: float
    wall_time: float
    real_time_factor: float
    spikes_per_second: float
    neurons: int
    synapses: int
    memory_usage: int = 0
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Get formatted summary."""
        return (
            f"Benchmark: {self.name}\n"
            f"  Simulation time: {self.simulation_time:.3f}s\n"
            f"  Wall time: {self.wall_time:.3f}s\n"
            f"  Real-time factor: {self.real_time_factor:.2f}x\n"
            f"  Spikes/sec: {self.spikes_per_second:.0f}\n"
            f"  Neurons: {self.neurons}\n"
            f"  Synapses: {self.synapses}"
        )


class SimulationRunner:
    """
    High-level simulation runner with benchmarking and monitoring.
    
    Provides a unified interface for running simulations with
    progress tracking, benchmarking, and result collection.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize simulation runner.
        
        Args:
            config: Simulation configuration
        """
        self.config = config or SimulationConfig()
        self._network: Optional[SpikingNeuralNetwork] = None
        self._esn: Optional[EchoStateNetwork] = None
        self._lsm: Optional[LiquidStateMachine] = None
        
        # Monitoring
        self._callbacks: List[Callable] = []
        self._probes: Dict[str, List[float]] = {}
        
        # Benchmark results
        self._results: List[BenchmarkResult] = []
    
    def create_snn(self, network_config: Optional[NetworkConfig] = None,
                   lif_config: Optional[LIFConfig] = None) -> SpikingNeuralNetwork:
        """
        Create spiking neural network.
        
        Args:
            network_config: Network topology config
            lif_config: LIF neuron config
            
        Returns:
            Created network
        """
        self._network = SpikingNeuralNetwork(
            network_id=0,
            network_config=network_config,
            lif_config=lif_config,
            sim_config=self.config
        )
        return self._network
    
    def create_esn(self, reservoir_config: Optional[ReservoirConfig] = None,
                  seed: Optional[int] = None) -> EchoStateNetwork:
        """
        Create Echo State Network.
        
        Args:
            reservoir_config: Reservoir config
            seed: Random seed
            
        Returns:
            Created ESN
        """
        self._esn = EchoStateNetwork(reservoir_config, seed)
        return self._esn
    
    def create_lsm(self, reservoir_config: Optional[ReservoirConfig] = None,
                  seed: Optional[int] = None) -> LiquidStateMachine:
        """
        Create Liquid State Machine.
        
        Args:
            reservoir_config: Reservoir config
            seed: Random seed
            
        Returns:
            Created LSM
        """
        self._lsm = LiquidStateMachine(reservoir_config, seed)
        return self._lsm
    
    def add_progress_callback(self, callback: Callable[[float], None]) -> None:
        """
        Add progress callback.
        
        Args:
            callback: Function called with progress (0-1)
        """
        self._callbacks.append(callback)
    
    def add_probe(self, name: str) -> None:
        """
        Add a probe for monitoring.
        
        Args:
            name: Probe name
        """
        self._probes[name] = []
    
    def record_probe(self, name: str, value: float) -> None:
        """Record probe value."""
        if name not in self._probes:
            self._probes[name] = []
        self._probes[name].append(value)
    
    def run_snn(self, t_end: Optional[float] = None,
                progress: bool = True) -> SpikingNeuralNetwork:
        """
        Run spiking neural network simulation.
        
        Args:
            t_end: End time
            progress: Show progress
            
        Returns:
            The network (for chaining)
        """
        if self._network is None:
            raise RuntimeError("No network created. Call create_snn() first.")
        
        start_time = time.perf_counter()
        
        self._network.run(t_end=t_end, progress=progress)
        
        wall_time = time.perf_counter() - start_time
        
        # Collect benchmark data
        stats = self._network.get_statistics()
        
        result = BenchmarkResult(
            name="SNN Simulation",
            simulation_time=t_end or self.config.t_max,
            wall_time=wall_time,
            real_time_factor=(t_end or self.config.t_max) / wall_time if wall_time > 0 else 0,
            spikes_per_second=stats["total_spikes"] / wall_time if wall_time > 0 else 0,
            neurons=stats["neuron_count"],
            synapses=stats["connectivity"]["synapses"],
        )
        self._results.append(result)
        
        if progress:
            print(result.summary())
        
        return self._network
    
    def run_esn(self, input_sequence: np.ndarray,
               burn_in: int = 50,
               progress: bool = True) -> np.ndarray:
        """
        Run Echo State Network.
        
        Args:
            input_sequence: Input time series
            burn_in: Burn-in steps
            progress: Show progress
            
        Returns:
            Output time series
        """
        if self._esn is None:
            raise RuntimeError("No ESN created. Call create_esn() first.")
        
        start_time = time.perf_counter()
        
        outputs = self._esn.run(input_sequence, burn_in=burn_in)
        
        wall_time = time.perf_counter() - start_time
        
        result = BenchmarkResult(
            name="ESN Simulation",
            simulation_time=len(input_sequence) * self.config.dt,
            wall_time=wall_time,
            real_time_factor=len(input_sequence) * self.config.dt / wall_time if wall_time > 0 else 0,
            spikes_per_second=0,  # ESN doesn't have spikes
            neurons=self._esn.config.num_reservoir,
            synapses=0,
        )
        self._results.append(result)
        
        if progress:
            print(result.summary())
        
        return outputs
    
    def run_benchmark_snn(self, n_neurons: int = 1000,
                         connection_prob: float = 0.1,
                         duration: float = 1.0) -> BenchmarkResult:
        """
        Run SNN benchmark with specified parameters.
        
        Args:
            n_neurons: Number of neurons
            connection_prob: Connection probability
            duration: Simulation duration
            
        Returns:
            Benchmark result
        """
        net_config = NetworkConfig(
            num_neurons=n_neurons,
            num_inputs=int(n_neurons * 0.1),
            num_outputs=int(n_neurons * 0.01),
            connection_probability=connection_prob,
        )
        
        self.create_snn(net_config)
        
        # Inject some input
        for _ in range(10):
            self._network.inject_poisson_input(
                rate=10.0,
                duration=0.1
            )
        
        start_time = time.perf_counter()
        self._network.run(t_end=duration, progress=False)
        wall_time = time.perf_counter() - start_time
        
        stats = self._network.get_statistics()
        
        result = BenchmarkResult(
            name=f"SNN Benchmark ({n_neurons} neurons)",
            simulation_time=duration,
            wall_time=wall_time,
            real_time_factor=duration / wall_time if wall_time > 0 else 0,
            spikes_per_second=stats["total_spikes"] / wall_time if wall_time > 0 else 0,
            neurons=n_neurons,
            synapses=stats["connectivity"]["synapses"],
        )
        self._results.append(result)
        
        return result
    
    def run_scaling_benchmark(self, neuron_counts: List[int],
                             duration: float = 0.5) -> List[BenchmarkResult]:
        """
        Run scaling benchmark across different network sizes.
        
        Args:
            neuron_counts: List of neuron counts to test
            duration: Simulation duration per test
            
        Returns:
            List of benchmark results
        """
        results = []
        
        print("Running scaling benchmark...")
        for n in neuron_counts:
            print(f"\nTesting with {n} neurons...")
            result = self.run_benchmark_snn(
                n_neurons=n,
                duration=duration
            )
            results.append(result)
            print(f"  Wall time: {result.wall_time:.3f}s")
        
        return results
    
    def run_performance_benchmark(self) -> Dict[str, Any]:
        """
        Run comprehensive performance benchmark suite.
        
        Returns:
            Dictionary of benchmark results
        """
        benchmarks = {}
        
        # Small network
        print("Benchmark: Small Network (100 neurons)")
        small = self.run_benchmark_snn(n_neurons=100, duration=1.0)
        benchmarks["small"] = small
        
        # Medium network
        print("\nBenchmark: Medium Network (1000 neurons)")
        medium = self.run_benchmark_snn(n_neurons=1000, duration=1.0)
        benchmarks["medium"] = medium
        
        # Large network
        print("\nBenchmark: Large Network (5000 neurons)")
        large = self.run_benchmark_snn(n_neurons=5000, duration=0.5)
        benchmarks["large"] = large
        
        return benchmarks
    
    def get_results(self) -> List[BenchmarkResult]:
        """Get all benchmark results."""
        return self._results.copy()
    
    def get_probe_data(self, name: str) -> np.ndarray:
        """Get recorded probe data."""
        if name not in self._probes:
            return np.array([])
        return np.array(self._probes[name])
    
    def clear_results(self) -> None:
        """Clear all results and probes."""
        self._results.clear()
        self._probes.clear()
    
    def __repr__(self) -> str:
        return f"SimulationRunner(results={len(self._results)})"
