"""
Spiking Neural Network Module
==============================

Implements complete spiking neural networks with LIF and HH neurons,
synaptic connectivity, and event-driven simulation.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
import numpy as np

from cogniton.core.config import NetworkConfig, LIFConfig, HHConfig, STDPConfig, SimulationConfig
from cogniton.core.event import Event, EventQueue, EventType, EventDrivenSimulation
from cogniton.core.time import SimulationTime
from cogniton.neurons.lif import LIFNeuron, LIFNeuronGroup
from cogniton.neurons.hh import HodgkinHuxleyNeuron, HodgkinHuxleyGroup
from cogniton.synapses.synapse import Synapse, SynapseGroup, ConnectionType
from cogniton.synapses.plasticity import STDPPlasticity, SynapticPlasticity


class SpikingNeuralNetwork:
    """
    Complete Spiking Neural Network.
    
    Integrates neurons, synapses, and plasticity into a unified
    simulation framework with support for both LIF and HH neuron models.
    
    Attributes:
        network_id: Network identifier
        config: Network configuration
        lif_config: LIF neuron configuration
        hh_config: Hodgkin-Huxley configuration
        stdp_config: STDP configuration
        neurons: Dictionary of neuron groups
        synapses: Dictionary of synapse groups
        sim_time: Simulation time manager
        simulation: Event-driven simulation engine
    """
    
    def __init__(self, network_id: int,
                 network_config: Optional[NetworkConfig] = None,
                 lif_config: Optional[LIFConfig] = None,
                 hh_config: Optional[HHConfig] = None,
                 stdp_config: Optional[STDPConfig] = None,
                 sim_config: Optional[SimulationConfig] = None):
        """
        Initialize spiking neural network.
        
        Args:
            network_id: Network identifier
            network_config: Network topology configuration
            lif_config: LIF neuron parameters
            hh_config: Hodgkin-Huxley parameters
            stdp_config: STDP plasticity parameters
            sim_config: Simulation parameters
        """
        self.network_id = network_id
        
        # Configurations
        self.config = network_config or NetworkConfig()
        self.lif_config = lif_config or LIFConfig()
        self.hh_config = hh_config or HHConfig()
        self.stdp_config = stdp_config or STDPConfig()
        self.sim_config = sim_config or SimulationConfig()
        
        # Neuron groups
        self.neuron_groups: Dict[str, LIFNeuronGroup] = {}
        self.hh_groups: Dict[str, HodgkinHuxleyGroup] = {}
        self._neuron_to_group: Dict[int, Tuple[str, int]] = {}  # neuron_id -> (group_name, idx)
        
        # Synapses
        self.synapse_group = SynapseGroup(group_id=0)
        self._plasticity: Optional[SynapticPlasticity] = None
        
        # Simulation
        self.sim_time = SimulationTime(
            t_end=self.sim_config.t_max,
            dt=self.sim_config.dt
        )
        self.event_sim = EventDrivenSimulation(
            t_max=self.sim_config.t_max,
            dt=self.sim_config.dt,
            record_events=self.sim_config.record_spikes
        )
        
        # Statistics
        self._total_spikes = 0
        self._spike_times: List[Tuple[float, int]] = []
        
        # Build network
        self._build_network()
    
    def _build_network(self) -> None:
        """Build network topology."""
        # Create input layer
        self.neuron_groups["input"] = LIFNeuronGroup(
            group_id=0,
            n=self.config.num_inputs,
            config=self.lif_config
        )
        
        # Create excitatory and inhibitory populations
        n_exc = int(self.config.num_neurons * self.config.excitatory_ratio)
        n_inh = self.config.num_neurons - n_exc
        
        self.neuron_groups["excitatory"] = LIFNeuronGroup(
            group_id=1,
            n=n_exc,
            config=self.lif_config
        )
        
        self.neuron_groups["inhibitory"] = LIFNeuronGroup(
            group_id=2,
            n=n_inh,
            config=self.lif_config
        )
        
        # Create output layer
        self.neuron_groups["output"] = LIFNeuronGroup(
            group_id=3,
            n=self.config.num_outputs,
            config=self.lif_config
        )
        
        # Build connectivity
        self._build_connectivity()
        
        # Register with event simulation
        self._register_with_simulation()
    
    def _build_connectivity(self) -> None:
        """Build synaptic connectivity."""
        rng = np.random.RandomState(self.config.seed)
        
        n_total = self.config.num_neurons
        all_neurons = list(range(n_total))
        
        # Random connectivity
        for pre in range(n_total):
            for post in range(n_total):
                if pre == post:
                    continue
                    
                if rng.rand() < self.config.connection_probability:
                    # Determine type
                    is_excitatory = (pre < self.config.num_inputs or 
                                   self.config.excitatory_ratio > 0.5)
                    
                    # Sample weight
                    w_mean = self.config.initial_weight_mean
                    w_std = self.config.initial_weight_std
                    w = max(0, rng.randn() * w_std + w_mean)
                    
                    if not is_excitatory:
                        w = -w
                    
                    # Sample delay
                    d_mean = self.config.delay_mean
                    d_std = self.config.delay_std
                    d = abs(rng.randn() * d_std + d_mean)
                    
                    synapse = Synapse(
                        synapse_id=len(self.synapse_group),
                        pre_neuron_id=pre,
                        post_neuron_id=post,
                        weight=w,
                        delay=d,
                        connection_type=(ConnectionType.EXCITATORY if is_excitatory 
                                       else ConnectionType.INHIBITORY)
                    )
                    self.synapse_group.add_synapse(synapse)
    
    def _register_with_simulation(self) -> None:
        """Register components with event simulation."""
        for group_name, group in self.neuron_groups.items():
            for neuron in group.neurons:
                self.event_sim.register_neuron(neuron.neuron_id, neuron)
    
    def set_stdp(self, enabled: bool = True) -> None:
        """
        Enable or disable STDP plasticity.
        
        Args:
            enabled: Whether to enable STDP
        """
        if enabled:
            self._plasticity = STDPPlasticity(self.stdp_config)
        else:
            self._plasticity = None
    
    def inject_poisson_input(self, rate: float, 
                            duration: float,
                            weights: Optional[np.ndarray] = None) -> None:
        """
        Inject Poisson spike trains as input.
        
        Args:
            rate: Firing rate (Hz)
            duration: Duration (seconds)
            weights: Optional custom weights for each input
        """
        t = 0.0
        dt = 0.1e-3  # 0.1 ms bins
        
        while t < duration:
            for i in range(self.config.num_inputs):
                if np.random.rand() < rate * dt:
                    # Emit spike
                    synapses = self.synapse_group.get_outgoing_synapses(i)
                    w = weights[i] if weights is not None else 0.5
                    
                    for syn in synapses:
                        spike_event = Event(
                            time=t + syn.delay,
                            event_type=EventType.SPIKE,
                            source_id=i,
                            target_id=syn.post_neuron_id,
                            data={"weight": syn.weight * w}
                        )
                        self.event_sim.schedule_event(spike_event)
            
            t += dt
    
    def inject_current(self, neuron_id: int, current: float) -> None:
        """
        Inject constant current into neuron.
        
        Args:
            neuron_id: Target neuron ID
            current: Current magnitude (nA)
        """
        for group in self.neuron_groups.values():
            for neuron in group.neurons:
                if neuron.neuron_id == neuron_id:
                    neuron.inject_current(current)
                    return
    
    def inject_ramp_current(self, neuron_id: int, start_current: float,
                           end_current: float, duration: float) -> None:
        """
        Inject ramping current.
        
        Args:
            neuron_id: Target neuron ID
            start_current: Starting current
            end_current: Ending current
            duration: Duration (seconds)
        """
        dt = 1e-4
        t = 0.0
        current = start_current
        
        while t < duration:
            for group in self.neuron_groups.values():
                for neuron in group.neurons:
                    if neuron.neuron_id == neuron_id:
                        neuron.inject_current(current)
            
            current = start_current + (end_current - start_current) * (t / duration)
            t += dt
    
    def step(self, dt: Optional[float] = None) -> int:
        """
        Execute single simulation step.
        
        Args:
            dt: Timestep (uses config if None)
            
        Returns:
            Number of spikes in this step
        """
        if dt is None:
            dt = self.sim_config.dt
        
        t = self.sim_time.t
        spikes = 0
        
        # Update each neuron group
        for group_name, group in self.neuron_groups.items():
            spiked = group.update(t, dt)
            spikes += np.sum(spiked)
            
            # Process spikes
            for spike_time, neuron_id in group.spike_raster:
                if spike_time >= t and spike_time < t + dt:
                    self._process_spike(neuron_id, spike_time)
        
        # Update simulation time
        self.sim_time.advance(dt)
        
        return spikes
    
    def _process_spike(self, neuron_id: int, spike_time: float) -> None:
        """Process spike event."""
        self._total_spikes += 1
        self._spike_times.append((spike_time, neuron_id))
        
        # Get outgoing synapses
        synapses = self.synapse_group.get_outgoing_synapses(neuron_id)
        
        for syn in synapses:
            # Deliver to postsynaptic neuron
            weight = syn.pre_spike(spike_time)
            
            spike_event = Event(
                time=spike_time + syn.delay,
                event_type=EventType.SPIKE,
                source_id=neuron_id,
                target_id=syn.post_neuron_id,
                data={"weight": weight, "synapse_id": syn.synapse_id}
            )
            self.event_sim.schedule_event(spike_event)
            
            # Apply STDP if enabled
            if self._plasticity is not None and syn.model.value == "stdp":
                self._plasticity.record_pre_spike(syn.synapse_id, spike_time)
    
    def run(self, t_end: Optional[float] = None,
            progress: bool = True) -> None:
        """
        Run simulation until end time.
        
        Args:
            t_end: End time (uses config if None)
            progress: Whether to show progress
        """
        if t_end is not None:
            self.sim_time.t_end = t_end
        elif self.sim_config.t_max:
            self.sim_time.t_end = self.sim_config.t_max
        
        if progress:
            print(f"Running SNN simulation: {self.sim_time.t_end}s")
        
        while not self.sim_time.is_finished:
            spikes = self.step()
            
            if progress and self.sim_time.t % 0.1 < self.sim_config.dt:
                rate = self.get_population_rate()
                print(f"  t={self.sim_time.t:.3f}s, "
                      f"spikes={self._total_spikes}, "
                      f"rate={rate:.1f}Hz")
        
        if progress:
            print(f"Simulation complete: {self._total_spikes} total spikes")
    
    def get_population_rate(self, window: float = 0.1) -> float:
        """
        Get average population firing rate.
        
        Args:
            window: Time window (seconds)
            
        Returns:
            Firing rate in Hz
        """
        t = self.sim_time.t
        t_start = max(0, t - window)
        
        spikes_in_window = sum(1 for st, _ in self._spike_times 
                              if t_start <= st <= t)
        
        n_neurons = sum(len(g) for g in self.neuron_groups.values())
        return spikes_in_window / (window * n_neurons)
    
    def get_firing_rates(self) -> Dict[str, np.ndarray]:
        """Get firing rates for each group."""
        rates = {}
        for name, group in self.neuron_groups.items():
            rates[name] = group.get_firing_rates()
        return rates
    
    def get_weight_matrix(self) -> np.ndarray:
        """Get full weight matrix."""
        n_neurons = sum(len(g) for g in self.neuron_groups.values())
        return self.synapse_group.get_weight_matrix(n_neurons)
    
    def get_connectivity_matrix(self) -> np.ndarray:
        """Get binary connectivity matrix."""
        return (self.get_weight_matrix() != 0).astype(int)
    
    def get_raster_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get spike raster data.
        
        Returns:
            Tuple of (times, neuron_ids)
        """
        if not self._spike_times:
            return np.array([]), np.array([])
        
        times = np.array([t for t, _ in self._spike_times])
        ids = np.array([n for _, n in self._spike_times])
        return times, ids
    
    def reset(self) -> None:
        """Reset network to initial state."""
        self._total_spikes = 0
        self._spike_times.clear()
        
        for group in self.neuron_groups.values():
            group.reset()
        
        self.synapse_group.reset()
        self.sim_time.reset()
        self.event_sim.reset()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get network statistics."""
        rates = self.get_firing_rates()
        
        return {
            "total_spikes": self._total_spikes,
            "simulation_time": self.sim_time.t,
            "population_rates": {name: np.mean(r) for name, r in rates.items()},
            "synapse_count": len(self.synapse_group),
            "neuron_count": sum(len(g) for g in self.neuron_groups.values()),
            "connectivity": {
                "synapses": len(self.synapse_group),
                "probability": self.config.connection_probability,
                "mean_weight": np.mean(self.synapse_group.get_weights()),
                "std_weight": np.std(self.synapse_group.get_weights()),
            }
        }
    
    def save_state(self, filepath: str) -> None:
        """Save network state to file."""
        import pickle
        state = {
            "config": self.config,
            "lif_config": self.lif_config,
            "hh_config": self.hh_config,
            "stdp_config": self.stdp_config,
            "spike_times": self._spike_times.copy(),
            "weights": self.synapse_group.get_weights(),
            "sim_time": self.sim_time.t,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, filepath: str) -> None:
        """Load network state from file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state["config"]
        self.lif_config = state["lif_config"]
        self.hh_config = state["hh_config"]
        self.stdp_config = state["stdp_config"]
        self._spike_times = state["spike_times"]
        self.synapse_group.set_weights(state["weights"])
        self.sim_time.set_time(state["sim_time"])
    
    def __repr__(self) -> str:
        n_total = sum(len(g) for g in self.neuron_groups.values())
        return (f"SpikingNeuralNetwork(id={self.network_id}, "
                f"neurons={n_total}, "
                f"synapses={len(self.synapse_group)}, "
                f"spikes={self._total_spikes})")
