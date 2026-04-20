#!/usr/bin/env python3
"""
Network Simulation Example
==========================

Complete example of spiking neural network simulation.
"""

import numpy as np
import matplotlib.pyplot as plt

from cogniton.network.network import SpikingNeuralNetwork
from cogniton.core.config import NetworkConfig, LIFConfig, SimulationConfig
from cogniton.utils.visualization import plot_raster
from cogniton.utils.analysis import compute_raster_statistics


def example_balanced_network():
    """Simulate a balanced excitatory-inhibitory network."""
    print("Example: Balanced E/I Network")
    print("-" * 40)
    
    # Network configuration
    net_config = NetworkConfig(
        num_neurons=500,
        num_inputs=50,
        num_outputs=10,
        connection_probability=0.1,
        excitatory_ratio=0.8,
        initial_weight_mean=0.3,
        initial_weight_std=0.05,
    )
    
    # LIF configuration
    lif_config = LIFConfig(
        tau_mem=20e-3,
        v_thresh=-55.0,
        v_reset=-70.0,
        v_rest=-70.0,
    )
    
    # Create network
    network = SpikingNeuralNetwork(
        network_id=0,
        network_config=net_config,
        lif_config=lif_config,
    )
    
    print(f"Network created: {network}")
    
    # Inject Poisson input
    print("Injecting Poisson input...")
    network.inject_poisson_input(rate=10.0, duration=0.5)
    
    # Run simulation
    print("Running simulation...")
    network.run(t_end=1.0, progress=True)
    
    # Get results
    times, ids = network.get_raster_data()
    
    # Statistics
    stats = network.get_statistics()
    print(f"\nNetwork Statistics:")
    print(f"  Total spikes: {stats['total_spikes']}")
    print(f"  Mean firing rate: {stats['population_rates'].get('excitatory', 0):.1f} Hz")
    
    # Plot raster
    if len(times) > 0:
        fig, ax = plot_raster(times, ids, markersize=1)
        ax.set_title("Balanced Network Spike Raster")
        plt.show()
    
    return stats


def example_ring_network():
    """Simulate a ring network with spatial connectivity."""
    print("\nExample: Ring Network")
    print("-" * 40)
    
    from cogniton.neurons.lif import LIFNeuronGroup
    from cogniton.synapses.synapse import create_random_connectivity
    
    # Create ring network
    n_neurons = 100
    
    # Create neurons
    group = LIFNeuronGroup(group_id=0, n=n_neurons)
    
    # Create spatially structured connectivity
    positions = np.linspace(0, 2*np.pi, n_neurons)
    
    # Connectivity based on distance on ring
    def connectivity_probability(distance, sigma=0.5):
        """Gaussian connectivity profile."""
        return np.exp(-distance**2 / (2 * sigma**2))
    
    synapses = []
    for i in range(n_neurons):
        for j in range(n_neurons):
            if i != j:
                # Distance on ring (shortest path)
                d = min(abs(positions[i] - positions[j]),
                       2*np.pi - abs(positions[i] - positions[j]))
                
                if np.random.rand() < connectivity_probability(d):
                    synapses.append((i, j, 0.3))  # (pre, post, weight)
    
    print(f"Created {len(synapses)} synapses")
    
    # Simulation with external input
    t_max = 2.0
    dt = 1e-4
    t = 0.0
    
    while t < t_max:
        # Propagate activity around the ring
        if int(t * 1000) % 100 == 0:
            # Inject pulse at one location
            pulse_loc = int((np.sin(2 * np.pi * t / 0.5) + 1) * n_neurons / 2) % n_neurons
            group.neurons[pulse_loc].inject_current(2.0)
        
        group.update(t, dt)
        t += dt
    
    # Get spike raster
    spike_times = []
    spike_ids = []
    for neuron in group.neurons:
        for st in neuron.spike_times:
            spike_times.append(st)
            spike_ids.append(neuron.neuron_id)
    
    if spike_times:
        fig, ax = plot_raster(
            np.array(spike_times),
            np.array(spike_ids),
            title="Ring Network Activity"
        )
        plt.show()
    
    return len(spike_times)


def example_hodgkin_huxley():
    """Simulate Hodgkin-Huxley neurons."""
    print("\nExample: Hodgkin-Huxley Neurons")
    print("-" * 40)
    
    from cogniton.neurons.hh import HodgkinHuxleyNeuron
    from cogniton.core.config import HHConfig
    
    # Create HH neuron
    config = HHConfig()
    neuron = HodgkinHuxleyNeuron(neuron_id=0, config=config)
    
    # Simulation
    t_max = 0.5
    dt = 1e-5  # HH needs smaller timestep
    t = 0.0
    
    times = []
    voltages = []
    
    # Inject step current at t=0.1s
    current_amplitude = 10.0  # μA/cm²
    
    while t < t_max:
        # Inject current
        if 0.05 < t < 0.35:
            neuron.inject_current(current_amplitude)
        
        neuron.update(t, dt)
        
        times.append(t)
        voltages.append(neuron.v)
        
        t += dt
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    
    times = np.array(times) * 1000  # Convert to ms
    voltages = np.array(voltages)
    
    ax.plot(times, voltages, 'b-', linewidth=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane Potential (mV)')
    ax.set_title('Hodgkin-Huxley Neuron Response')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Spikes: {len(neuron.spike_times)}")
    
    return len(neuron.spike_times)


def example_stochastic_network():
    """Simulate network with stochastic dynamics."""
    print("\nExample: Stochastic Network")
    print("-" * 40)
    
    from cogniton.network.network import SpikingNeuralNetwork
    from cogniton.core.config import NetworkConfig, LIFConfig
    
    # Small network for stochastic analysis
    net_config = NetworkConfig(
        num_neurons=50,
        num_inputs=10,
        connection_probability=0.2,
    )
    
    network = SpikingNeuralNetwork(
        network_id=0,
        network_config=net_config,
    )
    
    # Run multiple trials
    n_trials = 5
    all_spike_counts = []
    
    for trial in range(n_trials):
        network.reset()
        network.inject_poisson_input(rate=5.0, duration=0.2)
        network.run(t_end=0.5, progress=False)
        
        times, ids = network.get_raster_data()
        all_spike_counts.append(len(times))
        print(f"  Trial {trial + 1}: {len(times)} spikes")
    
    print(f"\nSpike count statistics:")
    print(f"  Mean: {np.mean(all_spike_counts):.1f}")
    print(f"  Std: {np.std(all_spike_counts):.1f}")
    print(f"  CV: {np.std(all_spike_counts) / np.mean(all_spike_counts):.2f}")
    
    return all_spike_counts


if __name__ == "__main__":
    example_balanced_network()
    example_ring_network()
    example_hodgkin_huxley()
    example_stochastic_network()
