#!/usr/bin/env python3
"""
Basic LIF Neuron Example
========================

Demonstrates basic usage of Leaky Integrate-and-Fire neurons.
"""

import numpy as np
import matplotlib.pyplot as plt

from cogniton.neurons.lif import LIFNeuron, LIFNeuronGroup
from cogniton.core.config import LIFConfig
from cogniton.utils.visualization import plot_voltage, plot_raster


def example_single_neuron():
    """Simulate a single LIF neuron with injected current."""
    print("Example: Single LIF Neuron")
    print("-" * 40)
    
    # Create LIF neuron
    config = LIFConfig(
        tau_mem=20e-3,     # 20 ms
        v_thresh=-55.0,    # mV
        v_reset=-75.0,     # mV
        v_rest=-70.0,     # mV
    )
    
    neuron = LIFNeuron(neuron_id=0, config=config)
    
    # Simulation parameters
    t_max = 0.5  # seconds
    dt = 1e-4    # 0.1 ms
    t = 0.0
    
    # Inject current step at t=0.1s
    neuron.inject_current(1.0)  # 1 nA
    
    # Run simulation
    times = []
    voltages = []
    
    while t < t_max:
        neuron.update(t, dt)
        
        times.append(t)
        voltages.append(neuron.v)
        
        t += dt
    
    # Plot results
    times = np.array(times)
    voltages = np.array(voltages)
    
    fig, ax = plot_voltage(
        times, voltages,
        v_thresh=config.v_thresh,
        v_reset=config.v_reset
    )
    ax.set_title("Single LIF Neuron Response")
    plt.show()
    
    print(f"Total spikes: {len(neuron.spike_times)}")
    print(f"Firing rate: {neuron.get_firing_rate():.1f} Hz")


def example_neuron_group():
    """Simulate a group of LIF neurons with Poisson input."""
    print("\nExample: LIF Neuron Group")
    print("-" * 40)
    
    # Create neuron group
    n_neurons = 100
    group = LIFNeuronGroup(group_id=0, n=n_neurons)
    
    # Simulation parameters
    t_max = 1.0
    dt = 1e-4
    t = 0.0
    
    # Inject Poisson input to all neurons
    poisson_rate = 10.0  # Hz
    
    # Run simulation
    while t < t_max:
        # Generate Poisson spikes
        if np.random.rand() < poisson_rate * dt:
            # Select random neuron to spike
            neuron_idx = np.random.randint(0, n_neurons)
            group.neurons[neuron_idx].inject_current(0.5)
        
        group.update(t, dt)
        t += dt
    
    # Get spike raster
    spike_times, spike_ids = [], []
    for neuron in group.neurons:
        for st in neuron.spike_times:
            spike_times.append(st)
            spike_ids.append(neuron.neuron_id)
    
    # Plot raster
    if spike_times:
        spike_times = np.array(spike_times)
        spike_ids = np.array(spike_ids)
        
        fig, ax = plot_raster(spike_times, spike_ids, markersize=2)
        ax.set_title(f"Spike Raster ({n_neurons} neurons)")
        plt.show()
    
    # Statistics
    rates = group.get_firing_rates()
    print(f"Mean firing rate: {np.mean(rates):.2f} Hz")
    print(f"Max firing rate: {np.max(rates):.2f} Hz")
    print(f"Total spikes: {len(spike_times)}")


def example_current_injection():
    """Test different current injection patterns."""
    print("\nExample: Current Injection Patterns")
    print("-" * 40)
    
    # Create neuron
    neuron = LIFNeuron(neuron_id=0)
    
    t_max = 0.5
    dt = 1e-4
    t = 0.0
    
    # Pattern: ramp current
    results = []
    
    while t < t_max:
        # Current ramps from 0 to 2 nA
        current = 4.0 * t  # Ramp
        neuron.inject_current(current * dt / 1e-4)  # Normalize
        
        neuron.update(t, dt)
        results.append((t, neuron.v, current))
        
        t += dt
    
    times = np.array([r[0] for r in results])
    voltages = np.array([r[1] for r in results])
    currents = np.array([r[2] for r in results])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    
    ax1.plot(times * 1000, voltages)
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_title('Ramp Current Injection')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(times * 1000, currents)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Current (nA)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Spikes fired: {len(neuron.spike_times)}")
    print(f"Firing rate: {neuron.get_firing_rate():.1f} Hz")


if __name__ == "__main__":
    example_single_neuron()
    example_neuron_group()
    example_current_injection()
