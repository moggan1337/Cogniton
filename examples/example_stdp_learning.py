#!/usr/bin/env python3
"""
STDP Learning Example
=====================

Demonstrates Spike-Timing-Dependent Plasticity learning in action.
"""

import numpy as np
import matplotlib.pyplot as plt

from cogniton.synapses.plasticity import STDPPlasticity, STDPSynapse
from cogniton.synapses.synapse import Synapse, SynapseGroup
from cogniton.core.config import STDPConfig
from cogniton.utils.visualization import plot_stdp_curve


def example_stdp_curve():
    """Plot the STDP learning window."""
    print("Example: STDP Learning Window")
    print("-" * 40)
    
    from cogniton.utils.visualization import plot_stdp_curve
    fig, ax = plot_stdp_curve()
    plt.show()


def example_stdp_synapse_pairing():
    """Simulate pre-post spike pairing and weight changes."""
    print("\nExample: STDP Spike Pairing")
    print("-" * 40)
    
    # Create STDP synapse
    config = STDPConfig(
        tau_plus=20e-3,
        tau_minus=20e-3,
        a_plus=0.01,
        a_minus=0.012,
        w_max=1.0,
        w_min=0.0,
    )
    
    synapse = STDPSynapse(
        synapse_id=0,
        pre_neuron_id=0,
        post_neuron_id=1,
        weight=0.5,
        stdp_config=config,
    )
    
    # Test different timing differences
    delta_ts = np.linspace(-50e-3, 50e-3, 100) * 1000  # ms
    weight_changes = []
    
    for dt in delta_ts:
        # Reset weight
        synapse.weight = 0.5
        
        # Simulate spike pair
        pre_time = 0.0
        post_time = dt * 1e-3  # Convert back to seconds
        
        initial_weight = synapse.weight
        synapse.apply_plasticity(pre_time, post_time)
        weight_changes.append(synapse.weight - initial_weight)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(delta_ts, np.array(weight_changes) * 1000, 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    ax.fill_between(delta_ts[delta_ts > 0], 0, 
                    np.array(weight_changes)[delta_ts > 0] * 1000,
                    alpha=0.3, color='blue', label='Potentiation')
    ax.fill_between(delta_ts[delta_ts < 0], 0,
                    np.array(weight_changes)[delta_ts < 0] * 1000,
                    alpha=0.3, color='red', label='Depression')
    
    ax.set_xlabel('Δt = t_post - t_pre (ms)')
    ax.set_ylabel('Δw (×10⁻³)')
    ax.set_title('STDP Weight Change')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def example_network_stdp():
    """Simulate STDP in a small network."""
    print("\nExample: Network with STDP")
    print("-" * 40)
    
    from cogniton.neurons.lif import LIFNeuronGroup
    from cogniton.core.config import LIFConfig
    
    # Network parameters
    n_exc = 80
    n_inh = 20
    n_total = n_exc + n_inh
    
    # Create neurons
    lif_config = LIFConfig(
        tau_mem=20e-3,
        v_thresh=-55.0,
        v_reset=-70.0,
    )
    
    exc_group = LIFNeuronGroup(group_id=0, n=n_exc, config=lif_config)
    inh_group = LIFNeuronGroup(group_id=1, n=n_inh, config=lif_config)
    
    # Create synapses with STDP
    stdp_config = STDPConfig(w_max=1.0, w_init=0.3)
    
    synapses = []
    for i in range(n_exc):
        for j in range(n_exc, n_total):
            if np.random.rand() < 0.1:  # 10% connectivity
                syn = STDPSynapse(
                    synapse_id=len(synapses),
                    pre_neuron_id=i,
                    post_neuron_id=j,
                    weight=0.3,
                    stdp_config=stdp_config,
                )
                synapses.append(syn)
    
    # Simulation
    t_max = 2.0
    dt = 1e-4
    t = 0.0
    
    spike_times = []
    weight_history = []
    
    # Record initial weights
    initial_weights = [s.weight for s in synapses]
    
    while t < t_max:
        # Poisson input
        for neuron in exc_group.neurons:
            if np.random.rand() < 0.01:  # ~10 Hz
                neuron.inject_current(1.0)
        
        # Update neurons
        exc_group.update(t, dt)
        inh_group.update(t, dt)
        
        # Record spikes
        for neuron in exc_group.neurons:
            for st in neuron.spike_times[-1:]:
                spike_times.append((st, neuron.neuron_id))
        
        # Apply STDP for recent spikes
        for syn in synapses:
            pre_spikes = [st for st, nid in spike_times if nid == syn.pre_neuron_id]
            post_spikes = [st for st, nid in spike_times if nid == syn.post_neuron_id]
            
            for pt in pre_spikes[-5:]:  # Last 5 pre spikes
                for po in post_spikes[-5:]:  # Last 5 post spikes
                    if abs(pt - po) < 100e-3:  # Within 100ms window
                        syn.apply_plasticity(pt, po)
        
        t += dt
    
    # Statistics
    final_weights = [s.weight for s in synapses]
    weight_change = np.array(final_weights) - np.array(initial_weights)
    
    print(f"Initial weight mean: {np.mean(initial_weights):.4f}")
    print(f"Final weight mean: {np.mean(final_weights):.4f}")
    print(f"Weight change: {np.mean(weight_change):.4f}")
    print(f"Max potentiation: {np.max(weight_change):.4f}")
    print(f"Max depression: {np.min(weight_change):.4f}")
    
    # Plot weight change distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.hist(weight_change, bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='r', linestyle='--')
    ax1.set_xlabel('Weight Change')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of STDP Weight Changes')
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(initial_weights, final_weights, alpha=0.5)
    ax2.plot([0, 1], [0, 1], 'r--', label='No change')
    ax2.set_xlabel('Initial Weight')
    ax2.set_ylabel('Final Weight')
    ax2.set_title('Initial vs Final Weights')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    example_stdp_curve()
    example_stdp_synapse_pairing()
    example_network_stdp()
