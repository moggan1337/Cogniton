#!/usr/bin/env python3
"""
Reservoir Computing Example
===========================

Demonstrates Echo State Networks for time series prediction.
"""

import numpy as np
import matplotlib.pyplot as plt

from cogniton.reservoir.reservoir import EchoStateNetwork, LiquidStateMachine, ReservoirConfig
from cogniton.utils.visualization import plot_voltage


def example_esn_mackey_glass():
    """Predict Mackey-Glass chaotic time series with ESN."""
    print("Example: ESN for Mackey-Glass Prediction")
    print("-" * 40)
    
    # Generate Mackey-Glass time series
    def mackey_glass(n_points, tau=17, dt=0.1, seed=42):
        """Generate Mackey-Glass chaotic time series."""
        np.random.seed(seed)
        
        x = np.zeros(n_points)
        x[0] = 1.5
        
        for t in range(1, n_points):
            x[t] = x[t-1] + dt * (0.2 * x[t-tau] / (1 + x[t-tau]**10) - 0.1 * x[t-1])
        
        return x
    
    # Parameters
    n_train = 2000
    n_test = 500
    n_total = n_train + n_test
    
    # Generate data
    data = mackey_glass(n_total + 100)[100:]  # Skip transient
    data = (data - data.mean()) / data.std()  # Normalize
    
    train_data = data[:n_train].reshape(-1, 1)
    test_data = data[n_train:n_train + n_test].reshape(-1, 1)
    
    # Create ESN
    config = ReservoirConfig(
        num_reservoir=400,
        num_inputs=1,
        num_outputs=1,
        spectral_radius=0.9,
        input_scaling=1.0,
        leak_rate=0.3,
    )
    
    esn = EchoStateNetwork(config, seed=42)
    
    print(f"ESN created: {esn}")
    
    # Run training (collect states)
    print("Training ESN...")
    esn.run(train_data, burn_in=100)
    
    # Train readout
    # Target: next step prediction
    targets = train_data[1:]
    states = np.array(esn.states_collected[100:])  # Skip burn-in
    
    esn.train_readout(targets, method='ridge')
    
    # Test prediction
    print("Testing prediction...")
    predictions = esn.predict(test_data)
    
    # Compute error
    mse = np.mean((predictions - test_data)**2)
    rmse = np.sqrt(mse)
    
    print(f"RMSE: {rmse:.4f}")
    
    # Plot results
    time_test = np.arange(n_test) * 0.1  # Assume dt=0.1
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    
    ax1.plot(time_test, test_data, 'b-', label='Target', alpha=0.7)
    ax1.plot(time_test, predictions, 'r-', label='ESN Prediction', alpha=0.7)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.set_title('ESN Mackey-Glass Prediction')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(time_test, test_data - predictions, 'g-', alpha=0.7)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Prediction Error')
    ax2.set_title('Prediction Error')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return rmse


def example_esn_pattern_recognition():
    """ESN for pattern recognition tasks."""
    print("\nExample: ESN for Pattern Recognition")
    print("-" * 40)
    
    # Create ESN
    config = ReservoirConfig(
        num_reservoir=200,
        num_inputs=2,
        num_outputs=3,
        spectral_radius=0.8,
    )
    
    esn = EchoStateNetwork(config, seed=42)
    
    # Generate three pattern classes
    n_samples_per_class = 50
    n_timesteps = 50
    
    patterns = []
    labels = []
    
    for class_id in range(3):
        for _ in range(n_samples_per_class):
            # Generate pattern with class-specific characteristics
            t = np.linspace(0, 2*np.pi, n_timesteps)
            
            if class_id == 0:
                # Sine wave
                pattern = np.sin(t + np.random.randn() * 0.1)
            elif class_id == 1:
                # Square wave
                pattern = np.sign(np.sin(t + np.random.randn() * 0.1))
            else:
                # Sawtooth
                pattern = 2 * (t / (2*np.pi) % 1) - 1
            
            patterns.append(pattern)
            labels.append(class_id)
    
    patterns = np.array(patterns)
    labels = np.array(labels)
    
    # Reshape for ESN (n_samples, timesteps, n_inputs)
    patterns_reshaped = patterns.reshape(n_samples_per_class * 3, n_timesteps, 1)
    # Duplicate for second input
    patterns_2d = np.concatenate([patterns_reshaped, patterns_reshaped * 0.5], axis=2)
    
    # Train ESN on all samples
    for i in range(len(patterns_2d)):
        esn.update(patterns_2d[i], store_state=True)
    
    # Use final state for classification
    states = np.array(esn.states_collected)
    states = states[:, :config.num_reservoir]  # Just reservoir states
    
    # Simple classification using nearest centroid
    centroids = np.zeros((3, config.num_reservoir))
    for c in range(3):
        mask = labels == c
        centroids[c] = np.mean(states[mask], axis=0)
    
    # Predict
    predictions = []
    for i in range(len(states)):
        distances = np.linalg.norm(states[i] - centroids, axis=1)
        pred = np.argmin(distances)
        predictions.append(pred)
    
    accuracy = np.mean(np.array(predictions) == labels)
    print(f"Classification accuracy: {accuracy:.1%}")
    
    return accuracy


def example_lsm_pattern_classification():
    """Liquid State Machine for spatiotemporal pattern classification."""
    print("\nExample: LSM for Pattern Classification")
    print("-" * 40)
    
    # Create LSM
    config = ReservoirConfig(
        num_reservoir=300,
        num_inputs=2,
        num_outputs=2,
        connectivity=0.05,
    )
    
    from cogniton.core.config import LIFConfig
    lif_config = LIFConfig()
    
    lsm = LiquidStateMachine(config, lif_config, seed=42)
    
    print(f"LSM created: {lsm}")
    print(f"Total spikes in liquid: {len(lsm.liquid.spike_raster)}")
    
    # Generate spatiotemporal input
    n_timesteps = 200
    dt = 1e-4
    
    # Pattern 1: Localized input
    input1 = np.zeros((n_timesteps, 2))
    for t in range(50, 100):
        input1[t, 0] = 1.0
    
    # Pattern 2: Distributed input
    input2 = np.zeros((n_timesteps, 2))
    for t in range(50, 150):
        input2[t, :] = 0.5
    
    # Run LSM on both patterns
    print("Running LSM on patterns...")
    
    # Pattern 1
    lsm.reset_state()
    lsm.liquid.reset()
    for t in range(n_timesteps):
        lsm.update(input1[t], dt=dt, store_state=True)
    
    states1 = np.array(lsm.states_collected)
    spikes1 = len(lsm.liquid.spike_raster)
    
    # Pattern 2
    lsm.reset_state()
    lsm.liquid.reset()
    for t in range(n_timesteps):
        lsm.update(input2[t], dt=dt, store_state=True)
    
    states2 = np.array(lsm.states_collected)
    spikes2 = len(lsm.liquid.spike_raster)
    
    print(f"Pattern 1 - States shape: {states1.shape}, Spikes: {spikes1}")
    print(f"Pattern 2 - States shape: {states2.shape}, Spikes: {spikes2}")
    
    # Compute separation
    separation = np.linalg.norm(states1.mean(axis=0) - states2.mean(axis=0))
    print(f"State separation: {separation:.4f}")
    
    return separation


def example_esn_scaling():
    """Test ESN performance across different sizes."""
    print("\nExample: ESN Scaling")
    print("-" * 40)
    
    # Test different reservoir sizes
    sizes = [50, 100, 200, 400, 800]
    times = []
    
    for n in sizes:
        config = ReservoirConfig(
            num_reservoir=n,
            num_inputs=5,
            num_outputs=5,
        )
        
        esn = EchoStateNetwork(config, seed=42)
        
        # Run simulation
        import time
        start = time.perf_counter()
        
        n_steps = 500
        for _ in range(n_steps):
            u = np.random.randn(5)
            esn.update(u, store_state=True)
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        print(f"  N={n}: {elapsed:.4f}s")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(sizes, np.array(times) * 1000, 'bo-')
    ax.set_xlabel('Reservoir Size')
    ax.set_ylabel('Computation Time (ms)')
    ax.set_title('ESN Scaling Performance')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    example_esn_mackey_glass()
    example_esn_pattern_recognition()
    example_lsm_pattern_classification()
    example_esn_scaling()
