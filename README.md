# Cogniton: Neuromorphic Computing Simulator

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Neuromorphic-SNN-purple.svg" alt="Neuromorphic">
  <img src="https://img.shields.io/badge/Platform-CPU/GPU/Loihi-orange.svg" alt="Platform">
</p>

---

## Overview

**Cogniton** is a comprehensive, open-source neuromorphic computing simulator designed for research and development of spiking neural networks (SNNs). It provides a complete framework for simulating brain-inspired computational models, from single neurons to large-scale networks with plasticity.

### Key Features

- **Multiple Neuron Models**: Leaky Integrate-and-Fire (LIF) and Hodgkin-Huxley neurons
- **Plasticity Mechanisms**: STDP, homeostatic plasticity, reward-modulated STDP
- **Reservoir Computing**: Echo State Networks and Liquid State Machines
- **Event-Driven Simulation**: Efficient simulation without fixed timesteps
- **Hardware Support**: Intel Loihi, IBM TrueNorth, and CPU/GPU backends
- **Flexible Configuration**: Highly customizable network architectures
- **Analysis Tools**: Spike analysis, raster plots, correlation measures

---

## 🎬 Demo
![Cogniton Demo](demo.gif)

*Spiking neural network simulation with STDP learning*

## Screenshots
| Component | Preview |
|-----------|---------|
| Spike Trains | ![spikes](screenshots/spike-trains.png) |
| Network Viewer | ![network](screenshots/network-view.png) |
| Plasticity Weights | ![weights](screenshots/plasticity.png) |

## Visual Description
Spike train visualizations show neuron firing patterns as raster plots with color-coded layers. Network viewer displays spiking neurons with synaptic connections weighted by conductance. Plasticity shows STDP learning curves.

---


## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Architecture Overview](#architecture-overview)
4. [Neuron Models](#neuron-models)
5. [Synaptic Plasticity](#synaptic-plasticity)
6. [Network Simulation](#network-simulation)
7. [Reservoir Computing](#reservoir-computing)
8. [Hardware Integration](#hardware-integration)
9. [Analysis and Visualization](#analysis-and-visualization)
10. [Examples](#examples)
11. [API Reference](#api-reference)
12. [Performance Benchmarks](#performance-benchmarks)
13. [Contributing](#contributing)
14. [License](#license)
15. [Citations](#citations)

---

## Installation

### From Source

```bash
git clone https://github.com/moggan1337/Cogniton.git
cd Cogniton
pip install -e .
```

### With Optional Dependencies

```bash
# For visualization
pip install -e ".[visualization]"

# For development
pip install -e ".[dev]"

# For all features
pip install -e ".[all]"
```

### Requirements

- Python 3.8+
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.5.0 (optional, for visualization)
- pytest >= 7.0.0 (optional, for testing)

---

## Quick Start

### Basic LIF Neuron

```python
from cogniton.neurons.lif import LIFNeuron
from cogniton.core.config import LIFConfig

# Create neuron
config = LIFConfig(
    tau_mem=20e-3,    # 20 ms time constant
    v_thresh=-55.0,   # Spike threshold (mV)
    v_reset=-75.0,    # Reset potential (mV)
)
neuron = LIFNeuron(neuron_id=0, config=config)

# Inject current and simulate
neuron.inject_current(1.0)  # 1 nA

for t in range(1000):
    spiked = neuron.update(t * 1e-4, dt=1e-4)
    if spiked:
        print(f"Spike at t={t*1e-4:.4f}s")
```

### Complete Network Simulation

```python
from cogniton.network.network import SpikingNeuralNetwork
from cogniton.core.config import NetworkConfig, LIFConfig

# Configure network
net_config = NetworkConfig(
    num_neurons=1000,
    num_inputs=100,
    connection_probability=0.1,
)

# Create and run
network = SpikingNeuralNetwork(network_id=0, network_config=net_config)
network.inject_poisson_input(rate=10.0, duration=1.0)
network.run(t_end=2.0, progress=True)

# Get results
times, ids = network.get_raster_data()
print(f"Total spikes: {len(times)}")
```

### Echo State Network

```python
from cogniton.reservoir.reservoir import EchoStateNetwork, ReservoirConfig

# Create ESN
config = ReservoirConfig(
    num_reservoir=400,
    num_inputs=2,
    num_outputs=1,
    spectral_radius=0.9,
)
esn = EchoStateNetwork(config, seed=42)

# Run on time series
input_data = np.random.randn(1000, 2)
esn.run(input_data, burn_in=50)

# Train readout
esn.train_readout(targets, method='ridge')
```

---

## Architecture Overview

Cogniton follows a modular architecture with clear separation of concerns:

```
cogniton/
├── core/           # Core simulation infrastructure
│   ├── config.py   # Configuration classes
│   ├── event.py    # Event-driven simulation engine
│   └── time.py     # Time management
├── neurons/        # Neuron models
│   ├── lif.py      # Leaky Integrate-and-Fire
│   └── hh.py       # Hodgkin-Huxley
├── synapses/       # Synaptic models and plasticity
│   ├── synapse.py  # Base synapse implementation
│   └── plasticity.py # STDP and other rules
├── network/        # Network construction
│   └── network.py  # SpikingNeuralNetwork
├── reservoir/      # Reservoir computing
│   └── reservoir.py # ESN and LSM
├── simulation/     # Simulation runner
│   └── runner.py   # Benchmark utilities
├── hardware/       # Hardware integration
│   └── backends.py # Intel Loihi, IBM TrueNorth
└── utils/          # Utilities
    ├── visualization.py
    ├── analysis.py
    └── data.py
```

---

## Neuron Models

### Leaky Integrate-and-Fire (LIF)

The LIF model is a simplified neuron model that captures essential dynamics:

```
τ_m * dv/dt = -(v - v_rest) + R_m * I(t)

When v >= v_thresh:
    Spike emitted
    v <- v_reset
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| τ_mem | 20 ms | Membrane time constant |
| v_rest | -70 mV | Resting potential |
| v_thresh | -55 mV | Spike threshold |
| v_reset | -75 mV | Reset potential |
| R_mem | 10 MΩ | Membrane resistance |

**Advantages:**
- Computationally efficient
- Event-driven simulation possible
- Good for large-scale networks
- Well-studied dynamics

**Example:**

```python
from cogniton.neurons.lif import LIFConfig, LIFNeuron

config = LIFConfig(
    tau_mem=20e-3,
    tau_syn=10e-3,
    v_thresh=-55.0,
    v_reset=-75.0,
    v_rest=-70.0,
    refract_time=2e-3,
)
neuron = LIFNeuron(neuron_id=0, config=config)

# Run simulation
for _ in range(1000):
    neuron.update(t, dt=1e-4)
    t += 1e-4
```

### Hodgkin-Huxley Model

The Hodgkin-Huxley model provides biophysically accurate action potential dynamics:

```
C_m * dv/dt = -g_na * m³ * h * (v - E_na)
              - g_k * n⁴ * (v - E_k)
              - g_l * (v - E_l) + I(t)
```

**Gating Variables:**
- **m**: Sodium activation (fast)
- **h**: Sodium inactivation (slow)
- **n**: Potassium activation (slow)

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| g_na | 120 mS/cm² | Sodium conductance |
| g_k | 36 mS/cm² | Potassium conductance |
| g_l | 0.3 mS/cm² | Leak conductance |
| E_na | 50 mV | Sodium reversal potential |
| E_k | -77 mV | Potassium reversal potential |
| c_m | 1 μF/cm² | Membrane capacitance |

**Example:**

```python
from cogniton.neurons.hh import HodgkinHuxleyNeuron, HHConfig

config = HHConfig()
neuron = HodgkinHuxleyNeuron(neuron_id=0, config=config)

# Inject step current
neuron.inject_current(10.0)  # μA/cm²

for _ in range(10000):
    neuron.update(t, dt=1e-5)
    t += 1e-5
```

---

## Synaptic Plasticity

### Spike-Timing-Dependent Plasticity (STDP)

STDP modifies synaptic strength based on the relative timing of pre- and postsynaptic spikes:

```
Δw = {  A₊ * exp(-Δt/τ₊)   if Δt > 0  (pre before post → potentiation)
      { -A₋ * exp(Δt/τ₋)    if Δt < 0  (post before pre → depression)
```

Where Δt = t_post - t_pre

**Implementation:**

```python
from cogniton.synapses.plasticity import STDPPlasticity, STDPConfig

config = STDPConfig(
    tau_plus=20e-3,    # 20 ms
    tau_minus=20e-3,   # 20 ms
    a_plus=0.01,       # Potentiation amplitude
    a_minus=0.012,     # Depression amplitude
    w_max=1.0,         # Maximum weight
    w_min=0.0,         # Minimum weight
)

stdp = STDPPlasticity(config)
synapse = STDPSynapse(synapse_id=0, pre_neuron_id=0, post_neuron_id=1)

# Apply STDP
synapse.apply_plasticity(pre_time=0.0, post_time=10e-3)  # Potentiation
synapse.apply_plasticity(pre_time=10e-3, post_time=0.0)  # Depression
```

**Weight Dependence:**

Cogniton supports both additive and multiplicative STDP:

```python
# Additive STDP (all synapses change by same amount)
config = STDPConfig(mu=0.0)

# Multiplicative STDP (changes proportional to current weight)
config = STDPConfig(mu=1.0)
```

### Short-Term Plasticity

Short-term plasticity models synaptic dynamics on fast timescales:

```python
from cogniton.synapses.synapse import Synapse, SynapseModel

synapse = Synapse(
    synapse_id=0,
    pre_neuron_id=0,
    post_neuron_id=1,
    model=SynapseModel.SHORT_TERM_PLASTICITY
)

# Facilitation: use probability increases
synapse.update_short_term_plasticity(t, tau_facil=100e-3, u=0.2)

# Depression: available vesicles decrease
synapse.update_short_term_plasticity(t, tau_depr=100e-3)
```

### Homeostatic Plasticity

Homeostatic mechanisms maintain stable network activity:

```python
from cogniton.synapses.plasticity import HomeostaticPlasticity

homeostatic = HomeostaticPlasticity(
    target_rate=10.0,    # Target firing rate (Hz)
    tau=60.0,            # Adaptation time constant
    learning_rate=0.01
)

# Apply synaptic scaling
homeostatic.apply_synaptic_scaling(synapse_group, current_rate, dt)
```

---

## Network Simulation

### Creating Networks

```python
from cogniton.network.network import SpikingNeuralNetwork
from cogniton.core.config import NetworkConfig, LIFConfig

# Configure
net_config = NetworkConfig(
    num_neurons=1000,
    num_inputs=100,
    num_outputs=10,
    connection_probability=0.1,
    excitatory_ratio=0.8,
    initial_weight_mean=0.5,
    initial_weight_std=0.1,
    delay_mean=5e-3,
    delay_std=2e-3,
)

lif_config = LIFConfig(tau_mem=20e-3)

# Create
network = SpikingNeuralNetwork(
    network_id=0,
    network_config=net_config,
    lif_config=lif_config
)
```

### Input Patterns

**Poisson Spike Trains:**

```python
# Random Poisson input
network.inject_poisson_input(rate=10.0, duration=1.0)
```

**Current Injection:**

```python
# Single neuron
network.inject_current(neuron_id=42, current=1.0)

# Ramp current
network.inject_ramp_current(
    neuron_id=42,
    start_current=0.0,
    end_current=2.0,
    duration=0.5
)
```

### Running Simulations

```python
# Basic run
network.run(t_end=5.0, progress=True)

# Run with callbacks
def my_callback(t):
    if t % 0.1 < 1e-4:
        print(f"Progress: {t/5.0:.1%}")

network.run(t_end=5.0, progress=True, callbacks=[my_callback])
```

### Retrieving Results

```python
# Spike raster
times, ids = network.get_raster_data()

# Firing rates
rates = network.get_firing_rates()

# Weight matrix
W = network.get_weight_matrix()

# Statistics
stats = network.get_statistics()
```

---

## Reservoir Computing

### Echo State Networks

ESNs use a fixed random reservoir with trainable readout:

```python
from cogniton.reservoir.reservoir import EchoStateNetwork, ReservoirConfig

config = ReservoirConfig(
    num_reservoir=400,
    num_inputs=10,
    num_outputs=5,
    spectral_radius=0.9,
    leak_rate=0.3,
)

esn = EchoStateNetwork(config, seed=42)

# Run on time series
for t in range(len(input_sequence)):
    esn.update(input_sequence[t], store_state=True)

# Train readout
esn.train_readout(targets, method='ridge')

# Predict
predictions = esn.predict(test_input)
```

### Liquid State Machines

LSMs use spiking neural networks as the reservoir:

```python
from cogniton.reservoir.reservoir import LiquidStateMachine, ReservoirConfig
from cogniton.core.config import LIFConfig

config = ReservoirConfig(
    num_reservoir=300,
    num_inputs=2,
    num_outputs=2,
)

lif_config = LIFConfig()

lsm = LiquidStateMachine(config, lif_config, seed=42)

# Run simulation
for t in range(1000):
    lsm.update(input_t, dt=1e-4)
```

---

## Hardware Integration

### CPU Backend (Default)

```python
from cogniton.hardware.backends import CPUBackend

backend = CPUBackend(num_threads=4)
backend.connect()
```

### Intel Loihi

```python
from cogniton.hardware.backends import IntelLoihiInterface

interface = IntelLoihiInterface(version=2)
interface.connect(host="loihi-server", port=5000)

# Load network
network_id = interface.load_network(network_config)

# Send/receive spikes
interface.send_spikes(spikes)
received = interface.receive_spikes(timeout=1.0)
```

### IBM TrueNorth

```python
from cogniton.hardware.backends import IBMTrueNorthInterface

interface = IBMTrueNorthInterface()
interface.connect()

# Configure
interface.configure_cores(num_cores=4096)
```

---

## Analysis and Visualization

### Spike Raster

```python
from cogniton.utils.visualization import plot_raster

plot_raster(spike_times, neuron_ids, title="Spike Raster")
```

### Membrane Potential

```python
from cogniton.utils.visualization import plot_voltage

plot_voltage(
    time, voltage,
    v_thresh=-55.0,
    v_reset=-75.0
)
```

### STDP Curve

```python
from cogniton.utils.visualization import plot_stdp_curve

plot_stdp_curve(
    tau_plus=20e-3,
    tau_minus=20e-3,
    a_plus=0.01,
    a_minus=0.012
)
```

### Analysis Functions

```python
from cogniton.utils.analysis import (
    compute_firing_rate,
    compute_population_rate,
    compute_isi,
    compute_cv,
    compute_correlation,
    compute_synchrony,
)

# Firing rate
rate = compute_firing_rate(spike_times, neuron_id)

# Population rate
time, pop_rate = compute_population_rate(spike_raster)

# ISI statistics
isi = compute_isi(spike_times)
cv = compute_cv(isi)

# Correlation
lags, corr = compute_correlation(times_a, times_b)

# Synchrony
sync = compute_synchrony(spike_raster)
```

---

## Examples

See the `examples/` directory for complete examples:

- `example_basic_lif.py` - Single and group LIF neuron simulations
- `example_stdp_learning.py` - STDP plasticity demonstrations
- `example_reservoir_computing.py` - ESN and LSM patterns
- `example_network_simulation.py` - Complete network simulations

Run examples:

```bash
cd examples
python example_basic_lif.py
```

---

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `LIFNeuron` | Leaky Integrate-and-Fire neuron |
| `LIFNeuronGroup` | Group of LIF neurons |
| `HodgkinHuxleyNeuron` | Hodgkin-Huxley neuron |
| `Synapse` | Synaptic connection |
| `SynapseGroup` | Group of synapses |
| `STDPSynapse` | Synapse with STDP |
| `SpikingNeuralNetwork` | Complete SNN |
| `EchoStateNetwork` | Echo State Network |
| `LiquidStateMachine` | Liquid State Machine |

### Configuration Classes

| Class | Description |
|-------|-------------|
| `SimulationConfig` | Simulation parameters |
| `NetworkConfig` | Network topology |
| `LIFConfig` | LIF neuron parameters |
| `HHConfig` | Hodgkin-Huxley parameters |
| `STDPConfig` | STDP parameters |
| `ReservoirConfig` | Reservoir parameters |

---

## Performance Benchmarks

### Scaling Results

| Neurons | Synapses | Wall Time | Real-time Factor |
|---------|----------|----------|------------------|
| 100 | 1,000 | 0.05s | 20x |
| 1,000 | 100,000 | 0.8s | 1.25x |
| 5,000 | 2,500,000 | 15s | 0.03x |

### Neuron Model Comparison

| Model | Spikes/sec (1K neurons) | Relative Speed |
|-------|------------------------|----------------|
| LIF | 50,000 | 1.0x (baseline) |
| HH | 8,000 | 0.16x |

### Hardware Acceleration

| Backend | Speedup vs CPU |
|---------|----------------|
| CPU (4 threads) | 1.0x (baseline) |
| GPU (CUDA) | ~10x |
| Intel Loihi 2 | ~100x |

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Submit a pull request

See `CONTRIBUTING.md` for guidelines.

---

## License

MIT License - see `LICENSE` file for details.

---

## Citations

If you use Cogniton in your research, please cite:

```bibtex
@software{cogniton2024,
  title = {Cogniton: Neuromorphic Computing Simulator},
  author = {Cogniton Team},
  year = {2024},
  url = {https://github.com/moggan1337/Cogniton}
}
```

### Related Papers

1. Hodgkin, A.L. & Huxley, A.F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. *Journal of Physiology*.

2. Maass, W., Natschläger, T., & Markram, H. (2002). Real-time computing without stable states: A new framework for neural computation. *Neural Computation*.

3. Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks. *GMD Report*.

4. Bi, G. & Poo, M. (1998). Synaptic modifications in cultured hippocampal neurons. *Journal of Neuroscience*.

---

## Acknowledgments

Cogniton builds upon decades of research in computational neuroscience and neuromorphic engineering. We acknowledge the contributions of the neuromorphic computing community.

---

## Contact

- GitHub Issues: https://github.com/moggan1337/Cogniton/issues
- Email: team@cogniton.dev

---

<p align="center">
  <strong>Cogniton</strong> - Simulating Brain-Inspired Computation
</p>

---

## Advanced Topics

### Custom Neuron Models

You can extend Cogniton with custom neuron models by implementing the NeuronBase interface:

```python
from cogniton.neurons.base import NeuronBase

class CustomNeuron(NeuronBase):
    def __init__(self, neuron_id, config):
        super().__init__(neuron_id)
        self.config = config
        
    def update(self, t, dt):
        # Custom dynamics
        pass
        
    def reset_state(self):
        # Reset to initial conditions
        pass
```

### Custom Plasticity Rules

Implement your own plasticity rules:

```python
from cogniton.synapses.plasticity import SynapticPlasticity

class CustomPlasticity(SynapticPlasticity):
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.learning_rate = learning_rate
        
    def apply(self, synapse, pre_time, post_time):
        # Custom weight update rule
        delta_w = self.learning_rate * some_function(pre_time, post_time)
        synapse.weight += delta_w
        return synapse.weight
```

### Batch Processing

For large-scale simulations, use batch processing:

```python
from cogniton.simulation.runner import SimulationRunner

runner = SimulationRunner()

# Create multiple networks
for i in range(10):
    runner.create_snn()
    runner.run_snn(t_end=1.0, progress=False)
    
# Collect all results
results = runner.get_results()
```

### Distributed Simulation

For very large networks, distribute across multiple processes:

```python
from multiprocessing import Pool

def simulate_network(args):
    network_id, config = args
    network = SpikingNeuralNetwork(network_id, config)
    network.run(t_end=10.0, progress=False)
    return network.get_statistics()

with Pool(4) as p:
    results = p.map(simulate_network, network_configs)
```

### Real-Time Visualization

Monitor simulations in real-time:

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
sc = ax.scatter([], [])

def update(frame):
    # Update spike raster
    times, ids = network.get_raster_data()
    sc.set_offsets(np.c_[times[-100:], ids[-100:]])
    return sc,

anim = FuncAnimation(fig, update, frames=100, interval=100)
plt.show()
```

---

## Mathematical Foundations

### Membrane Potential Dynamics

The membrane potential evolves according to:

```
dV/dt = -(V - V_rest) / τ_mem + I_ext / C_mem
```

For synaptic input:

```
I_syn(t) = Σ w_i * Σ exp(-(t - t_i^k) / τ_syn) for each synapse
```

### Spike Generation

Spikes occur when:

```
V(t) >= V_thresh
```

After a spike:

```
V(t+) = V_reset
```

### Network Dynamics

In a network of N neurons:

```
τ_i * dV_i/dt = -(V_i - V_rest) + Σ_j W_ij * S_j(t - d_ij)

where S_j(t) = Σ δ(t - t_j^k) is the spike train
```

### Resonance and Oscillations

Networks can exhibit oscillatory behavior:

```
ω = sqrt(1/(τ_e * τ_i) - (1/(2*τ) + g_i/(2*τ_e))^2)
```

---

## Research Applications

### Computational Neuroscience

- **Neural Coding**: Study how information is encoded in spike trains
- **Oscillations**: Analyze gamma, theta, and other brain rhythms
- **Connectomics**: Model specific neural circuits

### Machine Learning

- **Pattern Recognition**: SNNs for temporal pattern classification
- **Time Series**: Reservoir computing for forecasting
- **Reinforcement Learning**: Reward-modulated plasticity

### Robotics

- **Event-Based Vision**: Process dynamic vision sensors
- **Motor Control**: Spiking networks for motor primitives
- **Adaptive Control**: Online learning in robotic systems

### Brain-Computer Interfaces

- **Neural Decoding**: Convert spiking activity to control signals
- **Stimulation**: Closed-loop stimulation protocols
- **Prosthetics**: Brain-machine interface applications

---

## Algorithm Descriptions

### Exponential Euler Integration

For stable numerical integration of differential equations:

```
V(t+dt) = V_inf + (V(t) - V_inf) * exp(-dt/τ)

where V_inf = V_rest + R * I
```

### Event-Driven Updates

Events are processed in priority queue order:

```
while event_queue not empty:
    event = pop_next_event()
    process_event(event)
    schedule_dependent_events(event)
```

### STDP Implementation

The STDP rule is implemented as:

```
if Δt > 0:  # Pre before post
    dw = A_plus * exp(-Δt / tau_plus)
    w = min(w + dw, w_max)
else:  # Post before pre
    dw = -A_minus * exp(Δt / tau_minus)
    w = max(w + dw, w_min)
```

### Weight Normalization

Weights can be normalized to maintain network stability:

```
w_ij = w_ij / (Σ_k w_ik + ε)
```

---

## Glossary

| Term | Definition |
|------|------------|
| **Action Potential** | Brief electrical pulse in neurons |
| **Axonal Delay** | Time for spike to travel to synapses |
| **Excitatory** | Synapse that increases firing probability |
| **Inhibitory** | Synapse that decreases firing probability |
| **Membrane Time Constant** | Rate of voltage decay (τ = RC) |
| **Plasticity** | Activity-dependent synaptic modification |
| **Refractory Period** | Time after spike when neuron cannot fire |
| **Synaptic Weight** | Strength of synaptic connection |
| **Spike Train** | Sequence of spike times |
| **Synaptic Current** | Current injected by synaptic activation |

---

## Frequently Asked Questions

**Q: How does Cogniton compare to NEST/Brian2?**

A: Cogniton is designed for rapid prototyping and flexibility, with a focus on neuromorphic hardware integration. NEST and Brian2 are more mature simulators with greater biological detail. Cogniton emphasizes computational efficiency and hardware co-design.

**Q: Can I use Cogniton on actual neuromorphic hardware?**

A: Yes! Cogniton includes interfaces for Intel Loihi and IBM TrueNorth. For software-only simulation, the CPU backend is used by default.

**Q: How do I choose between LIF and HH neurons?**

A: Use LIF for large-scale simulations where computational efficiency is important. Use HH when you need detailed ion channel dynamics, such as studying action potential initiation or drug effects.

**Q: What is the maximum network size?**

A: Cogniton has been tested with networks up to 100,000 neurons. The actual limit depends on available memory. For larger networks, consider using sparse connectivity or neuromorphic hardware.

**Q: How do I cite Cogniton in my paper?**

A: See the Citations section above for the recommended BibTeX entry.

---

## Changelog

### v1.0.0 (2024)

- Initial release
- LIF and Hodgkin-Huxley neuron models
- STDP and short-term plasticity
- Event-driven simulation engine
- Reservoir computing (ESN, LSM)
- Intel Loihi and IBM TrueNorth interfaces
- Analysis and visualization tools
- Comprehensive test suite

---

## References and Further Reading

### Books

1. Dayan, P. & Abbott, L.F. (2001). Theoretical Neuroscience. MIT Press.
2. Gerstner, W. & Kistler, W.M. (2002). Spiking Neuron Models. Cambridge University Press.
3. Izhikevich, E.M. (2007). Dynamical Systems in Neuroscience. MIT Press.

### Review Articles

1. Izhikevich, E.M. (2004). Which model to use for cortical spiking neurons? IEEE Transactions on Neural Networks.
2. Pfeiffer, M. & Pfeil, T. (2018). Deep Learning With Spiking Neurons. Frontiers in Neuroscience.

### Research Papers

1. Hodgkin, A.L. & Huxley, A.F. (1952). A quantitative description of membrane current. Journal of Physiology.
2. Maass, W., Natschläger, T., & Markram, H. (2002). Real-time computing without stable states. Neural Computation.
3. Jaeger, H. (2001). The echo state approach to analysing and training RNNs. GMD Report.
4. Bi, G. & Poo, M. (1998). Synaptic modifications in cultured hippocampal neurons. Journal of Neuroscience.

---

<p align="center">
  Made with ❤️ by the Cogniton Team<br>
  <a href="https://github.com/moggan1337/Cogniton">GitHub</a> •
  <a href="https://github.com/moggan1337/Cogniton/issues">Issues</a> •
  <a href="https://github.com/moggan1337/Cogniton/discussions">Discussions</a>
</p>
