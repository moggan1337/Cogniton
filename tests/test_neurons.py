"""
Test Suite for Cogniton
========================

Unit tests for all core components.
"""

import unittest
import numpy as np

from cogniton.neurons.lif import LIFNeuron, LIFNeuronGroup
from cogniton.neurons.hh import HodgkinHuxleyNeuron
from cogniton.synapses.synapse import Synapse, SynapseGroup
from cogniton.synapses.plasticity import STDPPlasticity, STDPSynapse
from cogniton.core.config import LIFConfig, HHConfig, STDPConfig
from cogniton.core.event import Event, EventQueue, EventDrivenSimulation
from cogniton.core.time import SimulationTime
from cogniton.reservoir.reservoir import EchoStateNetwork, ReservoirConfig


class TestLIFNeuron(unittest.TestCase):
    """Tests for LIF neuron."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = LIFConfig(
            tau_mem=20e-3,
            v_thresh=-55.0,
            v_reset=-75.0,
            v_rest=-70.0,
        )
        self.neuron = LIFNeuron(neuron_id=0, config=self.config)
    
    def test_initial_state(self):
        """Test initial neuron state."""
        self.assertEqual(self.neuron.v, self.config.v_init)
        self.assertEqual(self.neuron.i_syn, 0.0)
        self.assertEqual(len(self.neuron.spike_times), 0)
    
    def test_current_injection(self):
        """Test current injection."""
        self.neuron.inject_current(1.0)
        self.assertEqual(self.neuron.i_syn, 1.0)
    
    def test_spike_generation(self):
        """Test spike generation with sufficient current."""
        self.neuron.inject_current(10.0)  # Large current
        
        for _ in range(100):
            spiked = self.neuron.update(0.0, dt=1e-4)
            if spiked:
                break
        
        self.assertGreater(len(self.neuron.spike_times), 0)
    
    def test_reset(self):
        """Test neuron reset."""
        self.neuron.v = -50.0
        self.neuron.spike_times.append(0.1)
        
        self.neuron.reset_state()
        
        self.assertEqual(self.neuron.v, self.config.v_init)
        self.assertEqual(len(self.neuron.spike_times), 0)
    
    def test_firing_rate(self):
        """Test firing rate calculation."""
        # Add some fake spikes
        self.neuron.spike_times = [0.1, 0.2, 0.3, 0.4]
        
        rate = self.neuron.get_firing_rate(t_start=0.0, t_end=0.5)
        
        self.assertAlmostEqual(rate, 8.0, places=1)


class TestLIFNeuronGroup(unittest.TestCase):
    """Tests for LIF neuron group."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.group = LIFNeuronGroup(group_id=0, n=10)
    
    def test_group_size(self):
        """Test group size."""
        self.assertEqual(len(self.group), 10)
    
    def test_vectorized_update(self):
        """Test vectorized group update."""
        spiked = self.group.update(t=0.0, dt=1e-4)
        
        self.assertEqual(len(spiked), 10)
        self.assertIsInstance(spiked, np.ndarray)
    
    def test_current_injection(self):
        """Test current injection to group."""
        current = np.ones(10) * 0.5
        self.group.inject_current(current)
        
        np.testing.assert_array_equal(self.group.i_syn, current)


class TestHodgkinHuxley(unittest.TestCase):
    """Tests for Hodgkin-Huxley neuron."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = HHConfig()
        self.neuron = HodgkinHuxleyNeuron(neuron_id=0, config=self.config)
    
    def test_initial_state(self):
        """Test initial HH state."""
        self.assertIsNotNone(self.neuron.v)
        self.assertIsNotNone(self.neuron.m)
        self.assertIsNotNone(self.neuron.h)
        self.assertIsNotNone(self.neuron.n)
    
    def test_current_injection(self):
        """Test current injection."""
        self.neuron.inject_current(5.0)
        self.assertEqual(self.neuron.i_applied, 5.0)
    
    def test_action_potential(self):
        """Test action potential generation."""
        # Inject large step current
        self.neuron.inject_current(10.0)
        
        for _ in range(1000):
            spiked = self.neuron.update(t=0.0, dt=1e-5)
            if spiked:
                break
        
        self.assertGreater(len(self.neuron.spike_times), 0)


class TestSynapse(unittest.TestCase):
    """Tests for synapse."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.synapse = Synapse(
            synapse_id=0,
            pre_neuron_id=0,
            post_neuron_id=1,
            weight=0.5,
            delay=5e-3,
        )
    
    def test_initial_state(self):
        """Test initial synapse state."""
        self.assertEqual(self.synapse.weight, 0.5)
        self.assertEqual(self.synapse.delay, 5e-3)
    
    def test_spike_delivery(self):
        """Test presynaptic spike."""
        weight = self.synapse.pre_spike(time=0.0)
        
        self.assertEqual(weight, 0.5)
        self.assertEqual(self.synapse.last_spike_time, 0.0)


class TestSTDP(unittest.TestCase):
    """Tests for STDP plasticity."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = STDPConfig()
        self.stdp = STDPPlasticity(self.config)
        self.synapse = STDPSynapse(
            synapse_id=0,
            pre_neuron_id=0,
            post_neuron_id=1,
            weight=0.5,
            stdp_config=self.config,
        )
    
    def test_potentiation(self):
        """Test STDP potentiation (pre before post)."""
        initial_weight = self.synapse.weight
        self.synapse.apply_plasticity(pre_time=0.0, post_time=10e-3)
        
        self.assertGreater(self.synapse.weight, initial_weight)
    
    def test_depression(self):
        """Test STDP depression (post before pre)."""
        initial_weight = self.synapse.weight
        self.synapse.apply_plasticity(pre_time=10e-3, post_time=0.0)
        
        self.assertLess(self.synapse.weight, initial_weight)


class TestEventQueue(unittest.TestCase):
    """Tests for event queue."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.queue = EventQueue()
    
    def test_push_pop(self):
        """Test basic push and pop."""
        event = Event(time=0.1, event_type=EventType.SPIKE)
        self.queue.push(event)
        
        self.assertEqual(self.queue.size, 1)
        
        popped = self.queue.pop()
        self.assertEqual(popped.time, 0.1)
        self.assertEqual(self.queue.size, 0)
    
    def test_priority_order(self):
        """Test priority ordering."""
        self.queue.push(Event(time=0.3, event_type=EventType.SPIKE))
        self.queue.push(Event(time=0.1, event_type=EventType.SPIKE))
        self.queue.push(Event(time=0.2, event_type=EventType.SPIKE))
        
        first = self.queue.pop()
        self.assertEqual(first.time, 0.1)


class TestSimulationTime(unittest.TestCase):
    """Tests for simulation time."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sim_time = SimulationTime(t_start=0.0, t_end=1.0, dt=0.1)
    
    def test_initial_state(self):
        """Test initial time state."""
        self.assertEqual(self.sim_time.t, 0.0)
        self.assertEqual(self.sim_time.progress, 0.0)
    
    def test_advance(self):
        """Test time advancement."""
        self.sim_time.advance()
        
        self.assertAlmostEqual(self.sim_time.t, 0.1)
        self.assertAlmostEqual(self.sim_time.progress, 0.1)
    
    def test_reset(self):
        """Test time reset."""
        self.sim_time.advance()
        self.sim_time.reset()
        
        self.assertEqual(self.sim_time.t, 0.0)


class TestEchoStateNetwork(unittest.TestCase):
    """Tests for Echo State Network."""
    
    def setUp(self):
        """Set up test fixtures."""
        config = ReservoirConfig(
            num_reservoir=100,
            num_inputs=2,
            num_outputs=1,
        )
        self.esn = EchoStateNetwork(config, seed=42)
    
    def test_initialization(self):
        """Test ESN initialization."""
        self.assertEqual(len(self.esn.reservoir_state), 100)
        self.assertEqual(len(self.esn.input_state), 2)
    
    def test_update(self):
        """Test ESN update."""
        u = np.array([1.0, 0.5])
        x = self.esn.update(u, store_state=False)
        
        self.assertEqual(len(x), 100)
    
    def test_reset(self):
        """Test ESN reset."""
        u = np.array([1.0, 0.5])
        self.esn.update(u)
        
        self.esn.reset_state()
        
        np.testing.assert_array_almost_equal(
            self.esn.reservoir_state, 
            np.zeros(100)
        )


class TestSynapseGroup(unittest.TestCase):
    """Tests for synapse group."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.group = SynapseGroup(group_id=0)
    
    def test_add_synapse(self):
        """Test adding synapses."""
        syn = Synapse(0, 0, 1, weight=0.5)
        self.group.add_synapse(syn)
        
        self.assertEqual(len(self.group), 1)
    
    def test_outgoing_synapses(self):
        """Test getting outgoing synapses."""
        self.group.add_synapse(Synapse(0, 0, 1, weight=0.5))
        self.group.add_synapse(Synapse(1, 0, 2, weight=0.3))
        self.group.add_synapse(Synapse(2, 1, 2, weight=0.4))
        
        outgoing = self.group.get_outgoing_synapses(0)
        
        self.assertEqual(len(outgoing), 2)


if __name__ == "__main__":
    # Add missing import
    from cogniton.core.event import EventType
    unittest.main()
