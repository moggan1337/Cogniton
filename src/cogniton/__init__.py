"""
Cogniton - Neuromorphic Computing Simulator
===========================================

A comprehensive framework for simulating spiking neural networks,
neuromorphic hardware, and reservoir computing systems.

Author: Cogniton Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Cogniton Team"

from cogniton.core.event import Event, EventQueue, EventDrivenSimulation
from cogniton.core.time import SimulationTime
from cogniton.neurons.lif import LIFNeuron, LIFNeuronGroup
from cogniton.neurons.hh import HodgkinHuxleyNeuron, HodgkinHuxleyGroup
from cogniton.synapses.synapse import Synapse, SynapseGroup
from cogniton.synapses.plasticity import STDPSynapse, STDPConfig
from cogniton.network.network import SpikingNeuralNetwork, NetworkConfig
from cogniton.simulation.runner import SimulationRunner
from cogniton.reservoir.reservoir import LiquidStateMachine, EchoStateNetwork

__all__ = [
    "Event",
    "EventQueue",
    "EventDrivenSimulation",
    "SimulationTime",
    "LIFNeuron",
    "LIFNeuronGroup",
    "HodgkinHuxleyNeuron",
    "HodgkinHuxleyGroup",
    "Synapse",
    "SynapseGroup",
    "STDPSynapse",
    "STDPConfig",
    "SpikingNeuralNetwork",
    "NetworkConfig",
    "SimulationRunner",
    "LiquidStateMachine",
    "EchoStateNetwork",
]
