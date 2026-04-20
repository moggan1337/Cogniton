"""Neuron module for Cogniton."""

from cogniton.neurons.lif import LIFNeuron, LIFNeuronGroup
from cogniton.neurons.hh import HodgkinHuxleyNeuron, HodgkinHuxleyGroup

__all__ = [
    "LIFNeuron",
    "LIFNeuronGroup",
    "HodgkinHuxleyNeuron",
    "HodgkinHuxleyGroup",
]
