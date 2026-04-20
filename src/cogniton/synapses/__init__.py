"""Synapse module for Cogniton."""

from cogniton.synapses.synapse import Synapse, SynapseGroup, ConnectionType
from cogniton.synapses.plasticity import STDPSynapse, STDPConfig, SynapticPlasticity

__all__ = [
    "Synapse",
    "SynapseGroup",
    "ConnectionType",
    "STDPSynapse",
    "STDPConfig",
    "SynapticPlasticity",
]
