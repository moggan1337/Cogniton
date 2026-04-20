"""Hardware integration module for Cogniton."""

from cogniton.hardware.backends import (
    HardwareBackend,
    HardwareInterface,
    IntelLoihiInterface,
    IBMTrueNorthInterface,
    CPUBackend,
)

__all__ = [
    "HardwareBackend",
    "HardwareInterface",
    "IntelLoihiInterface",
    "IBMTrueNorthInterface",
    "CPUBackend",
]
