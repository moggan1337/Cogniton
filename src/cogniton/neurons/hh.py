"""
Hodgkin-Huxley Neuron Model
===========================

Implementation of the Hodgkin-Huxley model with full ion channel dynamics.

The Hodgkin-Huxley model describes the initiation and propagation of 
action potentials in squid giant axons. It's a conductance-based model
that captures the dynamics of sodium and potassium ion channels.

Mathematical Model:
-------------------
C_m * dv/dt = -g_na * m^3 * h * (v - E_na) 
              - g_k * n^4 * (v - E_k) 
              - g_l * (v - E_l) + I(t)

Where the gating variables m, h, n follow:
    dx/dt = alpha_x * (1 - x) - beta_x * x

References:
-----------
Hodgkin, A.L. & Huxley, A.F. (1952). A quantitative description of 
    membrane current and its application to conduction and excitation 
    in nerve. Journal of Physiology, 117: 500-544.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any, Callable
import numpy as np

from cogniton.core.config import HHConfig


class HodgkinHuxleyNeuron:
    """
    Hodgkin-Huxley conductance-based neuron model.
    
    This model provides biophysically accurate simulation of action
    potentials including sodium and potassium channel dynamics.
    
    Gating Variables:
    - m: Sodium activation (fast)
    - h: Sodium inactivation (slow)
    - n: Potassium activation (slow)
    
    Attributes:
        neuron_id: Unique identifier
        config: HH model parameters
        v: Membrane potential (mV)
        m, h, n: Gating variables
    """
    
    def __init__(self, neuron_id: int, config: Optional[HHConfig] = None,
                 seed: Optional[int] = None):
        """
        Initialize Hodgkin-Huxley neuron.
        
        Args:
            neuron_id: Unique neuron identifier
            config: HH parameters (uses default if None)
            seed: Random seed (for noise injection)
        """
        self.neuron_id = neuron_id
        self.config = config or HHConfig()
        
        # State variables
        self.v = self.config.v_init
        self.m = self._alpha_m(self.v) / (self._alpha_m(self.v) + self._beta_m(self.v))
        self.h = self._alpha_h(self.v) / (self._alpha_h(self.v) + self._beta_h(self.v))
        self.n = self._alpha_n(self.v) / (self._alpha_n(self.v) + self._beta_n(self.v))
        
        # Applied current
        self.i_applied = 0.0
        
        # History
        self.spike_times: List[float] = []
        self.v_history: List[Tuple[float, float]] = []
        self.gating_history: List[Tuple[float, float, float, float]] = []
        
        # Timing
        self.last_update: float = 0.0
        self.threshold_crossed: bool = False
        
        # Spike callback
        self._spike_callback: Optional[Callable] = None
    
    # Rate functions for gating variables
    def _alpha_m(self, v: float) -> float:
        """Alpha rate for m gate (sodium activation)."""
        return 0.1 * (v + 40.0) / (1.0 - np.exp(-(v + 40.0) / 10.0))
    
    def _beta_m(self, v: float) -> float:
        """Beta rate for m gate."""
        return 4.0 * np.exp(-(v + 65.0) / 18.0)
    
    def _alpha_h(self, v: float) -> float:
        """Alpha rate for h gate (sodium inactivation)."""
        return 0.07 * np.exp(-(v + 65.0) / 20.0)
    
    def _beta_h(self, v: float) -> float:
        """Beta rate for h gate."""
        return 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))
    
    def _alpha_n(self, v: float) -> float:
        """Alpha rate for n gate (potassium activation)."""
        return 0.01 * (v + 55.0) / (1.0 - np.exp(-(v + 55.0) / 10.0))
    
    def _beta_n(self, v: float) -> float:
        """Beta rate for n gate."""
        return 0.125 * np.exp(-(v + 65.0) / 80.0)
    
    def _tau_m(self, v: float) -> float:
        """Time constant for m gate."""
        return 1.0 / (self._alpha_m(v) + self._beta_m(v))
    
    def _tau_h(self, v: float) -> float:
        """Time constant for h gate."""
        return 1.0 / (self._alpha_h(v) + self._beta_h(v))
    
    def _tau_n(self, v: float) -> float:
        """Time constant for n gate."""
        return 1.0 / (self._alpha_n(v) + self._beta_n(v))
    
    def _inf_m(self, v: float) -> float:
        """Steady-state value for m gate."""
        return self._alpha_m(v) / (self._alpha_m(v) + self._beta_m(v))
    
    def _inf_h(self, v: float) -> float:
        """Steady-state value for h gate."""
        return self._alpha_h(v) / (self._alpha_h(v) + self._beta_h(v))
    
    def _inf_n(self, v: float) -> float:
        """Steady-state value for n gate."""
        return self._alpha_n(v) / (self._alpha_n(v) + self._beta_n(v))
    
    @property
    def g_na(self) -> float:
        """Sodium conductance (mS/cm²)."""
        return self.config.g_na
    
    @property
    def g_k(self) -> float:
        """Potassium conductance (mS/cm²)."""
        return self.config.g_k
    
    @property
    def g_l(self) -> float:
        """Leak conductance (mS/cm²)."""
        return self.config.g_l
    
    def currents(self, v: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate all currents.
        
        Args:
            v: Membrane potential (uses current if None)
            
        Returns:
            Dictionary of currents
        """
        if v is None:
            v = self.v
        
        i_na = self.g_na * (self.m ** 3) * self.h * (v - self.config.e_na)
        i_k = self.g_k * (self.n ** 4) * (v - self.config.e_k)
        i_l = self.g_l * (v - self.config.e_l)
        
        return {
            "i_na": i_na,
            "i_k": i_k,
            "i_l": i_l,
            "i_total": i_na + i_k + i_l + self.i_applied
        }
    
    def receive_spike(self, time: float, weight: float) -> None:
        """Receive synaptic input."""
        self.i_applied += weight
    
    def inject_current(self, current: float) -> None:
        """Inject external current."""
        self.i_applied += current
    
    def update(self, t: float, dt: float) -> bool:
        """
        Update HH neuron using exponential Euler integration.
        
        Args:
            t: Current time
            dt: Integration timestep
            
        Returns:
            True if spike occurred
        """
        self.last_update = t
        
        # Check for spike threshold crossing (for spike detection)
        spiked = False
        if self.v >= 0 and not self.threshold_crossed:
            spiked = True
            self.spike_times.append(t)
            self.threshold_crossed = True
            
            if self._spike_callback:
                self._spike_callback(self, t)
        
        # Reset threshold flag when voltage goes back below -20mV
        if self.v < -20:
            self.threshold_crossed = False
        
        # Update gating variables using exponential Euler
        # dx/dt = (inf_x(v) - x) / tau_x
        # x(t+dt) = inf_x(v) + (x - inf_x(v)) * exp(-dt/tau_x)
        
        exp_m = np.exp(-dt / self._tau_m(self.v))
        exp_h = np.exp(-dt / self._tau_h(self.v))
        exp_n = np.exp(-dt / self._tau_n(self.v))
        
        self.m = self._inf_m(self.v) + (self.m - self._inf_m(self.v)) * exp_m
        self.h = self._inf_h(self.v) + (self.h - self._inf_h(self.v)) * exp_h
        self.n = self._inf_n(self.v) + (self.n - self._inf_n(self.v)) * exp_n
        
        # Clamp gating variables
        self.m = np.clip(self.m, 0, 1)
        self.h = np.clip(self.h, 0, 1)
        self.n = np.clip(self.n, 0, 1)
        
        # Update membrane potential
        # dv/dt = -(i_na + i_k + i_l + i_applied) / C_m
        i_curr = self.currents()
        
        # Membrane time constant
        g_total = self.g_na * (self.m ** 3) * self.h + \
                  self.g_k * (self.n ** 4) + self.g_l
        
        if g_total > 0:
            tau_v = self.config.c_m / g_total
            v_inf = (self.g_na * (self.m ** 3) * self.h * self.config.e_na +
                    self.g_k * (self.n ** 4) * self.config.e_k +
                    self.g_l * self.config.e_l - self.i_applied) / g_total
            
            exp_v = np.exp(-dt / tau_v)
            self.v = v_inf + (self.v - v_inf) * exp_v
        else:
            # Fallback: simple Euler
            self.v += dt * (-i_curr["i_total"] / self.config.c_m)
        
        # Record history
        self.v_history.append((t, self.v))
        self.gating_history.append((t, self.m, self.h, self.n))
        
        # Reset applied current (synaptic currents handled separately)
        self.i_applied = 0.0
        
        return spiked
    
    def reset_state(self) -> None:
        """Reset neuron to initial state."""
        self.v = self.config.v_init
        self.m = self._alpha_m(self.v) / (self._alpha_m(self.v) + self._beta_m(self.v))
        self.h = self._alpha_h(self.v) / (self._alpha_h(self.v) + self._beta_h(self.v))
        self.n = self._alpha_n(self.v) / (self._alpha_n(self.v) + self._beta_n(self.v))
        self.i_applied = 0.0
        self.spike_times.clear()
        self.v_history.clear()
        self.gating_history.clear()
        self.threshold_crossed = False
    
    def get_firing_rate(self, t_start: float = 0.0, 
                        t_end: Optional[float] = None) -> float:
        """Calculate firing rate."""
        if t_end is None:
            t_end = self.v_history[-1][0] if self.v_history else t_start
        
        duration = t_end - t_start
        if duration <= 0:
            return 0.0
        
        return len([st for st in self.spike_times if t_start <= st <= t_end]) / duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "neuron_id": self.neuron_id,
            "v": self.v,
            "m": self.m,
            "h": self.h,
            "n": self.n,
            "i_applied": self.i_applied,
            "spike_times": self.spike_times.copy(),
            "config": {
                "g_na": self.config.g_na,
                "g_k": self.config.g_k,
                "g_l": self.config.g_l,
                "e_na": self.config.e_na,
                "e_k": self.config.e_k,
                "e_l": self.config.e_l,
                "c_m": self.config.c_m,
                "v_init": self.config.v_init,
            }
        }
    
    def __repr__(self) -> str:
        return (f"HHNeuron(id={self.neuron_id}, v={self.v:.1f}mV, "
                f"m={self.m:.3f}, h={self.h:.3f}, n={self.n:.3f}, "
                f"spikes={len(self.spike_times)})")


class HodgkinHuxleyGroup:
    """
    Group of Hodgkin-Huxley neurons with vectorized operations.
    
    Note: Full vectorization of HH is complex due to nonlinear gating.
    This implementation uses batched updates for efficiency.
    """
    
    def __init__(self, group_id: int, n: int,
                 config: Optional[HHConfig] = None,
                 seeds: Optional[List[int]] = None):
        """
        Initialize HH neuron group.
        
        Args:
            group_id: Group identifier
            n: Number of neurons
            config: Shared HH configuration
            seeds: Random seeds
        """
        self.group_id = group_id
        self.config = config or HHConfig()
        self.n = n
        
        # Create neurons
        self.neurons: List[HodgkinHuxleyNeuron] = []
        for i in range(n):
            seed = seeds[i] if seeds is not None else None
            neuron = HodgkinHuxleyNeuron(i, self.config, seed)
            self.neurons.append(neuron)
        
        # State arrays
        self._init_arrays()
    
    def _init_arrays(self) -> None:
        """Initialize state arrays."""
        self.v = np.full(self.n, self.config.v_init)
        
        # Initialize gating variables
        alpha_m = 0.1 * (self.v + 40.0) / (1.0 - np.exp(-(self.v + 40.0) / 10.0))
        beta_m = 4.0 * np.exp(-(self.v + 65.0) / 18.0)
        self.m = alpha_m / (alpha_m + beta_m)
        
        alpha_h = 0.07 * np.exp(-(self.v + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + np.exp(-(self.v + 35.0) / 10.0))
        self.h = alpha_h / (alpha_h + beta_h)
        
        alpha_n = 0.01 * (self.v + 55.0) / (1.0 - np.exp(-(self.v + 55.0) / 10.0))
        beta_n = 0.125 * np.exp(-(self.v + 65.0) / 80.0)
        self.n = alpha_n / (alpha_n + beta_n)
        
        # Spike tracking
        self.spike_count = np.zeros(self.n, dtype=int)
        self.spike_times: List[Tuple[float, int]] = []
        self.threshold_crossed = np.zeros(self.n, dtype=bool)
        
        # Applied current
        self.i_applied = np.zeros(self.n)
    
    def update(self, t: float, dt: float) -> np.ndarray:
        """
        Update all neurons.
        
        Args:
            t: Current time
            dt: Integration timestep
            
        Returns:
            Boolean array of neurons that spiked
        """
        # Check for new spikes
        spiked = (self.v >= 0) & (~self.threshold_crossed)
        self.threshold_crossed = self.v >= 0
        
        # Update spike counts and records
        for idx in np.where(spiked)[0]:
            self.spike_count[idx] += 1
            self.spike_times.append((t, self.group_id * 1000 + idx))
            self.neurons[idx].spike_times.append(t)
        
        # Update gating variables (exponential Euler)
        # m
        alpha_m = 0.1 * (self.v + 40.0) / (1.0 - np.exp(-(self.v + 40.0) / 10.0))
        beta_m = 4.0 * np.exp(-(self.v + 65.0) / 18.0)
        tau_m = 1.0 / (alpha_m + beta_m)
        inf_m = alpha_m / (alpha_m + beta_m)
        self.m = inf_m + (self.m - inf_m) * np.exp(-dt / tau_m)
        
        # h
        alpha_h = 0.07 * np.exp(-(self.v + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + np.exp(-(self.v + 35.0) / 10.0))
        tau_h = 1.0 / (alpha_h + beta_h)
        inf_h = alpha_h / (alpha_h + beta_h)
        self.h = inf_h + (self.h - inf_h) * np.exp(-dt / tau_h)
        
        # n
        alpha_n = 0.01 * (self.v + 55.0) / (1.0 - np.exp(-(self.v + 55.0) / 10.0))
        beta_n = 0.125 * np.exp(-(self.v + 65.0) / 80.0)
        tau_n = 1.0 / (alpha_n + beta_n)
        inf_n = alpha_n / (alpha_n + beta_n)
        self.n = inf_n + (self.n - inf_n) * np.exp(-dt / tau_n)
        
        # Clamp gating variables
        self.m = np.clip(self.m, 0, 1)
        self.h = np.clip(self.h, 0, 1)
        self.n = np.clip(self.n, 0, 1)
        
        # Update membrane potential
        g_na = self.config.g_na * (self.m ** 3) * self.h
        g_k = self.config.g_k * (self.n ** 4)
        g_l = self.config.g_l
        
        i_na = g_na * (self.v - self.config.e_na)
        i_k = g_k * (self.v - self.config.e_k)
        i_l = g_l * (self.v - self.config.e_l)
        
        # Total current
        i_total = i_na + i_k + i_l + self.i_applied
        
        # Membrane potential update
        g_total = g_na + g_k + g_l
        tau_v = np.where(g_total > 0, self.config.c_m / g_total, 1.0)
        
        v_inf = np.where(
            g_total > 0,
            (g_na * self.config.e_na + g_k * self.config.e_k + 
             g_l * self.config.e_l - self.i_applied) / g_total,
            self.v
        )
        
        self.v = v_inf + (self.v - v_inf) * np.exp(-dt / tau_v)
        
        # Reset applied current
        self.i_applied.fill(0)
        
        return spiked
    
    def inject_current(self, current: np.ndarray) -> None:
        """Inject current into all neurons."""
        self.i_applied += np.asarray(current)
    
    def reset(self) -> None:
        """Reset all neurons."""
        self._init_arrays()
        for neuron in self.neurons:
            neuron.reset_state()
    
    def get_firing_rates(self, t_start: float = 0.0,
                         t_end: Optional[float] = None) -> np.ndarray:
        """Calculate firing rates."""
        if t_end is None:
            t_end = t_start + 1.0
        
        duration = t_end - t_start
        return self.spike_count / max(duration, 1e-9)
    
    def __len__(self) -> int:
        return self.n
    
    def __repr__(self) -> str:
        return f"HHGroup(id={self.group_id}, n={self.n}, spikes={len(self.spike_times)})"
