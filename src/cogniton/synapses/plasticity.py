"""
Synaptic Plasticity Module
==========================

Implements Spike-Timing-Dependent Plasticity (STDP) and other
plasticity mechanisms for learning in spiking neural networks.

STDP Rule:
----------
The weight update depends on the relative timing of pre- and 
postsynaptic spikes:

    Δw = {  A_plus  * exp(-Δt/τ_plus)   if Δt > 0  (pre before post)
        { -A_minus * exp( Δt/τ_minus)   if Δt < 0  (post before pre)

Where Δt = t_post - t_pre

References:
-----------
Bi, G. & Poo, M. (1998). Synaptic modifications in cultured hippocampal 
    neurons: dependence on spike timing, synaptic strength, and 
    postsynaptic cell type. Journal of Neuroscience.
    
Markram et al. (1997). Regulation of synaptic efficacy by coincidence 
    of postsynaptic APs and EPSPs. Science.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
import numpy as np

from cogniton.core.config import STDPConfig
from cogniton.synapses.synapse import Synapse, SynapseGroup, SynapseModel


class SynapticPlasticity:
    """
    Base class for synaptic plasticity mechanisms.
    
    Provides interface for implementing various plasticity rules.
    """
    
    def __init__(self):
        """Initialize plasticity mechanism."""
        self.enabled = True
    
    def apply(self, synapse: Synapse, pre_time: float, 
              post_time: float) -> float:
        """
        Apply plasticity rule.
        
        Args:
            synapse: Synapse to update
            pre_time: Presynaptic spike time
            post_time: Postsynaptic spike time
            
        Returns:
            New weight
        """
        raise NotImplementedError
    
    def batch_apply(self, synapses: List[Synapse],
                    pre_times: List[float],
                    post_times: List[float]) -> np.ndarray:
        """
        Apply plasticity to batch of synapses.
        
        Args:
            synapses: List of synapses
            pre_times: Presynaptic spike times
            post_times: Postsynaptic spike times
            
        Returns:
            Array of new weights
        """
        weights = np.zeros(len(synapses))
        for i, (s, pt, po) in enumerate(zip(synapses, pre_times, post_times)):
            weights[i] = self.apply(s, pt, po)
        return weights
    
    def enable(self) -> None:
        """Enable plasticity."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable plasticity."""
        self.enabled = False


class STDPPlasticity(SynapticPlasticity):
    """
    Spike-Timing-Dependent Plasticity implementation.
    
    Implements both additive and multiplicative STDP rules
    with optional soft bounds.
    
    Attributes:
        config: STDP parameters
        pre_traces: Presynaptic eligibility traces
        post_traces: Postsynaptic eligibility traces
    """
    
    def __init__(self, config: Optional[STDPConfig] = None):
        """
        Initialize STDP plasticity.
        
        Args:
            config: STDP parameters
        """
        super().__init__()
        self.config = config or STDPConfig()
        
        # Eligibility traces
        self.pre_traces: Dict[int, float] = {}
        self.post_traces: Dict[int, float] = {}
        
        # Spike history for batch processing
        self._pre_spikes: Dict[int, List[float]] = {}
        self._post_spikes: Dict[int, List[float]] = {}
    
    def apply(self, synapse: Synapse, pre_time: float,
              post_time: float) -> float:
        """
        Apply STDP update to single synapse.
        
        Args:
            synapse: Synapse to update
            pre_time: Presynaptic spike time
            post_time: Postsynaptic spike time
            
        Returns:
            New weight
        """
        if not self.enabled:
            return synapse.weight
        
        delta_t = post_time - pre_time
        
        if self.config.mu == 0:
            # Additive STDP
            if delta_t > 0:
                # Potentiation
                dw = self.config.a_plus * np.exp(-delta_t / self.config.tau_plus)
                synapse.weight = min(synapse.weight + dw, self.config.w_max)
            else:
                # Depression
                dw = -self.config.a_minus * np.exp(delta_t / self.config.tau_minus)
                synapse.weight = max(synapse.weight + dw, self.config.w_min)
        else:
            # Multiplicative STDP
            if delta_t > 0:
                dw = self.config.a_plus * (self.config.w_max - synapse.weight) * \
                     np.exp(-delta_t / self.config.tau_plus)
                synapse.weight += dw
            else:
                dw = -self.config.a_minus * synapse.weight * \
                     np.exp(delta_t / self.config.tau_minus)
                synapse.weight += dw
            
            # Clamp
            synapse.weight = max(self.config.w_min, 
                               min(self.config.w_max, synapse.weight))
        
        return synapse.weight
    
    def apply_potentiation_only(self, synapse: Synapse,
                                pre_time: float, post_time: float) -> float:
        """
        Apply only potentiation (for reward-modulated STDP).
        
        Args:
            synapse: Synapse to update
            pre_time: Presynaptic spike time
            post_time: Postsynaptic spike time
            
        Returns:
            New weight
        """
        if not self.enabled:
            return synapse.weight
        
        delta_t = post_time - pre_time
        if delta_t > 0:
            dw = self.config.a_plus * np.exp(-delta_t / self.config.tau_plus)
            if self.config.use_soft_bounds:
                synapse.weight = min(synapse.weight + dw, self.config.w_max)
            else:
                synapse.weight += dw
        
        return synapse.weight
    
    def apply_depression_only(self, synapse: Synapse,
                             pre_time: float, post_time: float) -> float:
        """
        Apply only depression.
        
        Args:
            synapse: Synapse to update
            pre_time: Presynaptic spike time
            post_time: Postsynaptic spike time
            
        Returns:
            New weight
        """
        if not self.enabled:
            return synapse.weight
        
        delta_t = post_time - pre_time
        if delta_t < 0:
            dw = -self.config.a_minus * np.exp(delta_t / self.config.tau_minus)
            synapse.weight = max(synapse.weight + dw, self.config.w_min)
        
        return synapse.weight
    
    def record_pre_spike(self, synapse_id: int, time: float) -> None:
        """Record presynaptic spike for batch processing."""
        if synapse_id not in self._pre_spikes:
            self._pre_spikes[synapse_id] = []
        self._pre_spikes[synapse_id].append(time)
    
    def record_post_spike(self, synapse_id: int, time: float) -> None:
        """Record postsynaptic spike for batch processing."""
        if synapse_id not in self._post_spikes:
            self._post_spikes[synapse_id] = []
        self._post_spikes[synapse_id].append(time)
    
    def process_batch(self, synapse_group: SynapseGroup,
                     window: float = 100e-3) -> None:
        """
        Process all recorded spikes in batch.
        
        Args:
            synapse_group: Synapse group to update
            window: Time window for considering spike pairs
        """
        for synapse in synapse_group.synapses:
            sid = synapse.synapse_id
            
            pre_spikes = self._pre_spikes.get(sid, [])
            post_spikes = self._post_spikes.get(sid, [])
            
            # Process all spike pairs within window
            for pt in pre_spikes:
                for po in post_spikes:
                    if abs(po - pt) <= window:
                        self.apply(synapse, pt, po)
            
            # Clear recorded spikes
            if sid in self._pre_spikes:
                self._pre_spikes[sid].clear()
            if sid in self._post_spikes:
                self._post_spikes[sid].clear()
    
    def clear_history(self) -> None:
        """Clear spike history."""
        self._pre_spikes.clear()
        self._post_spikes.clear()
        self.pre_traces.clear()
        self.post_traces.clear()


@dataclass
class STDPSynapse(Synapse):
    """
    Synapse with built-in STDP learning.
    
    Extends base Synapse with STDP state and methods.
    """
    stdp_config: STDPConfig = field(default_factory=STDPConfig)
    
    # STDP state
    pre_trace: float = 0.0
    post_trace: float = 0.0
    
    def apply_plasticity(self, pre_time: float, post_time: float) -> float:
        """
        Apply STDP weight update.
        
        Args:
            pre_time: Presynaptic spike time
            post_time: Postsynaptic spike time
            
        Returns:
            New weight
        """
        delta_t = post_time - pre_time
        
        if self.stdp_config.mu == 0:
            # Additive STDP
            if delta_t > 0:
                dw = self.stdp_config.a_plus * np.exp(-delta_t / self.stdp_config.tau_plus)
                self.weight = min(self.weight + dw, self.stdp_config.w_max)
            else:
                dw = -self.stdp_config.a_minus * np.exp(delta_t / self.stdp_config.tau_minus)
                self.weight = max(self.weight + dw, self.stdp_config.w_min)
        else:
            # Multiplicative STDP
            if delta_t > 0:
                dw = self.stdp_config.a_plus * (self.stdp_config.w_max - self.weight) * \
                     np.exp(-delta_t / self.stdp_config.tau_plus)
                self.weight += dw
            else:
                dw = -self.stdp_config.a_minus * self.weight * \
                     np.exp(delta_t / self.stdp_config.tau_minus)
                self.weight += dw
            
            self.weight = max(self.stdp_config.w_min, 
                            min(self.stdp_config.w_max, self.weight))
        
        return self.weight
    
    def update_traces(self, time: float, dt: float) -> None:
        """
        Update eligibility traces using exponential decay.
        
        Args:
            time: Current time
            dt: Time step
        """
        # Decay traces
        self.pre_trace *= np.exp(-dt / self.stdp_config.tau_plus)
        self.post_trace *= np.exp(-dt / self.stdp_config.tau_minus)
    
    def on_pre_spike(self, time: float) -> float:
        """
        Handle presynaptic spike.
        
        Args:
            time: Spike time
            
        Returns:
            Weight to deliver
        """
        self.pre_trace = 1.0
        return self.get_effective_weight()
    
    def on_post_spike(self, time: float) -> None:
        """
        Handle postsynaptic spike.
        
        Args:
            time: Spike time
        """
        self.post_trace = 1.0


class HomeostaticPlasticity(SynapticPlasticity):
    """
    Homeostatic plasticity for maintaining stable network activity.
    
    Implements synaptic scaling and intrinsic plasticity to 
    regulate firing rates.
    """
    
    def __init__(self, target_rate: float = 10.0,
                 tau: float = 60.0,
                 learning_rate: float = 0.01):
        """
        Initialize homeostatic plasticity.
        
        Args:
            target_rate: Target firing rate (Hz)
            tau: Time constant for adaptation (seconds)
            learning_rate: Learning rate
        """
        super().__init__()
        self.target_rate = target_rate
        self.tau = tau
        self.learning_rate = learning_rate
        self._firing_history: Dict[int, List[float]] = {}
    
    def apply(self, synapse: Synapse, pre_time: float,
              post_time: float) -> float:
        """
        Apply homeostatic scaling (placeholder - uses firing rates).
        
        Args:
            synapse: Synapse to update
            pre_time: Ignored
            post_time: Ignored
            
        Returns:
            Current weight (scaling handled separately)
        """
        return synapse.weight
    
    def compute_scaling_factor(self, current_rate: float) -> float:
        """
        Compute multiplicative scaling factor.
        
        Args:
            current_rate: Current firing rate
            
        Returns:
            Scaling factor
        """
        if current_rate <= 0:
            return 1.0
        
        # Multiplicative scaling
        return self.target_rate / current_rate
    
    def apply_synaptic_scaling(self, synapse_group: SynapseGroup,
                              current_rate: float,
                              dt: float) -> None:
        """
        Apply synaptic scaling to all synapses.
        
        Args:
            synapse_group: Synapse group to update
            current_rate: Current average firing rate
            dt: Time step
        """
        if not self.enabled:
            return
        
        scale = self.compute_scaling_factor(current_rate)
        log_scale = np.log(scale) / self.tau * dt * self.learning_rate
        
        for synapse in synapse_group.synapses:
            synapse.weight *= np.exp(log_scale)


class RewardModulatedSTDP(SynapticPlasticity):
    """
    Reward-modulated STDP (R-STDP).
    
    Combines STDP with reward signals for reinforcement learning.
    Weight changes are gated by reward/feedback signals.
    
    References:
    -----------
    Izhikevich, E.M. (2007). Solving the distal reward problem through 
        linkage of STDP and dopamine signaling. Cerebral Cortex.
    """
    
    def __init__(self, stdp_config: Optional[STDPConfig] = None,
                 eligibility_trace_tau: float = 1.0,
                 reward_scale: float = 1.0):
        """
        Initialize reward-modulated STDP.
        
        Args:
            stdp_config: Base STDP configuration
            eligibility_trace_tau: Eligibility trace time constant
            reward_scale: Scaling factor for reward signal
        """
        super().__init__()
        self.stdp = STDPPlasticity(stdp_config)
        self.eligibility_trace_tau = eligibility_trace_tau
        self.reward_scale = reward_scale
        
        # Eligibility traces
        self.eligibility_traces: Dict[int, float] = {}
        self.pending_updates: List[Tuple[int, float]] = []
    
    def apply(self, synapse: Synapse, pre_time: float,
              post_time: float) -> float:
        """
        Apply STDP and store eligibility trace.
        
        Args:
            synapse: Synapse to update
            pre_time: Presynaptic spike time
            post_time: Postsynaptic spike time
            
        Returns:
            Current weight
        """
        # Apply standard STDP to get potential change
        delta_w = self.stdp.apply(synapse, pre_time, post_time) - synapse.weight
        
        # Store eligibility trace
        self.eligibility_traces[synapse.synapse_id] = (
            self.eligibility_traces.get(synapse.synapse_id, 0) + delta_w
        )
        
        return synapse.weight
    
    def apply_reward(self, synapse_group: SynapseGroup,
                    reward: float,
                    dt: float) -> None:
        """
        Apply reward signal to all eligible synapses.
        
        Args:
            synapse_group: Synapse group
            reward: Reward signal (can be positive or negative)
            dt: Time step
        """
        if not self.enabled:
            return
        
        for synapse in synapse_group.synapses:
            sid = synapse.synapse_id
            
            # Get and decay eligibility trace
            eligibility = self.eligibility_traces.get(sid, 0)
            eligibility *= np.exp(-dt / self.eligibility_trace_tau)
            
            # Apply reward-modulated update
            if abs(eligibility) > 1e-9:
                delta_w = self.reward_scale * reward * eligibility
                synapse.weight += delta_w
                
                # Clamp to bounds
                synapse.weight = max(self.stdp.config.w_min,
                                   min(self.stdp.config.w_max, synapse.weight))
            
            self.eligibility_traces[sid] = eligibility
    
    def decay_traces(self, dt: float) -> None:
        """Decay all eligibility traces."""
        for sid in self.eligibility_traces:
            self.eligibility_traces[sid] *= np.exp(-dt / self.eligibility_trace_tau)


class OjaLearning(SynapticPlasticity):
    """
    Oja's rule for competitive learning and normalization.
    
    dw = η * y * (x - w * y)
    
    Where x is presynaptic activity, y is postsynaptic activity,
    and w is the weight.
    """
    
    def __init__(self, learning_rate: float = 0.01,
                 normalization: float = 1.0):
        """
        Initialize Oja learning.
        
        Args:
            learning_rate: Learning rate η
            normalization: Normalization factor
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.normalization = normalization
    
    def apply(self, synapse: Synapse, pre_time: float,
              post_time: float) -> float:
        """
        Apply Oja's rule.
        
        Args:
            synapse: Synapse to update
            pre_time: Presynaptic activity (as weight proxy)
            post_time: Postsynaptic activity
            
        Returns:
            New weight
        """
        if not self.enabled:
            return synapse.weight
        
        x = pre_time  # Use times as activity proxies
        y = post_time
        
        # Oja's rule
        dw = self.learning_rate * y * (x - synapse.weight * y / self.normalization)
        synapse.weight += dw
        
        return synapse.weight


def create_stdp_synapse_group(n_synapses: int,
                              config: Optional[STDPConfig] = None,
                              seed: Optional[int] = None) -> Tuple[SynapseGroup, List[STDPSynapse]]:
    """
    Create group of STDP synapses.
    
    Args:
        n_synapses: Number of synapses
        config: STDP configuration
        seed: Random seed
        
    Returns:
        Tuple of (SynapseGroup, list of STDPSynapses)
    """
    rng = np.random.RandomState(seed)
    group = SynapseGroup(group_id=0)
    stdp_synapses = []
    
    for i in range(n_synapses):
        # Create STDP synapse
        synapse = STDPSynapse(
            synapse_id=i,
            pre_neuron_id=i % 100,  # Placeholder
            post_neuron_id=(i % 100) + 1,
            weight=rng.rand() * 0.5 + 0.1,
            delay=abs(rng.randn() * 2e-3 + 5e-3),
            stdp_config=config or STDPConfig(),
        )
        stdp_synapses.append(synapse)
        group.add_synapse(synapse)
    
    return group, stdp_synapses
