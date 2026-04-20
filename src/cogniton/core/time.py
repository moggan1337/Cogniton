"""
Time Management Module
======================

Manages simulation time for event-driven and fixed-timestep simulations.
"""

from dataclasses import dataclass, field
from typing import Optional
import time as system_time


@dataclass
class SimulationTime:
    """
    Manages simulation time and timing utilities.
    
    Attributes:
        t: Current simulation time (seconds)
        t_start: Start time of simulation
        t_end: End time of simulation
        dt: Fixed timestep (seconds)
        real_time_factor: Ratio of sim time to real time
        wall_time_start: Wall clock start time
    """
    t: float = 0.0
    t_start: float = 0.0
    t_end: float = 1.0
    dt: float = 1e-4
    _real_time_factor: float = field(default=0.0, repr=False)
    _wall_time_start: float = field(default=0.0, repr=False)
    _sim_time_start: float = field(default=0.0, repr=False)
    
    def __post_init__(self):
        """Initialize timing state."""
        self._wall_time_start = system_time.perf_counter()
        self._sim_time_start = self.t_start
    
    def reset(self) -> None:
        """Reset simulation time to start."""
        self.t = self.t_start
        self._wall_time_start = system_time.perf_counter()
        self._sim_time_start = self.t_start
        self._real_time_factor = 0.0
    
    def advance(self, dt: Optional[float] = None) -> float:
        """
        Advance simulation time.
        
        Args:
            dt: Time step to advance (uses self.dt if None)
            
        Returns:
            New simulation time
        """
        if dt is None:
            dt = self.dt
        
        self.t = min(self.t + dt, self.t_end)
        
        # Calculate real-time factor
        wall_elapsed = system_time.perf_counter() - self._wall_time_start
        sim_elapsed = self.t - self._sim_time_start
        
        if wall_elapsed > 0:
            self._real_time_factor = sim_elapsed / wall_elapsed
        
        return self.t
    
    def set_time(self, t: float) -> None:
        """
        Set simulation time directly.
        
        Args:
            t: New simulation time
        """
        self.t = max(self.t_start, min(t, self.t_end))
    
    @property
    def progress(self) -> float:
        """Get simulation progress (0 to 1)."""
        total = self.t_end - self.t_start
        if total <= 0:
            return 1.0
        return (self.t - self.t_start) / total
    
    @property
    def is_finished(self) -> bool:
        """Check if simulation has finished."""
        return self.t >= self.t_end
    
    @property
    def remaining_time(self) -> float:
        """Get remaining simulation time."""
        return max(0.0, self.t_end - self.t)
    
    @property
    def real_time_factor(self) -> float:
        """Get real-time factor (sim seconds per real second)."""
        return self._real_time_factor
    
    def time_to_next_event(self, event_times: list) -> Optional[float]:
        """
        Calculate time until next event.
        
        Args:
            event_times: List of event times
            
        Returns:
            Time until next event, or None if no events
        """
        future_events = [et for et in event_times if et > self.t]
        
        if not future_events:
            return None
        
        return min(future_events) - self.t
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"SimulationTime(t={self.t:.6f}, "
                f"t_end={self.t_end:.6f}, "
                f"progress={self.progress:.1%})")
