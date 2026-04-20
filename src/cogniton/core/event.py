"""
Event-Driven Simulation Module
==============================

Implements event-driven simulation with priority queue for efficient
neuromorphic computation.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Set, Tuple
from enum import Enum, auto
from heapq import heappush, heappop
import numpy as np


class EventType(Enum):
    """Types of events in the simulation."""
    SPIKE = auto()
    SYNAPSE_UPDATE = auto()
    WEIGHT_UPDATE = auto()
    INPUT = auto()
    EXTERNAL = auto()
    MONITOR = auto()
    CHECKPOINT = auto()


@dataclass(order=True)
class Event:
    """
    Represents a simulation event.
    
    Attributes:
        time: Event time (seconds)
        event_type: Type of event
        source_id: Source neuron/synapse ID
        target_id: Target neuron/synapse ID
        data: Additional event data (dict)
        priority: Event priority for tie-breaking
    """
    time: float
    event_type: EventType = field(compare=False)
    source_id: int = field(compare=False, default=0)
    target_id: int = field(compare=False, default=0)
    data: Dict[str, Any] = field(compare=False, default_factory=dict)
    priority: int = field(compare=False, default=0)
    
    def __post_init__(self):
        """Ensure time is non-negative."""
        if self.time < 0:
            raise ValueError("Event time must be non-negative")


class EventQueue:
    """
    Priority queue for event-driven simulation.
    
    Uses a min-heap for O(log n) insertion and extraction.
    """
    
    def __init__(self):
        """Initialize empty event queue."""
        self._heap: List[Event] = []
        self._event_count: int = 0
    
    def push(self, event: Event) -> None:
        """
        Add event to queue.
        
        Args:
            event: Event to add
        """
        heappush(self._heap, event)
        self._event_count += 1
    
    def pop(self) -> Optional[Event]:
        """
        Remove and return earliest event.
        
        Returns:
            Earliest event, or None if queue is empty
        """
        if not self._heap:
            return None
        
        self._event_count -= 1
        return heappop(self._heap)
    
    def peek(self) -> Optional[Event]:
        """
        View earliest event without removing.
        
        Returns:
            Earliest event, or None if queue is empty
        """
        if not self._heap:
            return None
        return self._heap[0]
    
    def get_next_time(self) -> Optional[float]:
        """
        Get time of earliest event.
        
        Returns:
            Time of earliest event, or None if queue is empty
        """
        if not self._heap:
            return None
        return self._heap[0].time
    
    def clear(self) -> None:
        """Clear all events from queue."""
        self._heap.clear()
        self._event_count = 0
    
    def clear_until(self, t: float) -> int:
        """
        Remove all events before time t.
        
        Args:
            t: Time threshold
            
        Returns:
            Number of events removed
        """
        removed = 0
        while self._heap and self._heap[0].time < t:
            heappop(self._heap)
            removed += 1
            self._event_count -= 1
        return removed
    
    @property
    def size(self) -> int:
        """Number of events in queue."""
        return len(self._heap)
    
    @property
    def total_events_processed(self) -> int:
        """Total events ever pushed to queue."""
        return self._event_count
    
    def get_events_in_range(self, t_start: float, t_end: float) -> List[Event]:
        """
        Get all events in time range (inefficient, for debugging).
        
        Args:
            t_start: Start time
            t_end: End time
            
        Returns:
            List of events in range
        """
        return [e for e in self._heap if t_start <= e.time <= t_end]
    
    def __len__(self) -> int:
        """Number of events in queue."""
        return len(self._heap)
    
    def __bool__(self) -> bool:
        """Check if queue has events."""
        return len(self._heap) > 0


class EventHandler:
    """
    Handler for processing events.
    
    Defines callback methods for different event types.
    """
    
    def __init__(self):
        """Initialize event handler."""
        self._handlers: Dict[EventType, List[Callable]] = {
            event_type: [] for event_type in EventType
        }
    
    def register_handler(self, event_type: EventType, 
                         handler: Callable[[Event], None]) -> None:
        """
        Register event handler callback.
        
        Args:
            event_type: Type of event to handle
            handler: Callback function (receives Event)
        """
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)
    
    def unregister_handler(self, event_type: EventType,
                           handler: Callable[[Event], None]) -> None:
        """
        Unregister event handler.
        
        Args:
            event_type: Event type
            handler: Handler to remove
        """
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
    
    def handle(self, event: Event) -> None:
        """
        Process event through registered handlers.
        
        Args:
            event: Event to process
        """
        for handler in self._handlers[event.event_type]:
            handler(event)
    
    def clear_handlers(self, event_type: Optional[EventType] = None) -> None:
        """
        Clear handlers.
        
        Args:
            event_type: Specific type to clear, or None for all
        """
        if event_type is None:
            for et in EventType:
                self._handlers[et].clear()
        else:
            self._handlers[event_type].clear()


@dataclass
class EventStatistics:
    """Statistics about event processing."""
    spikes_processed: int = 0
    synapse_updates: int = 0
    weight_updates: int = 0
    external_inputs: int = 0
    total_events: int = 0
    
    def increment(self, event_type: EventType, count: int = 1) -> None:
        """Increment counter for event type."""
        self.total_events += count
        
        if event_type == EventType.SPIKE:
            self.spikes_processed += count
        elif event_type == EventType.SYNAPSE_UPDATE:
            self.synapse_updates += count
        elif event_type == EventType.WEIGHT_UPDATE:
            self.weight_updates += count
        elif event_type == EventType.INPUT:
            self.external_inputs += count
    
    def summary(self) -> Dict[str, int]:
        """Get summary as dictionary."""
        return {
            "total_events": self.total_events,
            "spikes_processed": self.spikes_processed,
            "synapse_updates": self.synapse_updates,
            "weight_updates": self.weight_updates,
            "external_inputs": self.external_inputs,
        }


class EventDrivenSimulation:
    """
    Main event-driven simulation engine.
    
    Coordinates event processing, time advancement, and state updates.
    """
    
    def __init__(self, t_max: float = 1.0, dt: float = 1e-4,
                 record_events: bool = True):
        """
        Initialize event-driven simulation.
        
        Args:
            t_max: Maximum simulation time
            dt: Default timestep for hybrid mode
            record_events: Whether to record all events
        """
        self.t_max = t_max
        self.dt = dt
        self.record_events = record_events
        
        self.event_queue = EventQueue()
        self.handler = EventHandler()
        self.stats = EventStatistics()
        
        self._t: float = 0.0
        self._running: bool = False
        self._event_history: List[Event] = []
        self._spike_history: Dict[int, List[Tuple[float, int]]] = {}  # neuron_id -> [(time, target_id), ...]
        
        # Registered simulation components
        self._neurons: Dict[int, Any] = {}
        self._synapses: Dict[int, Any] = {}
        
    @property
    def t(self) -> float:
        """Current simulation time."""
        return self._t
    
    @property
    def is_running(self) -> bool:
        """Check if simulation is running."""
        return self._running
    
    @property
    def spike_history(self) -> Dict[int, List[Tuple[float, int]]]:
        """Get spike history."""
        return self._spike_history
    
    @property
    def event_history(self) -> List[Event]:
        """Get event history (if recording enabled)."""
        return self._event_history
    
    def register_neuron(self, neuron_id: int, neuron: Any) -> None:
        """Register neuron for spike event handling."""
        self._neurons[neuron_id] = neuron
    
    def register_synapse(self, synapse_id: int, synapse: Any) -> None:
        """Register synapse for event handling."""
        self._synapses[synapse_id] = synapse
    
    def schedule_event(self, event: Event) -> None:
        """
        Schedule an event for processing.
        
        Args:
            event: Event to schedule
        """
        if event.time < self._t:
            raise ValueError(f"Cannot schedule event in past: {event.time} < {self._t}")
        
        self.event_queue.push(event)
    
    def schedule_spike(self, source_id: int, time: float, 
                      target_ids: List[int], 
                      weight: float = 1.0,
                      delay: float = 0.0) -> None:
        """
        Schedule spike events for multiple targets.
        
        Args:
            source_id: Source neuron ID
            time: Spike time
            target_ids: Target neuron IDs
            weight: Synaptic weight
            delay: Axonal delay
        """
        spike_time = time + delay
        
        # Create spike event
        spike_event = Event(
            time=spike_time,
            event_type=EventType.SPIKE,
            source_id=source_id,
            data={"weight": weight, "original_time": time}
        )
        self.schedule_event(spike_event)
        
        # Record in spike history
        if source_id not in self._spike_history:
            self._spike_history[source_id] = []
        self._spike_history[source_id].append((time, target_ids))
    
    def schedule_periodic(self, interval: float, handler: Callable,
                          start_time: Optional[float] = None) -> None:
        """
        Schedule periodic event handler.
        
        Args:
            interval: Interval between calls
            handler: Handler function
            start_time: First call time (defaults to current time)
        """
        if start_time is None:
            start_time = self._t
        
        event = Event(
            time=start_time,
            event_type=EventType.EXTERNAL,
            data={"handler": handler, "interval": interval, 
                  "is_periodic": True}
        )
        self.schedule_event(event)
    
    def _process_event(self, event: Event) -> None:
        """
        Process a single event.
        
        Args:
            event: Event to process
        """
        # Record event if enabled
        if self.record_events:
            self._event_history.append(event)
        
        # Update statistics
        self.stats.increment(event.event_type)
        
        # Handle based on event type
        if event.event_type == EventType.SPIKE:
            self._handle_spike(event)
        elif event.event_type == EventType.SYNAPSE_UPDATE:
            self._handle_synapse_update(event)
        elif event.event_type == EventType.WEIGHT_UPDATE:
            self._handle_weight_update(event)
        elif event.event_type == EventType.EXTERNAL:
            self._handle_external(event)
        
        # Call registered handlers
        self.handler.handle(event)
    
    def _handle_spike(self, event: Event) -> None:
        """Handle spike event."""
        # Deliver to target neurons
        if event.target_id in self._neurons:
            target = self._neurons[event.target_id]
            if hasattr(target, 'receive_spike'):
                target.receive_spike(event.time, event.data.get('weight', 0))
    
    def _handle_synapse_update(self, event: Event) -> None:
        """Handle synaptic update event."""
        if event.target_id in self._synapses:
            synapse = self._synapses[event.target_id]
            if hasattr(synapse, 'update'):
                synapse.update(event.time)
    
    def _handle_weight_update(self, event: Event) -> None:
        """Handle weight update event (e.g., STDP)."""
        if event.target_id in self._synapses:
            synapse = self._synapses[event.target_id]
            if hasattr(synapse, 'apply_plasticity'):
                pre_time = event.data.get('pre_time', event.time)
                post_time = event.data.get('post_time', event.time)
                synapse.apply_plasticity(pre_time, post_time)
    
    def _handle_external(self, event: Event) -> None:
        """Handle external event with optional periodic scheduling."""
        if "handler" in event.data:
            handler = event.data["handler"]
            handler(event.time)
            
            # Reschedule if periodic
            if event.data.get("is_periodic", False):
                interval = event.data["interval"]
                new_event = Event(
                    time=event.time + interval,
                    event_type=EventType.EXTERNAL,
                    data=event.data
                )
                self.schedule_event(new_event)
    
    def step(self) -> bool:
        """
        Execute single simulation step.
        
        Returns:
            True if step executed, False if simulation finished
        """
        if self._t >= self.t_max:
            return False
        
        # Get next event time
        next_time = self.event_queue.get_next_time()
        
        if next_time is None:
            # No more events, advance to end
            self._t = self.t_max
            return False
        
        # Check if next event is beyond simulation time
        if next_time > self.t_max:
            self._t = self.t_max
            return False
        
        # Process event and advance time
        event = self.event_queue.pop()
        if event is not None:
            self._t = event.time
            self._process_event(event)
        
        return True
    
    def run(self, callbacks: Optional[List[Callable[[float], None]]] = None,
            progress: bool = True) -> None:
        """
        Run simulation until completion.
        
        Args:
            callbacks: Optional progress callbacks (called with time)
            progress: Whether to print progress
        """
        self._running = True
        
        if progress:
            print(f"Running event-driven simulation (t_max={self.t_max}s)")
        
        step_count = 0
        while self._t < self.t_max and self.event_queue:
            self.step()
            step_count += 1
            
            # Call progress callbacks
            if callbacks:
                for cb in callbacks:
                    cb(self._t)
            
            if progress and step_count % 10000 == 0:
                print(f"  t={self._t:.4f}s, events={self.stats.total_events}")
        
        self._running = False
        
        if progress:
            print(f"Simulation complete: {step_count} steps, "
                  f"{self.stats.total_events} events")
    
    def reset(self) -> None:
        """Reset simulation to initial state."""
        self._t = 0.0
        self._running = False
        self.event_queue.clear()
        self.stats = EventStatistics()
        self._event_history.clear()
        self._spike_history.clear()
    
    def get_neurons_firing_at(self, t: float, tolerance: float = 1e-6) -> List[int]:
        """
        Get IDs of neurons that fired at approximately time t.
        
        Args:
            t: Target time
            tolerance: Time tolerance
            
        Returns:
            List of neuron IDs
        """
        firing = []
        for neuron_id, spikes in self._spike_history.items():
            for spike_time, _ in spikes:
                if abs(spike_time - t) <= tolerance:
                    firing.append(neuron_id)
                    break
        return firing
