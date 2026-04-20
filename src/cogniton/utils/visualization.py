"""
Visualization Utilities
=======================

Plotting functions for spiking neural network simulation results.
"""

from typing import Optional, List, Tuple, Dict, Any
import numpy as np

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_raster(spike_times: np.ndarray,
                neuron_ids: np.ndarray,
                title: str = "Spike Raster Plot",
                xlabel: str = "Time (s)",
                ylabel: str = "Neuron ID",
                ax: Optional[Any] = None,
                markersize: float = 1.0,
                color: str = "black",
                alpha: float = 0.5) -> Any:
    """
    Plot spike raster.
    
    Args:
        spike_times: Array of spike times
        neuron_ids: Array of neuron IDs
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        ax: Matplotlib axis (creates new if None)
        markersize: Marker size
        color: Marker color
        alpha: Marker transparency
        
    Returns:
        Matplotlib figure and axis
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping plot.")
        return None, None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure
    
    ax.scatter(spike_times, neuron_ids, s=markersize, c=color, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_voltage(time: np.ndarray,
                voltage: np.ndarray,
                label: str = "Membrane Potential",
                title: str = "Membrane Potential vs Time",
                v_thresh: Optional[float] = None,
                v_reset: Optional[float] = None,
                ax: Optional[Any] = None) -> Tuple[Any, Any]:
    """
    Plot membrane potential over time.
    
    Args:
        time: Time array
        voltage: Voltage array
        label: Line label
        title: Plot title
        v_thresh: Threshold voltage (optional, plotted as horizontal line)
        v_reset: Reset voltage (optional, plotted as horizontal line)
        ax: Matplotlib axis
        
    Returns:
        Figure and axis
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping plot.")
        return None, None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure
    
    ax.plot(time, voltage, label=label)
    
    if v_thresh is not None:
        ax.axhline(y=v_thresh, color='r', linestyle='--', 
                   label=f'Threshold ({v_thresh} mV)')
    
    if v_reset is not None:
        ax.axhline(y=v_reset, color='g', linestyle=':',
                   label=f'Reset ({v_reset} mV)')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Membrane Potential (mV)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_weights(weight_matrix: np.ndarray,
                title: str = "Synaptic Weight Matrix",
                cmap: str = "RdBu_r",
                vmin: Optional[float] = None,
                vmax: Optional[float] = None,
                ax: Optional[Any] = None) -> Tuple[Any, Any]:
    """
    Plot synaptic weight matrix as heatmap.
    
    Args:
        weight_matrix: 2D weight matrix (pre x post)
        title: Plot title
        cmap: Colormap
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        ax: Matplotlib axis
        
    Returns:
        Figure and axis
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping plot.")
        return None, None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    im = ax.imshow(weight_matrix, cmap=cmap, aspect='auto',
                   vmin=vmin, vmax=vmax)
    ax.set_xlabel('Postsynaptic Neuron')
    ax.set_ylabel('Presynaptic Neuron')
    ax.set_title(title)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weight')
    
    plt.tight_layout()
    return fig, ax


def plot_firing_rate_distribution(firing_rates: np.ndarray,
                                   bins: int = 50,
                                   title: str = "Firing Rate Distribution",
                                   ax: Optional[Any] = None) -> Tuple[Any, Any]:
    """
    Plot histogram of firing rates.
    
    Args:
        firing_rates: Array of firing rates
        bins: Number of histogram bins
        title: Plot title
        ax: Matplotlib axis
        
    Returns:
        Figure and axis
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping plot.")
        return None, None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    
    ax.hist(firing_rates, bins=bins, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Firing Rate (Hz)')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_connectivity(connectivity: np.ndarray,
                     title: str = "Network Connectivity",
                     ax: Optional[Any] = None) -> Tuple[Any, Any]:
    """
    Plot binary connectivity matrix.
    
    Args:
        connectivity: Binary connectivity matrix
        title: Plot title
        ax: Matplotlib axis
        
    Returns:
        Figure and axis
    """
    return plot_weights(connectivity, title=title, cmap='binary')


def plot_stdp_curve(tau_plus: float = 20e-3,
                    tau_minus: float = 20e-3,
                    a_plus: float = 0.01,
                    a_minus: float = 0.012,
                    dt_max: float = 100e-3,
                    ax: Optional[Any] = None) -> Tuple[Any, Any]:
    """
    Plot STDP learning window.
    
    Args:
        tau_plus: Potentiation time constant
        tau_minus: Depression time constant
        a_plus: Potentiation amplitude
        a_minus: Depression amplitude
        dt_max: Maximum time difference
        ax: Matplotlib axis
        
    Returns:
        Figure and axis
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping plot.")
        return None, None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    
    dt = np.linspace(-dt_max, dt_max, 1000) * 1000  # Convert to ms
    
    potentiation = a_plus * np.exp(dt / (tau_plus * 1000))
    depression = -a_minus * np.exp(-dt / (tau_minus * 1000))
    
    potentiation[dt < 0] = 0
    depression[dt > 0] = 0
    
    delta_w = potentiation + depression
    
    ax.plot(dt, delta_w * 1000, 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    ax.fill_between(dt[dt > 0], 0, potentiation[dt > 0] * 1000, 
                    alpha=0.3, color='blue', label='Potentiation')
    ax.fill_between(dt[dt < 0], 0, depression[dt < 0] * 1000,
                    alpha=0.3, color='red', label='Depression')
    
    ax.set_xlabel('Δt = t_post - t_pre (ms)')
    ax.set_ylabel('Δw (a.u.)')
    ax.set_title('STDP Learning Window')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_neuron_dynamics(time: np.ndarray,
                        v_mem: np.ndarray,
                        i_syn: np.ndarray,
                        title: str = "Neuron Dynamics",
                        v_thresh: Optional[float] = None) -> Tuple[Any, Any]:
    """
    Plot membrane potential and synaptic current.
    
    Args:
        time: Time array
        v_mem: Membrane potential
        i_syn: Synaptic current
        title: Plot title
        v_thresh: Threshold voltage
        
    Returns:
        Figure and axes
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping plot.")
        return None, None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    ax1.plot(time * 1000, v_mem, 'b-', linewidth=1)
    if v_thresh is not None:
        ax1.axhline(y=v_thresh, color='r', linestyle='--', 
                    label=f'Threshold ({v_thresh} mV)')
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(time * 1000, i_syn * 1000, 'g-', linewidth=1)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Synaptic Current (nA)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, (ax1, ax2)


def save_figure(fig: Any, filepath: str, dpi: int = 300,
               transparent: bool = False) -> None:
    """
    Save figure to file.
    
    Args:
        fig: Matplotlib figure
        filepath: Output path
        dpi: Resolution
        transparent: Transparent background
    """
    if MATPLOTLIB_AVAILABLE and fig is not None:
        fig.savefig(filepath, dpi=dpi, transparent=transparent)
        print(f"Figure saved to {filepath}")
    else:
        print("Could not save figure.")
