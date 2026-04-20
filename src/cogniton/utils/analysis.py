"""
Analysis Utilities
=================

Functions for analyzing spiking neural network simulation results.
"""

from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from scipy import signal
from scipy.stats import pearsonr, spearmanr


def compute_firing_rate(spike_times: np.ndarray,
                       neuron_id: int,
                       window_size: float = 0.1,
                       dt: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute instantaneous firing rate using kernel density estimation.
    
    Args:
        spike_times: Array of spike times for the neuron
        neuron_id: Neuron ID
        window_size: Window size for rate calculation
        dt: Time step
        
    Returns:
        Tuple of (time, rate) arrays
    """
    if len(spike_times) == 0:
        return np.array([]), np.array([])
    
    # Create time bins
    t_start = spike_times[0] if len(spike_times) > 0 else 0
    t_end = spike_times[-1] if len(spike_times) > 0 else 1
    time = np.arange(t_start, t_end, dt)
    
    # Gaussian kernel
    sigma = window_size / 6  # 6-sigma window
    kernel = np.exp(-0.5 * (np.arange(-3*sigma/dt, 3*sigma/dt) * dt)**2 / sigma**2)
    kernel = kernel / np.sum(kernel)
    
    # Convolve with spike train
    spike_train = np.zeros(len(time))
    for st in spike_times:
        idx = int((st - t_start) / dt)
        if 0 <= idx < len(spike_train):
            spike_train[idx] = 1
    
    rate = np.convolve(spike_train, kernel, mode='same') / (dt * len(spike_times) + 1e-9)
    
    return time, rate


def compute_population_rate(spike_raster: List[Tuple[float, int]],
                           window_size: float = 0.1,
                           dt: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute population firing rate.
    
    Args:
        spike_raster: List of (time, neuron_id) tuples
        window_size: Window size
        dt: Time step
        
    Returns:
        Tuple of (time, population_rate) arrays
    """
    if not spike_raster:
        return np.array([]), np.array([])
    
    times = np.array([t for t, _ in spike_raster])
    ids = np.array([i for _, i in spike_raster])
    
    # Time bins
    t_start = times.min()
    t_end = times.max()
    time = np.arange(t_start, t_end, dt)
    
    # Gaussian kernel
    sigma = window_size / 6
    kernel = np.exp(-0.5 * (np.arange(-3*sigma/dt, 3*sigma/dt) * dt)**2 / sigma**2)
    kernel = kernel / np.sum(kernel)
    
    # Bin spikes
    bins = np.digitize(times, time)
    hist = np.bincount(bins, minlength=len(time))
    
    # Convolve
    rate = np.convolve(hist, kernel, mode='same')
    
    # Normalize by number of neurons and dt
    n_neurons = len(np.unique(ids))
    if n_neurons > 0:
        rate = rate / (window_size * n_neurons)
    
    return time, rate


def compute_isi(spike_times: np.ndarray) -> np.ndarray:
    """
    Compute inter-spike intervals.
    
    Args:
        spike_times: Array of spike times
        
    Returns:
        Array of ISIs
    """
    if len(spike_times) < 2:
        return np.array([])
    
    return np.diff(spike_times)


def compute_cv(isi: np.ndarray) -> float:
    """
    Compute coefficient of variation of ISIs.
    
    Args:
        isi: Inter-spike intervals
        
    Returns:
        CV (std/mean)
    """
    if len(isi) < 2:
        return 0.0
    
    return np.std(isi) / (np.mean(isi) + 1e-9)


def compute_correlation(spike_times_a: np.ndarray,
                       spike_times_b: np.ndarray,
                       dt: float = 0.001,
                       max_lag: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-correlation between two spike trains.
    
    Args:
        spike_times_a: Spike times for neuron A
        spike_times_b: Spike times for neuron B
        dt: Time bin size
        max_lag: Maximum lag to compute
        
    Returns:
        Tuple of (lags, correlation) arrays
    """
    if len(spike_times_a) == 0 or len(spike_times_b) == 0:
        return np.array([]), np.array([])
    
    # Create binary spike trains
    t_start = min(spike_times_a.min(), spike_times_b.min())
    t_end = max(spike_times_a.max(), spike_times_b.max())
    time = np.arange(t_start, t_end, dt)
    
    train_a = np.zeros(len(time))
    train_b = np.zeros(len(time))
    
    for st in spike_times_a:
        idx = int((st - t_start) / dt)
        if 0 <= idx < len(train_a):
            train_a[idx] = 1
    
    for st in spike_times_b:
        idx = int((st - t_start) / dt)
        if 0 <= idx < len(train_b):
            train_b[idx] = 1
    
    # Cross-correlation
    correlation = np.correlate(train_a, train_b, mode='full')
    lags = np.arange(-len(time)+1, len(time)) * dt
    
    # Trim to max_lag
    mask = np.abs(lags) <= max_lag
    lags = lags[mask]
    correlation = correlation[mask]
    
    # Normalize
    if len(correlation) > 0:
        correlation = correlation / (len(time) * dt)
    
    return lags, correlation


def compute_pairwise_correlation(spike_raster: List[Tuple[float, int]],
                                neuron_ids: List[int],
                                dt: float = 0.001) -> np.ndarray:
    """
    Compute pairwise correlation matrix.
    
    Args:
        spike_raster: Spike raster data
        neuron_ids: List of neuron IDs to analyze
        dt: Time bin size
        
    Returns:
        Correlation matrix
    """
    n = len(neuron_ids)
    corr_matrix = np.zeros((n, n))
    
    # Group spikes by neuron
    spikes_by_neuron = {nid: [] for nid in neuron_ids}
    for t, nid in spike_raster:
        if nid in spikes_by_neuron:
            spikes_by_neuron[nid].append(t)
    
    # Compute pairwise correlations
    for i, nid_a in enumerate(neuron_ids):
        for j, nid_b in enumerate(neuron_ids):
            if i <= j:
                times_a = np.array(spikes_by_neuron[nid_a])
                times_b = np.array(spikes_by_neuron[nid_b])
                
                _, corr = compute_correlation(times_a, times_b, dt)
                corr_matrix[i, j] = corr[len(corr)//2] if len(corr) > 0 else 0
                corr_matrix[j, i] = corr_matrix[i, j]
    
    return corr_matrix


def compute_raster_statistics(spike_raster: List[Tuple[float, int]]) -> Dict[str, Any]:
    """
    Compute statistics from spike raster.
    
    Args:
        spike_raster: Spike raster data
        
    Returns:
        Dictionary of statistics
    """
    if not spike_raster:
        return {
            "total_spikes": 0,
            "n_neurons": 0,
            "mean_rate": 0,
            "cv_isi": 0,
        }
    
    times = np.array([t for t, _ in spike_raster])
    ids = np.array([i for _, i in spike_raster])
    
    # Unique neurons
    unique_neurons = np.unique(ids)
    n_neurons = len(unique_neurons)
    
    # Time span
    t_span = times.max() - times.min() if len(times) > 0 else 1
    
    # Total spikes
    total_spikes = len(spike_raster)
    
    # Mean rate
    mean_rate = total_spikes / (n_neurons * t_span) if n_neurons > 0 else 0
    
    # Firing rates per neuron
    rates = []
    for nid in unique_neurons:
        n_spikes = np.sum(ids == nid)
        rate = n_spikes / t_span
        rates.append(rate)
    
    # CV of firing rates
    cv_rates = np.std(rates) / (np.mean(rates) + 1e-9) if len(rates) > 0 else 0
    
    return {
        "total_spikes": total_spikes,
        "n_neurons": n_neurons,
        "mean_rate": mean_rate,
        "std_rate": np.std(rates) if len(rates) > 0 else 0,
        "min_rate": np.min(rates) if len(rates) > 0 else 0,
        "max_rate": np.max(rates) if len(rates) > 0 else 0,
        "cv_rate": cv_rates,
        "t_start": times.min() if len(times) > 0 else 0,
        "t_end": times.max() if len(times) > 0 else 0,
        "t_span": t_span,
    }


def compute_synchrony(spike_raster: List[Tuple[float, int]],
                     bin_size: float = 0.005) -> float:
    """
    Compute spike synchrony (pairwise correlation at lag 0).
    
    Args:
        spike_raster: Spike raster data
        bin_size: Time bin size
        
    Returns:
        Synchrony index (0-1)
    """
    if not spike_raster:
        return 0.0
    
    times = np.array([t for t, _ in spike_raster])
    ids = np.array([i for _, i in spike_raster])
    
    # Bin spikes
    t_start = times.min()
    t_end = times.max()
    bins = np.arange(t_start, t_end + bin_size, bin_size)
    
    # Count spikes per bin per neuron
    n_bins = len(bins) - 1
    unique_ids = np.unique(ids)
    n_neurons = len(unique_ids)
    
    # Spike count per bin
    spike_counts = np.histogram(times, bins=bins)[0]
    
    # Synchrony: variance of population spike count
    mean_count = np.mean(spike_counts)
    var_count = np.var(spike_counts)
    
    if mean_count > 0:
        # Fano factor-like synchrony measure
        synchrony = var_count / (mean_count * n_neurons + 1e-9)
    else:
        synchrony = 0.0
    
    return np.clip(synchrony, 0, 1)


def compute_connectivity_from_spikes(spike_times: np.ndarray,
                                     neuron_ids: np.ndarray,
                                     window: float = 10e-3,
                                     method: str = "correlation") -> np.ndarray:
    """
    Estimate functional connectivity from spike data.
    
    Args:
        spike_times: Array of spike times
        neuron_ids: Array of neuron IDs
        window: Correlation window
        method: Method to use ('correlation', 'transfer_entropy')
        
    Returns:
        Estimated connectivity matrix
    """
    # Get unique neurons
    unique_ids = np.unique(neuron_ids)
    n = len(unique_ids)
    
    # Create ID mapping
    id_to_idx = {nid: i for i, nid in enumerate(unique_ids)}
    
    # Initialize connectivity
    conn = np.zeros((n, n))
    
    if method == "correlation":
        # Compute pairwise correlations
        for i, nid_a in enumerate(unique_ids):
            for j, nid_b in enumerate(unique_ids):
                if i != j:
                    times_a = spike_times[neuron_ids == nid_a]
                    times_b = spike_times[neuron_ids == nid_b]
                    
                    _, corr = compute_correlation(times_a, times_b)
                    
                    # Take correlation at small positive lag
                    if len(corr) > 0:
                        conn[i, j] = corr[len(corr)//4:len(corr)//2].mean()
    
    return conn
