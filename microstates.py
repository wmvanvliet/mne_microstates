"""
Functions to segment EEG into microstates. Based on the Microsegment toolbox
for EEGlab, written by Andreas Trier Poulsen [1]_.

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>

References
----------
.. [1]  Poulsen, A. T., Pedroni, A., Langer, N., &  Hansen, L. K. (2018).
        Microstate EEGlab toolbox: An introductionary guide. bioRxiv.
"""
import warnings
import numpy as np
from scipy.stats import zscore
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import mne
from mne.utils import logger, verbose


@verbose
def microstates_raw(raw, n_states=4, n_inits=10, max_iter=1000, thresh=1e-6,
                    normalize=False, min_peak_dist=0, max_n_peaks=10000,
                    random_state=None, verbose=None):
    """Segment a continuous signal into microstates.

    Peaks in the global field power (GFP) are used to find microstates, using a
    modified K-means algorithm. Several runs of the modified K-means algorithms
    are performed, using different random initializations. The run that
    resulted in the best segmentation, as measured by global explained variance
    (GEV), is used.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
        The data to find the microstates in
    n_states : int
        The number of unique microstates to find. Defaults to 4.
    n_inits : int
        The number of random initializations to use for the k-means algorithm.
        The best fitting segmentation across all initializations is used.
        Defaults to 10.
    max_iter : int
        The maximum number of iterations to perform in the k-means algorithm.
        Defaults to 1000.
    thresh : float
        The threshold of convergence for the k-means algorithm, based on
        relative change in noise variance. Defaults to 1e-6.
    normalize : bool
        Whether to normalize (z-score) the data across time before running the
        k-means algorithm. Defaults to ``False``.
    min_peak_dist : float
        Minimum distance (in seconds) between peaks in the GFP. Defaults to 0.
    max_n_peaks : int
        Maximum number of GFP peaks to use in the k-means algorithm. Chosen
        randomly. Defaults to 10000. 
    random_state : int | None
        The seed for the random number generator. Defaults to ``None``, in
        which case a different seed is chosen each time this function is
        called.
    verbose : int | bool | None
        Controls the verbosity.

    Returns
    -------
    maps : ndarray, shape (n_channels, n_states)
        The topographic maps of the found unique microstates.
    assignment : ndarray, shape (n_samples,)
        For each sample, the index of the microstate to which the sample has
        been assigned.

    References
    ----------
    .. [1] Pascual-Marqui, R. D., Michel, C. M., & Lehmann, D. (1995).
           Segmentation of brain electrical activity into microstates: model
           estimation and validation. IEEE Transactions on Biomedical
           Engineering.
    """
    # Convert min_peak_dist to samples
    min_peak_dist = 1 + int(round(min_peak_dist * raw.info['sfreq']))

    # Find peaks in the global field power (GFP)
    gfp = np.std(raw.get_data(), axis=0)
    peaks, _ = find_peaks(gfp, distance=min_peak_dist)
    n_peaks = len(peaks)

    # Limit the number of peaks by randomly selecting them
    if max_n_peaks is not None:
        max_n_peaks = min(n_peaks, max_n_peaks)
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        chosen_peaks = random_state.choice(n_peaks, size=max_n_peaks,
                                           replace=False)
        peaks = peaks[chosen_peaks]

    # Run microstates analysis on selected data
    data = raw.get_data()[:, peaks]
    return microstates_array(data, n_states, n_inits, max_iter, thresh,
                             normalize, random_state, verbose)


@verbose
def microstates_array(data, n_states=4, n_inits=10, max_iter=1000, thresh=1e-6,
                      normalize=False, random_state=None, verbose=None):
    """Segment a signal into microstates.

    Several runs of the modified K-means algorithms are performed, using
    different random initializations. The run that resulted in the best
    segmentation, as measured by global explained variance (GEV), is used.

    Notes
    -----
    This function performs no selection of the data.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
        The data to find the microstates in
    n_states : int
        The number of unique microstates to find. Defaults to 4.
    n_inits : int
        The number of random initializations to use for the k-means algorithm.
        The best fitting segmentation across all initializations is used.
        Defaults to 10.
    max_iter : int
        The maximum number of iterations to perform in the k-means algorithm.
        Defaults to 1000.
    thresh : float
        The threshold of convergence for the k-means algorithm, based on
        relative change in noise variance. Defaults to 1e-6.
    normalize : bool
        Whether to normalize (z-score) the data across time before running the
        k-means algorithm. Defaults to ``False``.
    random_state : int | None
        The seed for the random number generator. Defaults to ``None``, in
        which case a different seed is chosen each time this function is
        called.
    verbose : int | bool | None
        Controls the verbosity.

    Returns
    -------
    maps : ndarray, shape (n_channels, n_states)
        The topographic maps of the found unique microstates.
    assignment : ndarray, shape (n_samples,)
        For each sample, the index of the microstate to which the sample has
        been assigned.

    References
    ----------
    .. [1] Pascual-Marqui, R. D., Michel, C. M., & Lehmann, D. (1995).
           Segmentation of brain electrical activity into microstates: model
           estimation and validation. IEEE Transactions on Biomedical
           Engineering.
    """
    logger.info('Finding %d microstates, using %d random intitializations' %
                (n_states, n_inits))

    if normalize:
        data = zscore(data, axis=1)

    # Compute global field power (GFP)
    gfp = np.std(data, axis=0)
    gfp_sum_sq = np.sum(gfp ** 2)

    # Do several runs of the k-means algorithm, keep track of the best
    # segmentation.
    best_gev = 0
    best_maps = None
    best_assignment = None
    for _ in range(n_inits):
        maps, assignment = _mod_kmeans(data, n_states, n_inits, max_iter,
                                       thresh, random_state, verbose)
        map_corr = _corr_vectors(data, maps[assignment].T)

        # Compare across iterations using global explained variance (GEV) of
        # the found microstates.
        gev = sum((gfp * map_corr) ** 2) / gfp_sum_sq
        logger.info('GEV of found microstates: %f' % gev)
        if gev > best_gev:
            best_gev, best_maps, best_assignment = gev, maps, assignment

    return best_maps, best_assignment


@verbose
def _mod_kmeans(data, n_states=4, n_inits=10, max_iter=1000, thresh=1e-6,
                random_state=None, verbose=None):
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    n_channels, n_samples = data.shape

    # Cache this value for later
    data_sum_sq = np.sum(data ** 2)

    # Select random timepoints for our initial topographic maps
    init_times = random_state.choice(n_samples, size=n_states, replace=False)
    maps = data[:, init_times].T
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)  # Normalize the maps

    prev_residual = np.inf
    for iteration in range(max_iter):
        # Assign each sample to the best matching microstate
        activation = maps.dot(data)
        assignments = np.argmax(activation ** 2, axis=0)
        # assigned_activations = np.choose(assignments, all_activations)

        # Recompute the topographic maps of the microstates, based on the
        # samples that were assigned to each state.
        for state in range(n_states):
            idx = (assignments == state)
            if np.sum(idx) == 0:
                warnings.warn('Some microstates are never activated')
                maps[state] = 0
                continue
            maps[state] = data[:, idx].dot(activation[state, idx])
            maps[state] /= np.linalg.norm(maps[state])

        # Estimate residual noise
        act_sum_sq = np.sum(np.sum(maps[assignments].T * data, axis=0) ** 2)
        residual = abs(data_sum_sq - act_sum_sq)
        residual /= float(n_samples * (n_channels - 1))

        # Have we converged?
        if (prev_residual - residual) < (thresh * residual):
            logger.info('Converged at %d iterations.' % iteration)
            break

        prev_residual = residual
    else:
        warnings.warn('Modified K-means algorithm failed to converge.')

    # Compute final microstate assignments
    activation = maps.dot(data)
    assignments = np.argmax(activation ** 2, axis=0)

    return maps, assignments


def _corr_vectors(A, B, axis=0):
    """Compute pairwise correlation of multiple pairs of vectors.

    Fast way to compute correlation of multiple pairs of vectors without
    computing all pairs as would with corr(A,B). Borrowed from Oli at Stack
    overflow. Note the resulting coefficients vary slightly from the ones
    obtained from corr due differences in the order of the calculations.
    (Differences are of a magnitude of 1e-9 to 1e-17 depending of the tested
    data).

    Parameters
    ----------
    A : ndarray, shape (n, m)
        The first collection of vectors
    B : ndarray, shape (n, m)
        The second collection of vectors
    axis : int
        The axis that contains the elements of each vector. Defaults to 0.

    Returns
    -------
    corr : ndarray, shape (m,)
        For each pair of vectors, the correlation between them.
    """
    An = A - np.mean(A, axis=axis)
    Bn = B - np.mean(B, axis=axis)
    An /= np.linalg.norm(An, axis=axis)
    Bn /= np.linalg.norm(Bn, axis=axis)
    return np.sum(An * Bn, axis=axis)


def plot_assignment(assignment, times):
    """Plot a microstate segmentation.

    Parameters
    ----------
    assignment : list of int
        For each sample in time, the index of the state to which the sample has
        been assigned.
    times : list of float
        The time-stamp for each sample.
    """
    plt.figure(figsize=(6 * np.ptp(times), len(np.unique(assignment)) / 2.))
    for state in np.unique(assignment):
        idx = (assignment == state)
        plt.scatter(times[idx], state * np.ones(np.sum(idx)))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
    plt.grid(which='minor', axis='y', linestyle='solid', color='black')
    plt.ylabel('State')
    plt.xlabel('Time (s)')
    plt.tight_layout()


def plot_maps(maps, info):
    """Plot prototypical microstate maps.

    Parameters
    ----------
    maps : ndarray, shape (n_channels, n_maps)
        The prototypical microstate maps.
    info : instance of mne.io.Info
        The info structure of the dataset, containing the location of the
        sensors.
    """
    plt.figure(figsize=(2 * len(maps), 2))
    layout = mne.channels.find_layout(info)
    for i, map in enumerate(maps):
        plt.subplot(1, len(maps), i + 1)
        mne.viz.plot_topomap(map, layout.pos[:, :2])
        plt.title('%d' % i)
