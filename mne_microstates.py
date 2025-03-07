"""Functions to segment EEG into microstates.

Based on the Microsegment toolbox for EEGlab, written by Andreas Trier Poulsen [1]_.

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
import matplotlib as mpl
from matplotlib import pyplot as plt
import mne
from mne.utils import logger, verbose


__version__ = "0.4dev0"


@verbose
def segment(
    data,
    n_states=4,
    n_inits=10,
    max_iter=1000,
    thresh=1e-6,
    normalize=False,
    min_peak_dist=2,
    max_n_peaks=10000,
    return_polarity=False,
    return_best_gev=False,
    weights=None,
    random_state=None,
    verbose=None,
):
    """Segment a continuous signal into microstates.

    Peaks in the global field power (GFP) are used to find microstates, using a
    modified K-means algorithm. Several runs of the modified K-means algorithm
    are performed, using different random initializations. The run that
    resulted in the best segmentation, as measured by global explained variance
    (GEV), is used.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
        The data to find the microstates in.
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
    min_peak_dist : int
        Minimum distance (in samples) between peaks in the GFP. Defaults to 2.
    max_n_peaks : int | None
        Maximum number of GFP peaks to use in the k-means algorithm. Chosen
        randomly. Set to ``None`` to use all peaks. Defaults to 10000.
    return_polarity : bool
        Whether to return the polarity of the activation.
        Defaults to ``False``.
    return_best_gev : bool
        Option to return best golbally explained variance. Defaults to False.
    weights : array-like | None
        This is an optional array which can be used to weight each sample during the
        adapted k-means algorithm. It should have the same length as the number of
        samples in the data. Defaults to ``None``.
    random_state : int | numpy.random.RandomState | None
        The seed or ``RandomState`` for the random number generator. Defaults
        to ``None``, in which case a different seed is chosen each time this
        function is called.
    verbose : int | bool | None
        Controls the verbosity.

    Returns
    -------
    maps : ndarray, shape (n_channels, n_states)
        The topographic maps of the found unique microstates.
    segmentation : ndarray, shape (n_samples,)
        For each sample, the index of the microstate to which the sample has
        been assigned.
    polarity : ndarray, shape (n_samples,)
        For each sample, the polarity (+1 or -1) of the activation on the
        currently activate map.
    gev : float
        The best global explained variance.

    References
    ----------
    .. [1] Pascual-Marqui, R. D., Michel, C. M., & Lehmann, D. (1995).
           Segmentation of brain electrical activity into microstates: model
           estimation and validation. IEEE Transactions on Biomedical
           Engineering.
    """
    logger.info(
        "Finding %d microstates, using %d random intitializations" % (n_states, n_inits)
    )

    if weights is not None:
        if not isinstance(weights, np.ndarray):
            weight_array = np.array(weights)
        if len(weights) != data.shape[1]:
            raise ValueError(
                "The `weights` array must have the same length as the number of "
                "samples in the data."
            )
        weighted_analysis = True
    else:
        weighted_analysis = False

    if normalize:
        data = zscore(data, axis=1)

    # Find peaks in the global field power (GFP)
    gfp = np.std(data, axis=0)
    peaks, _ = find_peaks(gfp, distance=min_peak_dist)
    n_peaks = len(peaks)

    # Limit the number of peaks by randomly selecting them
    if max_n_peaks is not None:
        max_n_peaks = min(n_peaks, max_n_peaks)
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        chosen_peaks = random_state.choice(n_peaks, size=max_n_peaks, replace=False)
        peaks = peaks[chosen_peaks]

    # Cache this value for later
    gfp_sum_sq = np.sum(gfp**2)

    # Do several runs of the k-means algorithm, keep track of the best
    # segmentation.
    best_gev = 0
    best_maps = None
    best_segmentation = None
    best_polarity = None
    for _ in range(n_inits):
        maps = _mod_kmeans(
            data[:, peaks],
            n_states,
            n_inits,
            max_iter,
            thresh,
            random_state,
            verbose,
            weighted_analysis,
            weight_array,
        )
        activation = maps.dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)
        map_corr = _corr_vectors(data, maps[segmentation].T)
        # assigned_activations = np.choose(segmentations, activation)

        # Compare across iterations using global explained variance (GEV) of
        # the found microstates.
        gev = sum((gfp * map_corr) ** 2) / gfp_sum_sq
        logger.info("GEV of found microstates: %f" % gev)
        if gev > best_gev:
            best_gev, best_maps, best_segmentation = gev, maps, segmentation
            best_polarity = np.sign(np.choose(segmentation, activation))

    output = [best_maps, best_segmentation]
    if return_polarity:
        output.append(best_polarity)
    if return_best_gev:
        output.append(best_gev)
    return tuple(output)


@verbose
def _mod_kmeans(
    data,
    n_states=4,
    n_inits=10,
    max_iter=1000,
    thresh=1e-6,
    random_state=None,
    verbose=None,
    weighted_analysis=False,
    weight_array=None,
):
    """Performs the modified K-means clustering algorithm.

    See :func:`segment` for the meaning of the parameters and return
    values.
    """
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    n_channels, n_samples = data.shape

    # Cache this value for later
    if weighted_analysis:
        data_sum_sq = np.sum(data**2 * weight_array)
    else:
        data_sum_sq = np.sum(data**2)

    # Select random timepoints for our initial topographic maps
    init_times = random_state.choice(n_samples, size=n_states, replace=False)
    maps = data[:, init_times].T
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)  # Normalize the maps

    prev_residual = np.inf
    for iteration in range(max_iter):
        # Assign each sample to the best matching microstate
        activation = maps.dot(data)
        segmentation = np.argmax(np.abs(activation), axis=0)

        # Recompute the topographic maps of the microstates, based on the
        # samples that were assigned to each state.
        for state in range(n_states):
            idx = segmentation == state
            if np.sum(idx) == 0:
                warnings.warn("Some microstates are never activated")
                maps[state] = 0
                continue
            maps[state] = data[:, idx].dot(activation[state, idx])
            maps[state] /= np.linalg.norm(maps[state])

        # Estimate residual noise
        if weighted_analysis:
            act_sum_sq = np.sum(
                np.sum(maps[segmentation].T * data, axis=0) ** 2 * weight_array
            )
        else:
            act_sum_sq = np.sum(np.sum(maps[segmentation].T * data, axis=0) ** 2)
        residual = abs(data_sum_sq - act_sum_sq)
        residual /= float(n_samples * (n_channels - 1))

        # Have we converged?
        if (prev_residual - residual) < (thresh * residual):
            logger.info("Converged at %d iterations." % iteration)
            break

        prev_residual = residual
    else:
        warnings.warn("Modified K-means algorithm failed to converge.")

    return maps


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


def plot_segmentation(segmentation, data, times, polarity=None, show=True):
    """Plot a microstate segmentation.

    Parameters
    ----------
    segmentation : list of int
        For each sample in time, the index of the state to which the sample has
        been assigned.
    data : ndarray, shape (n_channels, n_samples)
        The data on which the microstates were computed.
    times : list of float
        The time-stamp for each sample.
    polarity : list of int | None
        For each sample in time, the polarity (+1 or -1) of the activation.
    show : bool
        Show figure if ``True``.
    """
    gfp = np.std(data, axis=0)
    if polarity is not None:
        gfp *= polarity

    n_states = len(np.unique(segmentation))
    plt.figure(figsize=(6 * np.ptp(times), 2))
    cmap = plt.cm.get_cmap("plasma", n_states)
    plt.plot(times, gfp, color="black", linewidth=1)
    for state, color in zip(range(n_states), cmap.colors):
        plt.fill_between(times, gfp, color=color, where=(segmentation == state))
    norm = mpl.colors.Normalize(vmin=0, vmax=n_states)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm)
    plt.xlabel("Time (s)")
    plt.title("Segmentation into %d microstates" % n_states)
    plt.autoscale(tight=True)
    plt.tight_layout()
    if show:
        plt.show()


def plot_maps(maps, info, show=True):
    """Plot prototypical microstate maps.

    Parameters
    ----------
    maps : ndarray, shape (n_maps, n_channels)
        The prototypical microstate maps.
    info : instance of mne.io.Info
        The info structure of the dataset, containing the location of the
        sensors.
    show : bool
        Show figure if ``True``.
    """
    assert len(maps) != 1, "Only one map found, cannot plot"
    fig, axes = plt.subplots(1, len(maps), figsize=(2 * len(maps), 2))
    for i, (ax, map) in enumerate(zip(axes, maps)):
        mne.viz.plot_topomap(map, info, axes=ax, show=False)
        ax.set_title("Microstate %d" % (i + 1))
    plt.tight_layout()
    if show:
        plt.show()
