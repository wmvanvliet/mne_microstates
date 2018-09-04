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
from matplotlib import pyplot as plt
import mne
from mne.utils import logger, verbose


@verbose
def find_microstates(data, n_states=4, n_inits=10, max_iter=1000, thresh=1e-6,
                     normalize=False, random_state=None, verbose=None):
    logger.info('Finding %d microstates, using %d random intitializations' %
                (n_states, n_inits))

    # Compute global field power (GFP)
    gfp = np.std(data, axis=0)
    gfp_sum_sq = np.sum(gfp ** 2)

    best_gev = 0
    best_maps = None
    best_assignment = None
    for _ in range(n_inits):
        maps, assignment = _mod_kmeans(data, n_states, n_inits, max_iter,
                                       thresh, random_state, verbose)
        map_corr = columncorr(data, maps[assignment].T)
        
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
    """Segment a signal into microstates using the modified K-means algorithm.

    Several runs of the algorithms are performed, using different random
    initializations. The run that resulted in the best segmentation, as
    measured by global explained variance (GEV), is used.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
        The data to find the clusters in
    n_states : int
        The number of unique microstates to find. Defaults to 4.
    n_inits : int
        The number of random initializations to use. The best fitting
        segmentation across all initializations is used. Defaults to 10.
    max_iter : int
        The maximum number of iterations to perform. Defaults to 1000.
    thresh : float
        The threshold of convergence based on relative change in noise
        variance. Defaults to 1e-6.
    smoothing_width : int
        The number of samples on each side of the current sample to use to
        temporally smooth the data. Defaults to 0.
    smoothing_weight : float
        TODO
    random_state : int | None
        The seed for the random number generator. Defaults to ``None``, in
        which case a different seed is chosen each time this function is
        called.

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
    n_channels, n_samples = data.shape
    rnd = np.random.RandomState(random_state)

    # Cache this value for later
    data_sum_sq = np.sum(data ** 2)

    # Select random timepoints for our initial topographic maps
    maps = data[:, rnd.choice(n_samples, size=n_states, replace=False)].T
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


def plot_assignment(assignment, times):
    plt.figure(figsize=(6 * np.ptp(times), len(np.unique(assignment)) / 4))
    for state in np.unique(assignment):
        idx = (assignment == state)
        plt.scatter(times[idx], state * np.ones(np.sum(idx)))
    plt.tight_layout()


def plot_maps(maps, info):
    plt.figure(figsize=(len(maps), 1))
    l = mne.channels.make_eeg_layout(info)
    for i, map in enumerate(maps, 1):
        plt.subplot(1, len(maps), i)
        mne.viz.plot_topomap(map, l.pos[:, :2])


def columncorr(A, B):
    """Fast way to compute correlation of multiple pairs of vectors without
    computing all pairs as would with corr(A,B). Borrowed from Oli at Stack
    overflow. Note the resulting coefficients vary slightly from the ones
    obtained from corr due differences in the order of the calculations.
    (Differences are of a magnitude of 1e-9 to 1e-17 depending of the tested
    data)."""
    An = A - np.mean(A, axis=0)
    Bn = B - np.mean(B, axis=0)
    An /= np.sqrt(np.sum(An ** 2, axis=0))
    Bn /= np.sqrt(np.sum(Bn ** 2, axis=0))
    return np.sum(An * Bn, axis=0)
