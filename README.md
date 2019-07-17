# Microstate analysis for use with MNE-Python

A small module that works with MNE-Python to perform microstate analysis in EEG
and MEG data.

## Usage

    import mne
    import microstates

    # Load MNE sample dataset
    from mne.datasets import sample
    fname = sample.data_path() + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
    raw = mne.io.read_raw_fif(fname, preload=True)

    # Always use an average EEG reference when doing microstate analysis
    raw.set_eeg_reference('average')

    # Highpass filter the data a little bit
    raw.filter(0.2, None)

    # Selecting the sensor types to use in the analysis. In this example, we
    # use only EEG channels
    raw.pick_types(meg=False, eeg=True)

    # Segment the data into 6 microstates
    maps, assignment = microstates.segment(raw.get_data(), n_states=6)

    # Plot the topographic maps of the found microstates
    microstates.plot_maps(maps, raw.info)

    # Plot the segmentation of the first 500 samples
    microstates.plot_segmentation(assignment[:500], raw.times[:500])

## Author
Marijn van Vliet <w.m.vanvliet@gmail.com>
