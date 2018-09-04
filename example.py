import mne
import microstates

from mne.datasets import sample
fname = sample.data_path() + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(fname, preload=True)
raw.set_eeg_reference('average')
raw.filter(0.2, 40)

# Select sensor type
raw.pick_types(meg=False, eeg=True)
# raw.pick_types(meg='mag', eeg=False)

# Segment the data in 6 microstates
maps, assignment = microstates.microstates_raw(raw, n_states=6)

# Plot the topographic maps of the microstates and the segmentation
microstates.plot_maps(maps, raw.info)
microstates.plot_assignment(assignment[:500], raw.times[:500])
