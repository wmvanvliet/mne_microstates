import mne
import microstates

from mne.datasets import sample
fname = sample.data_path() + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(fname, preload=True)
events = mne.find_events(raw)
raw.info['bads'] = ['MEG 2443', 'EEG 053']
raw.set_eeg_reference('average')
raw.pick_types(meg='mag', eeg=True, eog=True, ecg=True)
raw.filter(1, 40)

# Clean EOG with ICA
ica = mne.preprocessing.ICA(0.99).fit(raw)
bads_eog, _ = ica.find_bads_eog(raw)
bads_ecg, _ = ica.find_bads_ecg(raw)
ica.exclude = bads_eog[:2] + bads_ecg[:2]
raw = ica.apply(raw)

# Select sensor type
# raw.pick_types(meg=False, eeg=True)
raw.pick_types(meg='mag', eeg=False)

# Segment the data in 6 microstates
maps, segmentation = microstates.segment(raw.get_data(), n_states=5)

# Plot the topographic maps of the microstates and the segmentation
microstates.plot_maps(maps, raw.info)
microstates.plot_segmentation(segmentation[:500], raw.get_data()[:, :500],
                              raw.times[:500])
