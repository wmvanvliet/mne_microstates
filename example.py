import mne
import microstates

from mne.datasets import sample
fname = sample.data_path() + '/MEG/sample/sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(fname, preload=True)
raw.info['bads'] = ['MEG 2443', 'EEG 053']
raw.pick_types(meg='mag', eeg=True, eog=True, ecg=True)

# Microstate analysis needs average reference
raw.set_eeg_reference('average')

# Filter the data. Make sure slow drifts are eliminated.
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

# Segment the data into 5 microstates
maps, segmentation, polarity = microstates.segment(raw.get_data(), n_states=5,
                                                   random_state=0,
                                                   return_polarity=True)

# Plot the topographic maps of the microstates and part of the segmentation
microstates.plot_maps(maps, raw.info)
microstates.plot_segmentation(segmentation[:500], raw.get_data()[:, :500],
                              raw.times[:500], polarity=polarity)
