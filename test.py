import numpy as np
import mne
import microstates

states = np.array([
    [1, 0, 1, 0],
    [1, 1, 0, 0],
]).T.astype(float)

assignments = np.array([0, 1, 0, 1, 1, 0, 1, 0])
data = states[:, assignments] + 0.1 * np.random.randn(4, 8)

from mne.datasets import sample
fname = sample.data_path() + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(fname, preload=True)
events = mne.find_events(raw)
raw.info['bads'] = ['MEG 2443', 'EEG 053']
raw.pick_types(meg=False, eeg=True)
# raw.pick_types(meg='mag', eeg=False)
raw.set_eeg_reference('average')
raw.filter(0.2, 40)

maps_raw, assignment_raw = microstates.microstates_raw(raw, n_states=6, n_inits=50, verbose=True)

# Sort microstates in the order of first appearance
first_occurance = [np.argwhere(assignment_raw == state)[0, 0] for state in range(len(maps_raw))]
order = np.argsort(first_occurance)
maps_raw = maps_raw[order]
assignment_raw = order[assignment_raw]

microstates.plot_maps(maps_raw, raw.info)
microstates.plot_assignment(assignment_raw[:1000], raw.times[:1000])

# epochs = mne.Epochs(raw, events, event_id=[1, 2], tmin=-0.1, tmax=0.8, preload=True)
# evoked = epochs.average()
# 
# maps, assignment = microstates.microstates_array(evoked.data, n_states=6, verbose=True)
# microstates.plot_maps(maps, epochs.info)
# microstates.plot_assignment(assignment, epochs.times)
