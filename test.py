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
#raw.filter(0.2, 20)
events = mne.find_events(raw)
picks = mne.pick_types(raw.info, meg=False, eeg=True)
epochs = mne.Epochs(raw, events, event_id=[1, 2], tmin=-0.1, tmax=0.8,
                    picks=picks, preload=True)
epochs.set_eeg_reference('average')

maps, assignment = microstates.find_microstates(epochs.get_data()[0], n_states=6)

microstates.plot_maps(maps, epochs.info)
microstates.plot_assignment(assignment[:epochs.get_data().shape[2]], epochs.times)
