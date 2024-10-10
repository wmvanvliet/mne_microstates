# Microstate analysis for use with MNE-Python

Forked from Pascual Marqui:
Pascual-Marqui, R. D., Michel, C. M., & Lehmann, D. (1995).  Segmentation of
brain electrical activity into microstates: model estimation and validation.
IEEE Transactions on Biomedical Engineering.
https://ieeexplore.ieee.org/document/391164


## Installation

Install this package using PIP:

```
pip install git+https://github.com/KishiBayes/mne_microstates
```


## Usage

```python
import mne
import mne_microstates

# Load MNE sample dataset
from mne.datasets import sample
fname = sample.data_path() / 'MEG/sample/sample_audvis_filt.fif'
raw = mne.io.read_raw_fif(fname, preload=True)

# Always use an average EEG reference when doing microstate analysis
raw.set_eeg_reference('average')

# Highpass filter the data a little bit
raw.filter(0.2, None)

# Selecting the sensor types to use in the analysis. In this example, we
# use only EEG channels
raw.pick_types(meg=False, eeg=True)

# Segment the data into 6 microstates
maps, segmentation = mne_microstates.segment(raw.get_data(), n_states=6)

# Plot the topographic maps of the found microstates
mne_microstates.plot_maps(maps, raw.info)

# Plot the segmentation of the first 500 samples
mne_microstates.plot_segmentation(segmentation[:500], raw.get_data()[:, :500], raw.times[:500])
```

## Author
Marijn van Vliet <w.m.vanvliet@gmail.com>
