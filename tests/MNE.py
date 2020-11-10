# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

# %%

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.time_frequency import tfr_morlet
from mne.stats import permutation_cluster_1samp_test, spatio_temporal_cluster_1samp_test
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
tmin, tmax, event_id = -0.3, 0.6, 1

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname)
events = mne.find_events(raw, stim_channel='STI 014')

include = []
raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more

# picks MEG gradiometers
picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                       stim=False, include=include, exclude='bads')

# Load condition 1
event_id = 1
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True,
                    reject=dict(grad=4000e-13, eog=150e-6))
# just use right temporal sensors for speed
epochs.pick_channels(mne.read_selection('Right-temporal'))
evoked = epochs.average()

# Factor to down-sample the temporal dimension of the TFR computed by
# tfr_morlet. Decimation occurs after frequency decomposition and can
# be used to reduce memory usage (and possibly computational time of downstream
# operations such as nonparametric statistics) if you don't need high
# spectrotemporal resolution.
decim = 5
freqs = np.arange(8, 40, 2)  # define frequencies of interest
sfreq = raw.info['sfreq']  # sampling in Hz
tfr_epochs = tfr_morlet(epochs, freqs, n_cycles=4., decim=decim,
                        average=False, return_itc=False, n_jobs=1)

# Baseline power
tfr_epochs.apply_baseline(mode='logratio', baseline=(-.100, 0))

# Crop in time to keep only what is between 0 and 400 ms
evoked.crop(-0.1, 0.4)
tfr_epochs.crop(-0.1, 0.4)

epochs_power = tfr_epochs.data

sensor_adjacency, ch_names = mne.channels.find_ch_connectivity(
    tfr_epochs.info, 'grad')
# Subselect the channels we are actually using
use_idx = [ch_names.index(ch_name.replace(' ', ''))
           for ch_name in tfr_epochs.ch_names]
sensor_adjacency = sensor_adjacency[use_idx][:, use_idx]
assert sensor_adjacency.shape == \
    (len(tfr_epochs.ch_names), len(tfr_epochs.ch_names))
assert epochs_power.data.shape == (
    len(epochs), len(tfr_epochs.ch_names),
    len(tfr_epochs.freqs), len(tfr_epochs.times))
adjacency = mne.stats.combine_adjacency(
    sensor_adjacency, len(tfr_epochs.freqs), len(tfr_epochs.times))

# our adjacency is square with each dim matching the data size
assert adjacency.shape[0] == adjacency.shape[1] == \
    len(tfr_epochs.ch_names) * len(tfr_epochs.freqs) * len(tfr_epochs.times)

# %%

threshold = 3.
n_permutations = 50  # Warning: 50 is way too small for real-world analysis.
# T_obs, clusters, cluster_p_values, H0 = \
#     permutation_cluster_1samp_test(epochs_power, n_permutations=n_permutations,
#                                    threshold=threshold, tail=0,
#                                    connectivity=None,
#                                    out_type='mask', verbose=True)

T_obs, clusters, cluster_p_values, H0 = \
    spatio_temporal_cluster_1samp_test(epochs_power, n_permutations=n_permutations,
                                   threshold=threshold, tail=0,
                                   connectivity=None,
                                   out_type='mask', verbose=True)

# %%
