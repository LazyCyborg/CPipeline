# EEGPipeline
My code for preprocessing and extracting features from an EEG dataset in BIDS format.

## Packages used for preprocessing

### MNE-Python
The code relies mostly on the automatic functions in MNE-Python 

https://mne.tools/stable/index.html

Alexandre Gramfort, Martin Luessi, Eric Larson, Denis A. Engemann, Daniel Strohmeier, Christian Brodbeck, Roman Goj, Mainak Jas, Teon Brooks, Lauri Parkkonen, and Matti S. Hämäläinen. MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience, 7(267):1–13, 2013. doi:10.3389/fnins.2013.00267.

### MNE-FASTER
The actual automatisation of the preprocessing uses MNE-FASTER

https://github.com/wmvanvliet/mne-faster?tab=readme-ov-file

Nolan H, Whelan R, Reilly RB. FASTER: Fully Automated Statistical Thresholding for EEG artifact Rejection. J Neurosci Methods. 2010 Sep 30;192(1):152-62. doi: 10.1016/j.jneumeth.2010.07.015. Epub 2010 Jul 21. PMID: 20654646.

### MNE-ICALabel
Additionally MNE-ICALabel from MNE tools is used to automatically label ICA components

https://mne.tools/mne-icalabel/dev/index.html


## Packages used for feature extraction

### Pycrostates
Is used for extraction of microstate sequences in the data

https://pycrostates.readthedocs.io/en/latest/

https://joss.theoj.org/papers/10.21105/joss.04564 

### Antropy
Is used for extraction of entropy based features in EEG data

https://github.com/raphaelvallat/antropy

Currently the only sample dataset that the code is tested on:

https://openneuro.org/datasets/ds005305/versions/1.0.1

Chenot Quentin and Hamery Caroline and Truninger Moritz and De Boissezon Xavier and Langer Nicolas and Scannella Sébastien (2024). EEG Resting-state Microstates Correlates of Executive Functions. OpenNeuro. [Dataset] doi: doi:10.18112/openneuro.ds005305.v1.0.1

#### Install

```
!pip install git+https://github.com/LazyCyborg/EEGPipeline.git
```
#### Example usage 
```
from EEGPipeline.auto_preproc import Preprocessor
from EEGPipeline.extract_features import TS_Feature

preproc = Preprocessor(
    bids_root='/ds005305-download',
    output_dir='Preproc_eeg',
    montage_path='/ds005305-download',
    event_markers={'EO': 1, 'EC': 2},
    crop_events=True,
    preprocessing_steps=[
        'set_eeg_reference', 'filter', 'interpolate_bad_channels',
        'find_bad_channels', 'find_bad_channels_in_epochs', 'find_bad_epochs', 'ica',
        'baseline_correction', 'filter_epochs', 'set_average_reference'
    ]
)

preproc.run_preprocessing(reference_channel='B16')

```
