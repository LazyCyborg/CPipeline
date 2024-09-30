# tms_data_loader.py

import os
import numpy as np
import mne
from mne_bids import (
    BIDSPath,
    find_matching_paths,
    get_entity_vals,
    make_report,
    print_dir_tree,
    read_raw_bids,
)
from neurone_loader import Recording

class TMSDataLoader:
    def __init__(self, data_path, BIDS=True, preload=True, montage_path=None, extensions=None, event_markers=None):
        """
        Initialize the TMSDataLoader.

        Parameters:
            data_path (str): Path to the data directory.
            BIDS (bool): Indicates whether the data is in BIDS format.
            preload (bool): Whether to preload the data into memory.
            montage_path (str): Path to montage files (if applicable).
            extensions (list): List of file extensions to consider.
            event_markers (dict): Dictionary specifying event markers.
        """
        self.data_path = data_path
        self.BIDS = BIDS
        self.preload = preload
        self.montage_path = montage_path
        self.extensions = extensions if extensions else ['.set', '.fif', '.edf', '.xdf']
        self.event_markers = event_markers if event_markers else {}
        self.subjects = self.get_subjects()

    def get_subjects(self):
        if self.BIDS:
            from mne_bids import get_entity_vals
            bids_root = self.data_path
            subjects = get_entity_vals(bids_root, 'subject')
            return subjects
        else:
            # For non-BIDS data, assume each subject has a directory under data_path
            subjects = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
            return subjects

    def load_subject_data(self, subject):
        """
        Load data for a single subject.

        Parameters:
            subject (str): Subject identifier.

        Returns:
            raw (Raw): The raw EEG data.
            events (ndarray): The events array.
            channels (list): List of channel names.
        """
        if self.BIDS:
            raw, events, channels = self.load_bids_subject_data(subject)
        else:
            raw, events, channels = self.load_non_bids_subject_data(subject)
        return raw, events, channels

    def load_bids_subject_data(self, subject):
        from mne_bids import BIDSPath, read_raw_bids

        bids_path = BIDSPath(subject=subject, root=self.data_path, datatype='eeg', task='restingstate', suffix='eeg')
        try:
            raw = read_raw_bids(bids_path=bids_path, verbose=False)
            if self.preload:
                raw.load_data()
            # Get events from annotations
            events, event_id = mne.events_from_annotations(raw)
            channels = raw.info['ch_names']
            return raw, events, channels
        except Exception as e:
            print(f"Could not read data for subject {subject}: {e}")
            return None, None, None

    def load_neurone_subject_data(self, subject):
        """
        Load data not in BIDS format for a single subject.

        Parameters:
            subject (str): Subject identifier.

        Returns:
            raw (Raw): The raw EEG data.
            events (ndarray): The events array.
            channels (list): List of channel names.
        """
        # Assuming data for each subject is under data_path/subject
        subject_path = os.path.join(self.data_path, subject)
        if not os.path.exists(subject_path):
            print(f"Data path for subject {subject} does not exist.")
            return None, None, None

        # Import the custom Recording class from neurone_loader
        from neurone_loader import Recording

        # Create a Recording instance
        recording = Recording(path=subject_path, preload=self.preload)

        # Define channel type mappings
        eeg_channels = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
            'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
            'Fz', 'Cz', 'Pz', 'Iz', 'FC1', 'FC2', 'CP1', 'CP2',
            'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'AFz', 'FCz'
        ]

        channel_type_mappings = {ch: 'eeg' for ch in eeg_channels}
        channel_type_mappings['EMG1'] = 'emg'

        # Convert to MNE Raw object
        raw_array, events_df, channels = recording.to_mne(
            substitute_zero_events_with=10,
            channel_type_mappings=channel_type_mappings
        ), recording.events, recording.channels

        # Prepare events array
        events_np = events_df.to_numpy()
        sample_indices = events_np[:, 0].astype(int)
        event_ids = events_np[:, 12].astype(int)
        events = np.column_stack((sample_indices, np.zeros_like(sample_indices), event_ids))

        return raw_array, events, channels
