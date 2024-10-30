import os
import numpy as np
import mne
import pyxdf
from mne_bids import (
    BIDSPath,
    find_matching_paths,
    get_entity_vals,
    make_report,
    print_dir_tree,
    read_raw_bids,
)

from neurone_loader import Recording

class EEGDataLoader:
    def __init__(self, data_path, BIDS=True, preload=True, montage_path=None, extensions=None, event_markers=None):
        """
        Initialize the EEGDataLoader.

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
            bids_root = self.data_path
            subjects = get_entity_vals(bids_root, 'subject')
            return subjects
        else:
            # For non-BIDS data, include files with the specified extensions
            subjects = {}
            for f in os.listdir(self.data_path):
                full_path = os.path.join(self.data_path, f)
                ext = os.path.splitext(f)[1].lower()
                if os.path.isfile(full_path) and ext in self.extensions:
                    subject_id = os.path.splitext(f)[0]
                    subjects[subject_id] = f  # Map subject ID to filename
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
            return self.load_bids_subject_data(subject)
        else:
            filename = self.subjects.get(subject, None)
            if filename is None:
                print(f"No filename found for subject {subject}.")
                return None, None, None
            return self.load_non_bids_subject_data(filename)


    def load_bids_subject_data(self, subject):
        from mne_bids import BIDSPath, read_raw_bids

        # Create a BIDSPath with the required entities
        bids_path = BIDSPath(
            subject=subject,
            root=self.data_path,
            datatype='eeg',
            task='restingstate',  # Adjust task if needed
            suffix='eeg',
            extension=None  # We'll handle extensions separately
        )
        try:
            # Iterate over possible extensions to find the file
            for ext in self.extensions:
                bids_path_ext = bids_path.copy().update(extension=ext)
                if bids_path_ext.fpath.exists():
                    data_file = bids_path_ext.fpath
                    file_extension = data_file.suffix.lower()
                    if file_extension == '.xdf':
                        # Load XDF file using custom method
                        raw, events, channels = self.load_xdf_file(data_file)
                        return raw, events, channels
                    elif file_extension == '.set':
                        # Load EEGLAB .set file
                        raw = read_raw_bids(bids_path=bids_path_ext, verbose=False)
                        if self.preload:
                            raw.load_data()
                        events, event_id = mne.events_from_annotations(raw)
                        channels = raw.info['ch_names']
                        return raw, events, channels
                    else:
                        # Use read_raw_bids for other supported formats
                        raw = read_raw_bids(bids_path=bids_path_ext, verbose=False)
                        if self.preload:
                            raw.load_data()
                        events, event_id = mne.events_from_annotations(raw)
                        channels = raw.info['ch_names']
                        return raw, events, channels
            print(f"No data files found for subject {subject} with extensions {self.extensions}")
            return None, None, None
        except Exception as e:
            print(f"Could not read data for subject {subject}: {e}")
            return None, None, None

    def load_non_bids_subject_data(self, filename):
        """
        Load data not in BIDS format for a single subject.

        Parameters:
            filename (str): Filename of the data file.

        Returns:
            raw (Raw): The raw EEG data.
            events (ndarray): The events array.
            channels (list): List of channel names.
        """
        file_path = os.path.join(self.data_path, filename)
        if not os.path.exists(file_path):
            print(f"Data file {filename} does not exist.")
            return None, None, None

        if os.path.isfile(file_path):
            # If file_path is a file, load it directly
            return self.load_data_file(file_path)
        else:
            # If file_path is a directory, check for NeurOne data
            if self.is_neurone_data(file_path):
                return self.load_neurone_data(file_path)
            else:
                print(f"Data path {file_path} is not a file or a recognized data directory.")
                return None, None, None


    def load_data_file(self, data_file):
        """
        Load data from a file based on its extension.

        Parameters:
            data_file (str): Path to the data file.

        Returns:
            raw (Raw): The raw EEG data.
            events (ndarray): The events array (if any).
            channels (list): List of channel names.
        """
        file_extension = os.path.splitext(data_file)[1].lower()

        if file_extension == '.xdf':
            raw, events, channels = self.load_xdf_file(data_file)
            return raw, events, channels
        elif file_extension == '.set':
            # Load EEGLAB .set file
            raw = mne.io.read_raw_eeglab(data_file, preload=self.preload)
        elif file_extension == '.fif':
            raw = mne.io.read_raw_fif(data_file, preload=self.preload)
        elif file_extension == '.edf':
            raw = mne.io.read_raw_edf(data_file, preload=self.preload)
        else:
            print(f"Unsupported file extension {file_extension} for file {data_file}")
            return None, None, None

        events, event_id = mne.events_from_annotations(raw)
        channels = raw.info['ch_names']
        return raw, events, channels

    def load_xdf_file(self, xdf_file, sfreq=5000):
        """
        Load an XDF file and extract the raw data and events.

        Parameters:
            xdf_file (str): Path to the XDF file.

        Returns:
            raw (Raw): The raw EEG data.
            events (ndarray): The events array.
            channels (list): List of channel names.
        """
        import pyxdf

        # Load the XDF file
        streams, header = pyxdf.load_xdf(xdf_file)

        # Find the EEG stream
        eeg_stream = None
        for stream in streams:
            if stream['info']['type'][0].lower() == 'eeg':
                eeg_stream = stream
                break

        if eeg_stream is None:
            raise ValueError("No EEG stream found in the XDF file.")

        # Extract data and transpose it
        data = eeg_stream['time_series'].T
        data *= 1e-6  # Scaling to volts

        # Sampling frequency
        sfreq = float(eeg_stream['info']['nominal_srate'][0])

        # Channel names
        try:
            ch_dict = eeg_stream['info']['desc'][0]['channels'][0]['channel']
            ch_names = [ch['label'][0] for ch in ch_dict]
        except (TypeError, KeyError, IndexError):
            ch_names = [f'EEG {i+1}' for i in range(data.shape[0])]

        # Channel types
        ch_types = ['eeg'] * len(ch_names)

        # Create MNE Info object
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        # Create Raw object
        raw = mne.io.RawArray(data, info)

        # Handle events (Markers)
        events = []
        event_id = {}
        # Extract marker streams if present
        for stream in streams:
            if stream['info']['type'][0].lower() == 'markers':
                time_stamps = stream['time_stamps']
                markers = stream['time_series']
                for idx, marker in enumerate(markers):
                    # Find the sample index corresponding to the time_stamp
                    sample_idx = int((time_stamps[idx] - eeg_stream['time_stamps'][0]) * sfreq)
                    # Convert marker to string
                    marker_value = marker[0]
                    if isinstance(marker_value, bytes):
                        marker_value = marker_value.decode('utf-8')
                    # If the marker is not in event_id, assign a new id
                    if marker_value not in event_id:
                        event_id[marker_value] = len(event_id) + 1
                    events.append([sample_idx, 0, event_id[marker_value]])

        events = np.array(events) if events else None
        channels = raw.info['ch_names']
        return raw, events, channels

    def is_neurone_data(self, path):
        """
        Check if the given path contains NeurOne data.
        """
        # NeurOne data typically contains 'Protocol.xml' files
        return any('Protocol.xml' in files for _, _, files in os.walk(path))

    def load_neurone_data(self, data_path):
        """
        Load NeurOne data using neurone_loader.
        """
        try:
            # Create a Recording instance
            recording = Recording(path=data_path, preload=self.preload)

            # Define channel type mappings (adjust as needed)
            eeg_channels = [
                'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
                'Fz', 'Cz', 'Pz', 'Iz', 'FC1', 'FC2', 'CP1', 'CP2',
                'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'AFz', 'FCz'
            ]
            channel_type_mappings = {ch: 'eeg' for ch in eeg_channels}
            channel_type_mappings['EMG1'] = 'emg'

            # Convert to MNE Raw object
            raw = recording.to_mne(
                substitute_zero_events_with=10,
                channel_type_mappings=channel_type_mappings
            )

            events_df = recording.events
            channels = recording.channels

            # Prepare events array
            if events_df is not None and not events_df.empty:
                sample_indices = events_df['StartSampleIndex'].to_numpy(dtype=int)
                event_ids = events_df['Code'].to_numpy(dtype=int)
                events = np.column_stack((sample_indices, np.zeros_like(sample_indices), event_ids))
            else:
                events = None

            return raw, events, channels
        except Exception as e:
            print(f"Failed to load NeurOne data from {data_path}: {e}")
            return None, None, None
