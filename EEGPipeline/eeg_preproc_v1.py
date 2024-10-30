# eeg_preprocessor.py

import os
import numpy as np
import mne
from mne.preprocessing import ICA
from mne_bids import BIDSPath
from mne_faster import find_bad_channels, find_bad_epochs, find_bad_channels_in_epochs
from mne_icalabel import label_components

from CPipeline import EEGDataLoader  # Adjust the import path as needed

class EEGPreprocessor:
    def __init__(
        self, data_path, BIDS=True, preload=True, montage_path=None,
        extensions=None, event_markers=None, crop_events=True, preprocessing_steps=None
    ):
        """
        Initialize the Preprocessor.

        Parameters:
            data_path (str): The root directory of the data.
            BIDS (bool): Indicates whether the data is in BIDS format.
            preload (bool): Whether to preload the data into memory.
            montage_path (str): Path to montage files.
            extensions (list): List of file extensions to consider.
            event_markers (dict): Dictionary specifying event markers.
            crop_events (bool): Whether to crop data based on events.
            preprocessing_steps (list): List of preprocessing steps to perform.
                Options include: 'set_eeg_reference', 'filter', 'interpolate_bad_channels',
                'find_bad_channels', 'find_bad_epochs', 'ica', 'baseline_correction',
                'filter_epochs', 'set_average_reference'.
        """
        self.data_path = data_path
        self.BIDS = BIDS
        self.preload = preload
        self.montage_path = montage_path
        self.event_markers = event_markers if event_markers else {}
        self.crop_events = crop_events
        self.preprocessing_steps = preprocessing_steps if preprocessing_steps else []
        self.extensions = extensions if extensions else ['.set', '.fif', '.edf', '.xdf']

        # Initialize the EEGDataLoader
        self.data_loader = EEGDataLoader(
            data_path=self.data_path,
            BIDS=self.BIDS,
            preload=self.preload,
            montage_path=self.montage_path,
            extensions=self.extensions,
            event_markers=self.event_markers
        )

        # Get subjects from data loader
        self.subjects = self.data_loader.get_subjects()

    def load_subject_raw(self, subject):
        """
        Load raw EEG data for a subject.

        Parameters:
            subject (str): Subject identifier.

        Returns:
            raw (Raw): The raw EEG data.
            events (ndarray): The events array.
        """
        print(f"Loading data for subject: {subject}")
        raw, events, channels = self.data_loader.load_subject_data(subject)
        if raw is not None:
            raw.load_data()
        return raw, events

    def crop_subject_data(self, raw, events):
        """
        Crop the raw data based on event markers.

        Parameters:
            raw (Raw): The raw EEG data.
            events (ndarray): The events array.

        Returns:
            raw_cropped (Raw): The cropped raw data.
        """
        if not self.crop_events or not self.event_markers:
            print("Skipping data cropping.")
            return raw

        print('Cropping data based on event markers')

        if events is None or not events.size:
            print("No events found. Skipping cropping.")
            return raw

        # Create a list of start and stop times based on event markers
        start_marker = self.event_markers.get('start_marker')
        stop_marker = self.event_markers.get('stop_marker')
        if start_marker is None or stop_marker is None:
            print("Start or stop marker not defined in event_markers.")
            return raw

        event_times = {start_marker: [], stop_marker: []}

        for event in events:
            event_id = event[2]
            if event_id in event_times:
                event_times[event_id].append(event[0] / raw.info['sfreq'])

        # Assume events are paired (e.g., start and stop markers)
        cropped_raws = []
        num_epochs = min(len(event_times[start_marker]), len(event_times[stop_marker]))
        for i in range(num_epochs):
            tmin = event_times[start_marker][i]
            tmax = event_times[stop_marker][i]
            raw_cropped = raw.copy().crop(tmin=tmin, tmax=tmax)
            cropped_raws.append(raw_cropped)

        if not cropped_raws:
            print("No cropped data segments found.")
            return raw

        raw_cropped = mne.concatenate_raws(cropped_raws, preload=True)
        return raw_cropped

    def set_montage(self, raw, subject):
        """
        Set the montage for the raw data.

        Parameters:
            raw (Raw): The raw EEG data.
            subject (str): Subject identifier.
        """
        if not self.montage_path:
            print("No montage path provided. Skipping montage setting.")
            return

        montage_fname = os.path.join(
            self.montage_path, f"sub-{subject}", "eeg",
            f"sub-{subject}_task-restingstate_electrodes.tsv"
        )
        if os.path.isfile(montage_fname):
            montage = mne.channels.read_custom_montage(fname=montage_fname)
            # Adjust positions if necessary
            positions = montage.get_positions()['ch_pos']
            for ch_name, pos in positions.items():
                x, y, z = pos
                positions[ch_name] = np.array([y, x, z]) / 1000.0  # Swap x and y, convert mm to m
            montage = mne.channels.make_dig_montage(ch_pos=positions, coord_frame='head')
            raw.set_montage(montage, on_missing='raise', verbose=False)
            # Verify the head radius
            radius, _, _ = mne.bem.fit_sphere_to_headshape(raw.info, units='m')
            print(f"Estimated head radius for subject {subject}: {radius * 100:.2f} cm")
        else:
            print(f"Montage file not found for subject {subject}.")

    def preprocess_subject_epochs(self, raw, subject, reference, events=None):
        """
        Preprocess epochs for a subject.

        Parameters:
            raw (Raw): The raw EEG data.
            subject (str): Subject identifier.
            reference (str): Name of the reference channel, e.g., 'CZ'.
            events (ndarray): The events array.

        Returns:
            epochs (Epochs): The preprocessed epochs.
        """
        print(f"Preprocessing data for subject {subject}")

        # Apply EEG reference and filtering
        if 'set_eeg_reference' in self.preprocessing_steps:
            raw.set_eeg_reference(ref_channels=[reference])
        if 'filter' in self.preprocessing_steps:
            raw.filter(l_freq=1, h_freq=100, verbose=False)

        # Create epochs
        try:
            if events is not None:
                # Create event-based epochs
                event_id = {str(e): e for e in np.unique(events[:, 2])}
                epochs = mne.Epochs(raw, events, event_id=event_id, preload=True)
            else:
                # Create fixed-length epochs
                epochs = mne.make_fixed_length_epochs(
                    raw, duration=2.0, preload=True,
                    reject_by_annotation=True, proj=True, overlap=0.0, verbose=False
                )
        except Exception as error:
            print(f"An exception occurred during epoching for subject {subject}: {error}.")
            return None

        # Apply additional preprocessing steps
        epochs = self.apply_preprocessing_steps(epochs, subject)

        return epochs

    def apply_preprocessing_steps(self, epochs, subject):
        """
        Apply preprocessing steps to the epochs.

        Parameters:
            epochs (Epochs): The epochs to preprocess.
            subject (str): Subject identifier.

        Returns:
            epochs (Epochs): The preprocessed epochs.
        """
        if 'interpolate_bad_channels' in self.preprocessing_steps:
            # Interpolate zero variance channels
            data = epochs.get_data()
            zero_variance_channels = np.where(data.var(axis=2).mean(axis=0) == 0)[0]
            if zero_variance_channels.size > 0:
                channels_to_interpolate = [epochs.ch_names[idx] for idx in zero_variance_channels]
                print(f"Interpolating channels with zero variance: {channels_to_interpolate}")
                epochs.info['bads'] = list(set(epochs.info['bads']).union(set(channels_to_interpolate)))
                epochs.interpolate_bads(reset_bads=True)

        if 'find_bad_channels' in self.preprocessing_steps:
            # Mark bad channels
            try:
                bad_channels = find_bad_channels(epochs, eeg_ref_corr=False)
                if bad_channels:
                    print(f"Bad channels detected: {bad_channels}")
                    epochs.info['bads'] = list(set(epochs.info['bads']).union(set(bad_channels)))
                    epochs.interpolate_bads(reset_bads=True)
            except Exception as error:
                print(f"An exception occurred during bad channel detection for subject {subject}: {error}.")

        if 'find_bad_channels_in_epochs' in self.preprocessing_steps:
            # Find bad channels within epochs
            try:
                bad_channels_epochs = find_bad_channels_in_epochs(epochs)
                if bad_channels_epochs:
                    # Flatten the list if it's a list of lists
                    bad_channels_epochs = [ch for sublist in bad_channels_epochs for ch in sublist]
                    print(f"Bad channels in epochs detected: {bad_channels_epochs}")
                    # Mark them as bad and interpolate
                    epochs.info['bads'] = list(set(epochs.info['bads']).union(set(bad_channels_epochs)))
                    epochs.interpolate_bads(reset_bads=True)
            except Exception as error:
                print(f"An exception occurred during bad channel detection in epochs for subject {subject}: {error}.")

        if 'find_bad_epochs' in self.preprocessing_steps:
            # Mark bad epochs
            try:
                bad_epochs = find_bad_epochs(epochs)
                if bad_epochs:
                    print(f"Dropping bad epochs: {bad_epochs}")
                    epochs.drop(bad_epochs)
            except Exception as error:
                print(f"An exception occurred during bad epoch detection for subject {subject}: {error}.")

        if 'ica' in self.preprocessing_steps:
            # Independent Component Analysis (ICA)
            try:
                ica = ICA(n_components=15, max_iter="auto", random_state=42, method='infomax', fit_params=dict(extended=True))
                ica.fit(epochs)
                label_components(epochs, ica, method='iclabel')
                ica.exclude = []
                for key, components in ica.labels_.items():
                    if key != 'brain' and components:
                        ica.exclude.extend(components)
                if ica.exclude:
                    ica.apply(epochs)
            except Exception as error:
                print(f"An exception occurred during ICA for subject {subject}: {error}.")

        if 'baseline_correction' in self.preprocessing_steps:
            epochs.apply_baseline(epochs.baseline)

        if 'filter_epochs' in self.preprocessing_steps:
            epochs.filter(l_freq=2, h_freq=20, verbose=False)

        if 'set_average_reference' in self.preprocessing_steps:
            epochs.set_eeg_reference('average', projection=False)

        return epochs

    def save_preprocessed_data(self, epochs, subject):
        """
        Save the preprocessed epochs.

        Parameters:
            epochs (Epochs): The preprocessed epochs.
            subject (str): Subject identifier.
        """
        datadir = os.path.join(self.data_path, 'Preproc_eeg_pipeline')
        os.makedirs(datadir, exist_ok=True)
        fname = os.path.join(datadir, f"sub_{subject}_preproc_raw-epo.fif")
        epochs.save(fname, overwrite=True, verbose=False)
        print(f"Saved preprocessed data for subject {subject} to {fname}")

    def run_preprocessing(self, reference_channel):
        """
        Run the preprocessing pipeline for all subjects.

        Parameters:
            reference_channel (str): Name of the reference channel, e.g., 'CZ'.
        """
        for subject in self.subjects:
            raw, events = self.load_subject_raw(subject)
            if raw is None:
                continue
            if self.crop_events:
                raw = self.crop_subject_data(raw, events)
            self.set_montage(raw, subject)
            epochs = self.preprocess_subject_epochs(raw, subject, reference_channel, events)
            if epochs is not None:
                self.save_preprocessed_data(epochs, subject)
            else:
                print(f"Preprocessing failed for subject {subject}.")
