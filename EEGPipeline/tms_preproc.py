from mne_faster import find_bad_channels, find_bad_epochs, find_bad_channels_in_epochs
from mne_icalabel import label_components

import os
import numpy as np
import mne
from mne.preprocessing import ICA
from mne_bids import BIDSPath, read_raw_bids
from mne_faster import find_bad_channels, find_bad_epochs
from mne_icalabel import label_components

class TMSEEGPreprocessor:
    def __init__(
        self, data_loader, output_dir='Preproc_eeg_pipeline',
        preprocessing_steps=None
    ):
        """
        Initialize the TMSPreprocessor.

        Parameters:
            data_loader (TMSDataLoader): An instance of TMSDataLoader.
            output_dir (str): The directory where preprocessed data will be saved.
            preprocessing_steps (list): List of preprocessing steps to perform.
        """
        self.data_loader = data_loader
        self.output_dir = output_dir
        self.preprocessing_steps = preprocessing_steps if preprocessing_steps else []
        self.subjects = self.data_loader.subjects

    def run_preprocessing(self, reference_channel):
        """
        Run the preprocessing pipeline for all subjects.

        Parameters:
            reference_channel (str): Name of the reference channel, e.g., 'CZ'.
        """
        for subject in self.subjects:
            raw, events, channels = self.data_loader.load_subject_data(subject)
            if raw is None:
                continue
            self.set_montage(raw, subject)
            epochs = self.preprocess_subject_epochs(raw, subject, reference_channel)
            if epochs is not None:
                self.save_preprocessed_data(epochs, subject)
            else:
                print(f"Preprocessing failed for subject {subject}.")

    def detect_tms_pulses(self, raw, threshold=1e-4, dead_time=0.01):
        """
        Detect TMS pulses in the raw EEG data.

        Parameters:
            raw (Raw): The raw EEG data.
            threshold (float): Amplitude threshold for detecting pulses.
            dead_time (float): Minimum time between pulses to avoid detecting the same pulse multiple times (in seconds).

        Returns:
            events (numpy.ndarray): An array of events corresponding to TMS pulses.
        """
        from mne.annotations import Annotations

        # Get data from an EEG channel where the TMS artifact is prominent
        data, times = raw.get_data(picks='eeg', return_times=True)
        data = data[0]  # Use the first EEG channel or specify a channel with prominent TMS artifact

        # Compute the absolute derivative of the signal
        derivative = np.abs(np.diff(data))

        # Find where the derivative exceeds the threshold
        pulse_indices = np.where(derivative > threshold)[0]

        # Remove pulses that are too close together (within dead_time)
        sfreq = raw.info['sfreq']
        min_samples = int(dead_time * sfreq)
        pulse_indices = pulse_indices[np.diff(np.concatenate(([0], pulse_indices))) > min_samples]

        if len(pulse_indices) == 0:
            print("No TMS pulses detected.")
            return None

        # Create annotations for the pulses
        onset_times = times[pulse_indices]
        durations = np.full_like(onset_times, 0.001)  # Duration of 1 ms for the TMS pulse
        descriptions = ['TMS'] * len(onset_times)
        annotations = Annotations(onset=onset_times, duration=durations, description=descriptions)

        # Add annotations to raw data
        raw.set_annotations(raw.annotations + annotations)

        # Create events from annotations
        events, _ = mne.events_from_annotations(raw, event_id={'TMS': 1})

        return events

    def set_montage(self, raw, subject):
        """
        Set the montage for the raw data.

        Parameters:
            raw (Raw): The raw EEG data.
            subject (str): Subject identifier.
        """
        if not self.data_loader.montage_path:
            print("No montage path provided. Skipping montage setting.")
            return

        montage_fname = os.path.join(
            self.data_loader.montage_path, f"sub-{subject}", "eeg",
            f"sub-{subject}_task-restingstate_electrodes.tsv"
        )
        if os.path.isfile(montage_fname):
            montage = mne.channels.read_custom_montage(fname=montage_fname)
            # Convert positions from mm to m and adjust axes if necessary
            positions = montage.get_positions()['ch_pos']
            for ch_name, pos in positions.items():
                x, y, z = pos
                # Adjust axes as needed (e.g., swap x and y)
                positions[ch_name] = np.array([y, x, z]) / 1000.0  # Swap x and y, convert mm to m
            montage = mne.channels.make_dig_montage(ch_pos=positions, coord_frame='head')
            raw.set_montage(montage, on_missing='raise', verbose=False)
            # Verify the head radius
            radius, _, _ = mne.bem.fit_sphere_to_headshape(raw.info, units='m')
            print(f"Estimated head radius for subject {subject}: {radius * 100:.2f} cm")
        else:
            print(f"Montage file not found for subject {subject}.")

    def preprocess_subject_epochs(self, raw, subject, reference):
        """
        Preprocess epochs for a subject.

        Parameters:
            raw (Raw): The raw EEG data.
            subject (str): Subject identifier.
            reference (str): Name of the reference channel, e.g., 'CZ'.

        Returns:
            epochs (Epochs): The preprocessed epochs.
        """
        print(f"Preprocessing data for subject {subject}")

        # Apply EEG reference and filtering
        if 'set_eeg_reference' in self.preprocessing_steps:
            raw.set_eeg_reference(ref_channels=[reference])
        if 'filter' in self.preprocessing_steps:
            raw.filter(l_freq=1, h_freq=100, verbose=False)

        # Detect TMS pulses
        events = self.detect_tms_pulses(raw, threshold=1e-4, dead_time=0.01)
        if events is None or not events.size:
            print("No TMS events found. Skipping epoching.")
            return None

        # Create epochs around TMS pulses
        tmin = -0.2  # Start time before TMS pulse
        tmax = 0.8   # End time after TMS pulse

        try:
            epochs = mne.Epochs(raw, events, event_id={'TMS': 1}, tmin=tmin, tmax=tmax,
                                preload=True, baseline=None, reject_by_annotation=True)
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
        if 'detrend' in self.preprocessing_steps:
            epochs._data = mne.filter.detrend(epochs.get_data(), axis=-1)

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
                from mne_faster import find_bad_channels
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
                from mne_faster import find_bad_channels_in_epochs
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
                from mne_faster import find_bad_epochs
                bad_epochs = find_bad_epochs(epochs)
                if bad_epochs:
                    print(f"Dropping bad epochs: {bad_epochs}")
                    epochs.drop(bad_epochs)
            except Exception as error:
                print(f"An exception occurred during bad epoch detection for subject {subject}: {error}.")

        if 'ica' in self.preprocessing_steps:
            # Independent Component Analysis (ICA)
            try:
                ica = ICA(n_components=15, method='fastica', random_state=42, max_iter='auto')
                ica.fit(epochs)

                # Use mne_icalabel to label components
                try:
                    from mne_icalabel import label_components
                    labels = label_components(epochs, ica, method='iclabel')
                    ica.exclude = [idx for idx, label in enumerate(labels['labels']) if label != 'brain']
                    print(f"Excluding ICA components: {ica.exclude}")
                except ImportError:
                    print("mne_icalabel not installed. Proceeding without component labeling.")
                    # Alternatively, exclude components based on PSD or manual inspection

                # Apply ICA to remove the components
                ica.apply(epochs)
            except Exception as error:
                print(f"An exception occurred during ICA for subject {subject}: {error}.")

        if 'artifact_rejection' in self.preprocessing_steps:
            # Automatic artifact rejection based on amplitude threshold
            reject_criteria = dict(eeg=100e-6)  # 100 µV
            epochs.drop_bad(reject=reject_criteria)

        if 'baseline_correction' in self.preprocessing_steps:
            # Use the baseline period specified in epochs.baseline or (None, 0)
            epochs.apply_baseline(epochs.baseline if epochs.baseline is not None else (None, 0))

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
        datadir = os.path.join(self.data_loader.data_path, self.output_dir)
        os.makedirs(datadir, exist_ok=True)
        fname = os.path.join(datadir, f"sub_{subject}_preproc_raw-epo.fif")
        epochs.save(fname, overwrite=True, verbose=False)
        print(f"Saved preprocessed data for subject {subject} to {fname}")
