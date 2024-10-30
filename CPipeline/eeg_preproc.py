# eeg_preprocessor.py

import os
import numpy as np
import mne
from mne.preprocessing import ICA
from mne_bids import BIDSPath
from mne_faster import find_bad_channels, find_bad_epochs, find_bad_channels_in_epochs
from mne_icalabel import label_components

class EEGPreprocessor:
    def __init__(
        self, data_loader, output_dir='Preproc_eeg_pipeline', preprocessing_steps=None
    ):
        """
        Initialize the Preprocessor.

        Parameters:
            data_loader: An instance of EEGDataLoader.
            output_dir (str): Directory to save preprocessed data.
            preprocessing_steps (list): List of preprocessing steps to perform.
                Options include: 'set_eeg_reference', 'filter', 'interpolate_bad_channels',
                'find_bad_channels', 'find_bad_epochs', 'ica', 'baseline_correction',
                'filter_epochs', 'set_average_reference', 'crop_data', 'find_bad_channels_in_epochs'.
        """
        self.data_loader = data_loader
        self.output_dir = output_dir
        self.preprocessing_steps = preprocessing_steps if preprocessing_steps else []
        self.subjects = self.data_loader.get_subjects()

        # Mapping step names to functions
        self.step_functions_raw = {
            'set_eeg_reference': self.set_eeg_reference,
            'filter': self.filter_raw,
            'crop_data': self.crop_subject_data,
        }

        self.step_functions_epochs = {
            'interpolate_bad_channels': self.interpolate_bad_channels,
            'find_bad_channels': self.find_bad_channels_method,
            'find_bad_channels_in_epochs': self.find_bad_channels_in_epochs_method,
            'find_bad_epochs': self.find_bad_epochs_method,
            'ica': self.apply_ica,
            'baseline_correction': self.apply_baseline_correction,
            'filter_epochs': self.filter_epochs,
            'set_average_reference': self.set_average_reference,
        }

    def run_preprocessing(self, reference_channel):
        """
        Run the preprocessing pipeline for all subjects.

        Parameters:
            reference_channel (str): Name of the reference channel, e.g., 'CZ'.
        """
        for subject in self.subjects:
            raw, events, channels = self.data_loader.load_subject_data(subject)
            if raw is None:
                print(f"No raw data found for subject {subject}. Skipping.")
                continue

            preprocessing_info = {}

            # Set montage
            self.set_montage(raw, subject)
            preprocessing_info['set_montage'] = "Applied standard or custom montage"

            # Apply preprocessing steps to raw data
            raw = self.apply_preprocessing_steps_raw(raw, reference_channel, events)
            if raw is None:
                continue

            # Epoching
            if events is None or not events.size:
                print("No events found. Skipping epoching.")
                epochs = None
                initial_epoch_count = 0
            else:
                tmin, tmax = -0.5, 1.0
                try:
                    unique_event_ids = np.unique(events[:, 2])
                    event_id = {str(e): int(e) for e in unique_event_ids}
                    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                                        baseline=None, preload=True, reject_by_annotation=True)
                    initial_epoch_count = len(epochs)
                    print(f"Initial number of epochs for subject {subject}: {initial_epoch_count}")
                except Exception as error:
                    print(f"An exception occurred during initial epoching for subject {subject}: {error}.")
                    initial_epoch_count = 0
                    epochs = None

            if epochs is not None:
                # Preprocess epochs
                epochs = self.apply_preprocessing_steps_epochs(epochs, subject)
                if epochs is None:
                    print(f"Preprocessing failed for subject {subject}.")
                    continue

                # Save preprocessed data
                self.save_preprocessed_data(epochs, subject)
            else:
                print("No epochs available to preprocess.")

    def apply_preprocessing_steps_raw(self, raw, reference_channel, events):
        """
        Apply preprocessing steps to raw data.

        Parameters:
            raw (Raw): The raw EEG data.
            reference_channel (str): Name of the reference channel.
            events (ndarray): The events array.

        Returns:
            raw (Raw): The preprocessed raw data.
        """
        for step in self.preprocessing_steps:
            func = self.step_functions_raw.get(step, None)
            if func is not None:
                try:
                    if step == 'set_eeg_reference':
                        raw = func(raw, reference_channel)
                    elif step == 'crop_data':
                        raw = func(raw, events)
                    else:
                        raw = func(raw)
                    print(f"Applied {step} to raw data.")
                except Exception as error:
                    print(f"An error occurred during {step} on raw data: {error}")
            else:
                # Step not applicable to raw data
                pass
        return raw

    def apply_preprocessing_steps_epochs(self, epochs, subject):
        """
        Apply preprocessing steps to epochs.

        Parameters:
            epochs (Epochs): The epochs to preprocess.
            subject (str): Subject identifier.

        Returns:
            epochs (Epochs): The preprocessed epochs.
        """
        for step in self.preprocessing_steps:
            func = self.step_functions_epochs.get(step, None)
            if func is not None:
                try:
                    epochs = func(epochs)
                    print(f"Applied {step} to epochs.")
                except Exception as error:
                    print(f"An error occurred during {step} on epochs: {error}")
            else:
                # Step not applicable to epochs
                pass
        return epochs

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

    def set_eeg_reference(self, raw, reference_channel):
        """
        Set the EEG reference channel.

        Parameters:
            raw (Raw): The raw EEG data.
            reference_channel (str): Name of the reference channel.

        Returns:
            raw (Raw): The raw data with reference set.
        """
        raw.set_eeg_reference(ref_channels=[reference_channel])
        return raw

    def filter_raw(self, raw):
        """
        Apply band-pass filter to raw data.

        Parameters:
            raw (Raw): The raw EEG data.

        Returns:
            raw (Raw): The filtered raw data.
        """
        raw.filter(l_freq=1, h_freq=100, verbose=False)
        return raw

    def crop_subject_data(self, raw, events):
        """
        Crop the raw data based on event markers.

        Parameters:
            raw (Raw): The raw EEG data.
            events (ndarray): The events array.

        Returns:
            raw_cropped (Raw): The cropped raw data.
        """
        if events is None or not events.size:
            print("No events found. Skipping cropping.")
            return raw

        # Create a list of start and stop times based on event markers
        start_marker = self.data_loader.event_markers.get('start_marker')
        stop_marker = self.data_loader.event_markers.get('stop_marker')
        if start_marker is None or stop_marker is None:
            print("Start or stop marker not defined in event_markers.")
            return raw

        event_times = {start_marker: [], stop_marker: []}

        for event in events:
            event_id = event[2]
            if event_id == start_marker:
                event_times[start_marker].append(event[0] / raw.info['sfreq'])
            elif event_id == stop_marker:
                event_times[stop_marker].append(event[0] / raw.info['sfreq'])

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

    def interpolate_bad_channels(self, epochs):
        """
        Interpolate channels with zero variance.

        Parameters:
            epochs (Epochs): The epochs to process.

        Returns:
            epochs (Epochs): The epochs with bad channels interpolated.
        """
        data = epochs.get_data()
        zero_variance_channels = np.where(data.var(axis=2).mean(axis=0) == 0)[0]
        if zero_variance_channels.size > 0:
            channels_to_interpolate = [epochs.ch_names[idx] for idx in zero_variance_channels]
            print(f"Interpolating channels with zero variance: {channels_to_interpolate}")
            epochs.info['bads'] = list(set(epochs.info['bads']).union(set(channels_to_interpolate)))
            epochs.interpolate_bads(reset_bads=True)
        return epochs

    def find_bad_channels_method(self, epochs):
        """
        Detect and interpolate bad channels.

        Parameters:
            epochs (Epochs): The epochs to process.

        Returns:
            epochs (Epochs): The epochs with bad channels interpolated.
        """
        try:
            bad_channels = find_bad_channels(epochs, eeg_ref_corr=False)
            if bad_channels:
                print(f"Bad channels detected: {bad_channels}")
                epochs.info['bads'] = list(set(epochs.info['bads']).union(set(bad_channels)))
                epochs.interpolate_bads(reset_bads=True)
        except Exception as error:
            print(f"An exception occurred during bad channel detection: {error}.")
        return epochs

    def find_bad_channels_in_epochs_method(self, epochs):
        """
        Find bad channels within epochs and interpolate.

        Parameters:
            epochs (Epochs): The epochs to process.

        Returns:
            epochs (Epochs): The epochs with bad channels interpolated.
        """
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
            print(f"An exception occurred during bad channel detection in epochs: {error}.")
        return epochs

    def find_bad_epochs_method(self, epochs):
        """
        Detect and drop bad epochs.

        Parameters:
            epochs (Epochs): The epochs to process.

        Returns:
            epochs (Epochs): The epochs with bad epochs dropped.
        """
        try:
            bad_epochs = find_bad_epochs(epochs)
            if bad_epochs:
                print(f"Dropping bad epochs: {bad_epochs}")
                epochs.drop(bad_epochs)
        except Exception as error:
            print(f"An exception occurred during bad epoch detection: {error}.")
        return epochs

    def apply_ica(self, epochs):
        """
        Apply Independent Component Analysis (ICA) to epochs.

        Parameters:
            epochs (Epochs): The epochs to process.

        Returns:
            epochs (Epochs): The epochs after ICA.
        """
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
                print(f"Applied ICA and removed components: {ica.exclude}")
        except Exception as error:
            print(f"An exception occurred during ICA: {error}.")
        return epochs

    def apply_baseline_correction(self, epochs):
        """
        Apply baseline correction to epochs.

        Parameters:
            epochs (Epochs): The epochs to process.

        Returns:
            epochs (Epochs): The baseline-corrected epochs.
        """
        epochs.apply_baseline(epochs.baseline)
        return epochs

    def filter_epochs(self, epochs):
        """
        Apply band-pass filter to epochs.

        Parameters:
            epochs (Epochs): The epochs to process.

        Returns:
            epochs (Epochs): The filtered epochs.
        """
        epochs.filter(l_freq=2, h_freq=20, verbose=False)
        return epochs

    def set_average_reference(self, epochs):
        """
        Set average reference for epochs.

        Parameters:
            epochs (Epochs): The epochs to process.

        Returns:
            epochs (Epochs): The epochs with average reference set.
        """
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
