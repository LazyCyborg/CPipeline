import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne import compute_raw_covariance
from mne.viz import plot_epochs_image
from mne_icalabel import label_components
from mne.preprocessing import ICA, Xdawn
from mne_faster import find_bad_channels, find_bad_epochs, find_bad_channels_in_epochs
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.datasets import fetch_fsaverage

class TMSEEGPreprocessor:
    def __init__(self, raw_file_path, output_dir='Preproc_eeg_pipeline', pipeline_steps=None, step_params=None):
        self.raw_file_path = raw_file_path
        self.output_dir = output_dir
        self.raw = None
        self.epochs = None
        self.evoked = None
        self.ica = None
        self.xdawn = None
        self.stc = None
        self.SS = None
        self.pci = None

        # Default pipeline steps
        self.default_pipeline = [
            'load_data',
            'plot_raw',
            'check_stats',
            'check_data_consistency',
            'check_snr',
            'remove_unused_channels',
            'set_montage',
            'compute_signal_covariance',
            'create_epochs',
            'plot_epochs',
            'find_bad_channels',
            'interpolate_tms_artifact',
            'fix_stim_artifact',
            'find_bad_epochs',
            'apply_filter',
            'set_eeg_reference',
            'apply_ica',
            'apply_xdawn',
            'apply_baseline_correction',
            'apply_ssp',
            'apply_additional_filtering',
            'downsample',
            'compute_evoked',
            'plot_evoked',
            'plot_GFP',
            'compute_csd',
            'prepare_source_space',
            'compute_inverse_solution',
            'bootstrap_significance',
            'calculate_pci'
        ]

        # Use custom pipeline if provided, otherwise use default
        self.pipeline_steps = pipeline_steps if pipeline_steps is not None else self.default_pipeline

        # Default parameters for each step
        self.default_params = {
            'find_bad_channels': {'thres': 2},
            'detect_tms_artifact': {'artifact_window': (-0.02, 0.015), 'threshold_factor': 3.0},
            'find_bad_epochs': {'thres': 1},
            'apply_filter': {'l_freq': 0.1, 'h_freq': None},
            'apply_ica': {'n_components': 20},
            'apply_baseline_correction': {'baseline': (-0.5, 0)},
            'downsample': {'sfreq': 1000},
            'bootstrap_significance': {'n_bootstraps': 1000, 'alpha': 0.05},
            'create_epochs': {'tmin': -0.5, 'tmax': 1.0},
            'fix_stim_artifact': {'tmin': -0.02, 'tmax': 0.02, 'mode': 'linear'},
            'apply_ssp': {'n_eeg': 2},
            'apply_xdawn': {'n_components': 2}
        }

        # Update default parameters with custom parameters if provided
        self.step_params = self.default_params.copy()
        if step_params is not None:
            self.step_params.update(step_params)

    def run_pipeline(self):
        for step in self.pipeline_steps:
            if hasattr(self, step):
                method = getattr(self, step)
                if step in self.step_params:
                    method(**self.step_params[step])
                else:
                    method()
            else:
                print(f"Warning: Method '{step}' not found in TMSEEGPreprocessor class.")

    def load_data(self):
        self.raw = mne.io.read_raw_eeglab(self.raw_file_path, preload=True)
        
    def plot_raw(self):
        with mne.viz.use_browser_backend("matplotlib"):
            fig = self.raw.plot()
        plt.close(fig)

    def check_stats(self, data, label=""):
        raw_data = data.get_data() * 1e6  # Scale to microvolt
        print(f"=== {label} Data Statistics ===")
        print(f"Min: {np.min(raw_data)} µV")
        print(f"Max: {np.max(raw_data)} µV")
        print(f"Mean: {np.mean(raw_data)} µV")
        print(f"Standard Deviation: {np.std(raw_data)} µV")
        print()

    def check_data_consistency(self, data, label=""):
        raw_data = data.get_data() * 1e6  # Scale to microvolt
        if np.isnan(raw_data).any():
            print(f"{label}: NaN values found in the data!")
        max_expected_value = 150  # µV
        if (raw_data > max_expected_value).any() or (raw_data < -max_expected_value).any():
            print(f"{label}: Unexpected large values found (exceeding ±{max_expected_value} µV)")
        else:
            print(f"{label}: All values within expected range.")

    def check_snr(self, data, label=""):
        raw_data = data.get_data() * 1e6  # Scale to microvolt
        signal_power = np.mean(raw_data ** 2, axis=1)
        noise_power = np.std(raw_data, axis=1)
        snr = signal_power / noise_power
        print(f"=== Estimated SNR for Each Channel ({label}) ===")
        print(snr)

    def remove_unused_channels(self):
        for ch in self.raw.info['ch_names']:
            if ch == 'EMG1':
                self.raw.drop_channels(ch)

    def set_montage(self):
        montage = mne.channels.make_standard_montage('standard_1020')
        self.raw.set_montage(montage)

    def compute_signal_covariance(self):
        self.signal_cov = compute_raw_covariance(self.raw)
        self.xdawn = Xdawn(n_components=2, signal_cov=self.signal_cov)

    def create_epochs(self, tmin=-0.5, tmax=1.0):
        events, event_id = mne.events_from_annotations(self.raw)
        self.epochs = mne.Epochs(self.raw, events, event_id=event_id, tmin=tmin, tmax=tmax, 
                                 baseline=None, preload=True, reject_by_annotation=True)
        print(f"Initial number of epochs: {len(self.epochs)}")

    def fix_stim_artifact(self, tmin=-0.02, tmax=0.02, mode='linear'):
        events, event_id = mne.events_from_annotations(self.raw)
        self.epochs = mne.preprocessing.fix_stim_artifact(
            self.epochs, events=events, event_id=event_id, tmin=tmin, tmax=tmax, mode=mode
        )
        print(f"Applied fix_stim_artifact to epochs with mode '{mode}'.")

    def apply_ssp(self, n_eeg=2):
        projs_epochs = mne.compute_proj_epochs(self.epochs, n_eeg=n_eeg, n_jobs=-1, verbose=True)
        self.epochs.add_proj(projs_epochs)
        self.epochs.apply_proj()
        print(f"Applied SSP projections with n_eeg={n_eeg}.")

    def apply_xdawn(self, n_components=2):
        if not hasattr(self, 'xdawn') or self.xdawn is None:
            self.xdawn = Xdawn(n_components=n_components)
        self.xdawn.fit(self.epochs)
        print(f"Applied Xdawn with n_components={n_components}.")

    def plot_epochs(self):
        with mne.viz.use_browser_backend("matplotlib"):
            fig = self.epochs.plot()
        plt.close(fig)

    def find_bad_channels(self, thres=2):
        bad_channels = find_bad_channels(self.epochs, thres=thres)
        if bad_channels:
            print(f"Bad channels detected: {bad_channels}")
            self.epochs.info['bads'] = list(set(self.epochs.info['bads']).union(set(bad_channels)))
            self.epochs.interpolate_bads(reset_bads=True, verbose=True)
            print("Interpolated bad channels.")
        else:
            print("No bad channels detected.")

    def detect_tms_artifact(self, artifact_window=(-0.02, 0.015), threshold_factor=3.0):
        sfreq = self.epochs.info['sfreq']
        times = self.epochs.times
        data = self.epochs.get_data()
        n_epochs, n_channels, n_times = data.shape
        artifact_times = []

        for epoch_idx in range(n_epochs):
            start_time, end_time = artifact_window
            start_idx = np.searchsorted(times, start_time)
            end_idx = np.searchsorted(times, end_time)
            start_idx = max(start_idx, 0)
            end_idx = min(end_idx, n_times - 1)

            epoch_data_window = data[epoch_idx, :, start_idx:end_idx]
            baseline_data = data[epoch_idx, :, :start_idx]
            epoch_data_flat = epoch_data_window.flatten()
            baseline_data_flat = baseline_data.flatten()

            if baseline_data_flat.size == 0:
                baseline_data_flat = data[epoch_idx, :, :].flatten()
            amplitude_threshold = threshold_factor * np.std(baseline_data_flat)

            amplitude_artifact_indices = np.where(np.abs(epoch_data_flat) > amplitude_threshold)[0]
            gradient = np.diff(epoch_data_flat)
            gradient_threshold = threshold_factor * np.std(np.diff(baseline_data_flat))
            gradient_artifact_indices = np.where(np.abs(gradient) > gradient_threshold)[0]
            artifact_indices = np.unique(np.concatenate((amplitude_artifact_indices, gradient_artifact_indices)))

            if artifact_indices.size == 0:
                artifact_start_idx, artifact_end_idx = start_idx, end_idx
            else:
                artifact_start_idx = start_idx + artifact_indices[0] // n_channels
                artifact_end_idx = start_idx + artifact_indices[-1] // n_channels + 1

            artifact_start_time = times[artifact_start_idx]
            artifact_end_time = times[min(artifact_end_idx, n_times - 1)]
            artifact_times.append((artifact_start_time, artifact_end_time))

        return artifact_times

    def interpolate_tms_artifact(self):
        artifact_times = self.detect_tms_artifact()
        data = self.epochs.get_data()
        times = self.epochs.times
        n_epochs, n_channels, n_times = data.shape

        for epoch_idx in range(n_epochs):
            start_time, end_time = artifact_times[epoch_idx]
            if start_time is None or end_time is None:
                continue
            start_sample = np.searchsorted(times, start_time)
            end_sample = np.searchsorted(times, end_time)
            if start_sample <= 2 or end_sample >= n_times - 3:
                continue
            for ch_idx in range(n_channels):
                ts = data[epoch_idx, ch_idx, :]
                x = np.array([start_sample - 3, start_sample - 2, start_sample - 1, 
                              end_sample, end_sample + 1, end_sample + 2])
                y = ts[x]
                x_new = np.arange(start_sample, end_sample)
                f = interp1d(x, y, kind='cubic')
                ts[start_sample:end_sample] = f(x_new)

        self.epochs._data = data
        print("Interpolated TMS artifacts.")

    def find_bad_epochs(self, thres=1):
        bad_epochs = find_bad_epochs(self.epochs, thres=thres)
        if bad_epochs:
            print(f"Dropping bad epochs: {bad_epochs}")
            self.epochs.drop(bad_epochs)
            print("Dropped bad epochs.")
        else:
            print("No bad epochs to drop.")

    def apply_filter(self, l_freq=0.1, h_freq=None):
        self.epochs.filter(l_freq=l_freq, h_freq=h_freq, picks='eeg', method='fir', fir_design='firwin',
                           filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto', verbose=True)

    def set_eeg_reference(self):
        self.epochs.set_eeg_reference(ref_channels='average', projection=True)

    def apply_ica(self, n_components=20):
        self.ica = ICA(n_components=n_components, max_iter="auto", random_state=42, verbose=True)
        self.ica.fit(self.epochs)
        self.ica.plot_components()
        explained_var_ratio = self.ica.get_explained_variance_ratio(self.epochs)
        for channel_type, ratio in explained_var_ratio.items():
            print(f"Fraction of {channel_type} variance explained by all components: {ratio}")
        self.ica.plot_sources(self.epochs, show_scrollbars=True)
        
        # Manual selection of components
        print("Enter the indices of ICA components to exclude (comma-separated):")
        exclude_indices = input().strip().split(',')
        self.ica.exclude = [int(idx) for idx in exclude_indices if idx]
        
        self.ica.apply(self.epochs)

    def apply_baseline_correction(self, baseline=(-0.5, 0)):
        self.epochs.apply_baseline(baseline=baseline)

    def apply_additional_filtering(self):
        self.epochs.filter(l_freq=1, h_freq=90, picks='eeg', method='fir', fir_design='firwin',
                           filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto', verbose=True)
        self.epochs._data = mne.filter.notch_filter(self.epochs._data, Fs=self.epochs.info['sfreq'], 
                                                    freqs=50, notch_widths=1, method='fir', 
                                                    filter_length='auto', verbose=True)
        print("Applied bandpass filter (1-90 Hz) and notch filter at 50 Hz to epochs data.")

    def downsample(self, sfreq=1000):
        self.epochs = self.epochs.resample(sfreq=sfreq)

    def compute_evoked(self):
        self.evoked = self.epochs.average()

    def plot_evoked(self):
        self.evoked.plot(spatial_colors=True, titles='All electrodes', 
                         ylim=dict(eeg=[-6, 6]), show=True, xlim=(-0.1, 0.5))

    def plot_GFP(self, ylim=None, peak_height=None):
        gfp = np.std(self.evoked.data, axis=0)
        gfp_uv = gfp * 1e6
        if peak_height is None:
            peak_height = np.mean(gfp_uv)
        peaks, _ = find_peaks(gfp_uv, height=peak_height)
        times = self.evoked.times * 1000
        if ylim is None:
            ylim = (0, 1.5 * np.max(gfp_uv))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(times, gfp_uv, label='GFP', color='b')
        ax.plot(times[peaks], gfp_uv[peaks], "rx", label='Peaks')
        ax.axvline(x=0, color='r', linestyle='--', linewidth=1, label='Stimulus Onset')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('GFP (µV)')
        ax.set_title('Global Field Power (GFP)')
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlim(-100, 500)
        ax.grid(True)
        ax.legend()
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plt.show()

    def compute_csd(self):
        self.evoked_csd = mne.preprocessing.compute_current_source_density(self.evoked)
        self.evoked_csd.plot_joint(title="Current Source Density")

    def prepare_source_space(self):
        fs_dir = fetch_fsaverage(verbose=True)
        subjects_dir = os.path.dirname(fs_dir)
        self.bem = mne.read_bem_solution(os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif'))
        self.src = mne.read_source_spaces(os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif'))

    def compute_inverse_solution(self):
            self.raw.set_eeg_reference(ref_channels="average", projection=True)
            noise_cov = mne.compute_raw_covariance(self.raw, tmin=self.epochs.tmin, tmax=self.epochs.tmax)
            self.fwd = mne.make_forward_solution(self.raw.info, trans='fsaverage', src=self.src, bem=self.bem)
            self.inverse_operator = make_inverse_operator(self.raw.info, self.fwd, noise_cov)
            self.stc = apply_inverse(self.evoked, self.inverse_operator, lambda2=1./9., method='dSPM')
            print(self.stc)

    def bootstrap_significance(self, n_bootstraps=1000, alpha=0.05):
        data = self.stc.data
        n_sources, n_times = data.shape
        
        # Initialize SS matrix
        self.SS = np.zeros((n_sources, n_times), dtype=int)
        
        for t in range(n_times):
            time_data = data[:, t]
            bootstrap_distribution = np.zeros((n_sources, n_bootstraps))
            
            for b in range(n_bootstraps):
                # Generate bootstrap sample
                bootstrap_sample = np.random.choice(n_sources, size=n_sources, replace=True)
                bootstrap_distribution[:, b] = time_data[bootstrap_sample]
            
            # Calculate p-values
            p_values = np.sum(bootstrap_distribution >= time_data[:, np.newaxis], axis=1) / n_bootstraps
            
            # Apply significance threshold
            self.SS[:, t] = (p_values < alpha).astype(int)
        
        print(f"Shape of SS: {self.SS.shape}")
        print(f"Number of significant points: {np.sum(self.SS)}")


    def lempel_ziv_complexity(binary_sequence):
        binary_sequence = binary_sequence.astype(int)  # Ensure it's integers
        n = len(binary_sequence)
        i, c, l = 0, 1, 1
        h_pos = {}
        h_pos[tuple(binary_sequence[0:l])] = 0
        while i + l < n:
            current = tuple(binary_sequence[i + 1:i + l + 1])
            if current in h_pos:
                l += 1
            else:
                h_pos[current] = i + 1
                i += l
                l = 1
                c += 1
        return c

    def calculate_pci(self):
        if self.SS is None:
            raise ValueError("SS matrix is not computed. Run bootstrap_significance first.")

        # Flatten the SS matrix
        flattened_SS = self.SS.flatten()
        
        # Calculate Lempel-Ziv complexity
        c_L = self.lempel_ziv_complexity(flattened_SS)
        
        # Calculate source entropy
        p_1 = np.sum(flattened_SS) / len(flattened_SS)
        H_L = -p_1 * np.log2(p_1) - (1 - p_1) * np.log2(1 - p_1)
        
        # Normalize complexity
        self.pci = c_L / (len(flattened_SS) * H_L / np.log2(len(flattened_SS)))
        
        print(f"PCI: {self.pci}")

    def run_full_pipeline(self):
        self.run_pipeline()