import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mne.time_frequency import psd_array_welch
from mne_bids import (
    BIDSPath,
    find_matching_paths,
    get_entity_vals,
    make_report,
    print_dir_tree,
    read_raw_bids,
)
from pycrostates.preprocessing import extract_gfp_peaks
from pycrostates.cluster import ModKMeans
from pycrostates.io import ChData
from pycrostates.segmentation import (
    auto_information_function,
    excess_entropy_rate,
    partial_auto_information_function,
)
from antropy import sample_entropy, perm_entropy  # Import entropy functions
from antropy import lziv_complexity  # Import Lempel-Ziv Complexity function

class TS_Feature:
    def __init__(self, data_dir, td_seconds=1):
        self.time_delta = pd.Timedelta(seconds=td_seconds)
        self.data_dir = data_dir
        self.individual_cluster_centers = []
        self.segmentations = []
        self.segmentation_labels = []
        self.microstate_dfs = []
        self.entropy_dfs = []  # List to store entropy DataFrames
        self.ModK = None
        self.group_cluster_centers = None
        self.info = None
        self.date_time_idx = None
        self.chs = []
        self.df = None  # DataFrame to store segmentation labels
        self.entropy_df = None  # DataFrame to store entropy data

    def extract_entropy_features(
        self, subjects=None, entropy_type='sample', **entropy_kwargs
    ):
        """
        Extract entropy features from preprocessed EEG data and create a DataFrame.

        Parameters:
            subjects (list): List of subject IDs to include. If None, include all subjects.
            entropy_type (str): Type of entropy to compute ('sample' or 'permutation').
            entropy_kwargs: Additional keyword arguments for the entropy function.

        Returns:
            pd.DataFrame: DataFrame containing entropy features for all subjects.
        """
        file_suffix = "_preproc_raw-epo.fif"
        read_function = mne.read_epochs

        files = [
            fname for fname in os.listdir(self.data_dir)
            if fname.startswith("sub_") and fname.endswith(file_suffix)
        ]
        files_sorted = sorted(files, key=lambda x: int(fname.split('_')[1]))

        # Filter files based on the 'subjects' list
        if subjects is not None:
            files_filtered = []
            for fname in files_sorted:
                try:
                    n = int(fname.split('_')[1])
                    if n in subjects:
                        files_filtered.append(fname)
                except ValueError:
                    continue
            files_sorted = files_filtered

        for fname in files_sorted:
            try:
                n = int(fname.split('_')[1])
            except ValueError:
                continue

            fpath = os.path.join(self.data_dir, fname)
            if not os.path.isfile(fpath):
                continue

            print("Reading:", fname, '\n', fpath)
            epochs = read_function(fpath, preload=True)

            # Get the data from epochs
            data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

            # Initialize list to store entropy for each epoch
            entropy_list = []

            for epoch_data in data:
                # epoch_data shape: (n_channels, n_times)
                entropy_values = []
                for channel_data in epoch_data:
                    if entropy_type == 'sample':
                        ent = sample_entropy(channel_data, **entropy_kwargs)
                    elif entropy_type == 'permutation':
                        ent = perm_entropy(channel_data, **entropy_kwargs)
                    else:
                        raise ValueError("Invalid entropy type. Choose 'sample' or 'permutation'.")
                    entropy_values.append(ent)
                entropy_list.append(entropy_values)

            entropy_data = np.array(entropy_list)  # shape: (n_epochs, n_channels)

            # Create DataFrame
            entropy_df = pd.DataFrame(
                data=entropy_data,
                columns=epochs.ch_names
            )

            # Create a DatetimeIndex for each epoch
            subject_start_time = pd.Timestamp("2000-01-01 00:00:00") + self.time_delta
            datetime_index = [
                subject_start_time + i * self.time_delta for i in range(len(entropy_df))
            ]
            entropy_df.index = datetime_index
            entropy_df.index.name = 'timestamp'

            entropy_df['subject'] = n

            self.entropy_dfs.append(entropy_df)

        # Concatenate all entropy DataFrames into one
        if self.entropy_dfs:
            self.entropy_df = pd.concat(self.entropy_dfs)
            return self.entropy_df
        else:
            print("No entropy data extracted.")
            return None

    def save_entropy_hdf(self, fname):
        """
        Save the entropy DataFrame to an HDF5 file.

        Parameters:
            fname (str): Filename for the HDF5 file.
        """
        if self.entropy_df is not None and not self.entropy_df.empty:
            self.entropy_df.to_hdf(f"entropy_data_{fname}.h5", key='entropy_data', mode='w')
            print(f"Entropy data saved to entropy_data_{fname}.h5")
        else:
            print("Entropy DataFrame is empty. Please extract entropy features before saving.")

    def extract_lzc_features(
        self, subjects=None, normalize=True
    ):
        """
        Extract Lempel-Ziv Complexity (LZC) features from preprocessed EEG data.

        Parameters:
            subjects (list): List of subject IDs to include. If None, include all subjects.
            normalize (bool): Whether to normalize the LZC values.

        Returns:
            pd.DataFrame: DataFrame containing LZC features for all subjects.
        """
        file_suffix = "_preproc_raw-epo.fif"
        read_function = mne.read_epochs

        files = [
            fname for fname in os.listdir(self.data_dir)
            if fname.startswith("sub_") and fname.endswith(file_suffix)
        ]
        files_sorted = sorted(files, key=lambda x: int(x.split('_')[1]))

        # Filter files based on the 'subjects' list
        if subjects is not None:
            files_filtered = []
            for fname in files_sorted:
                try:
                    n = int(fname.split('_')[1])
                    if n in subjects:
                        files_filtered.append(fname)
                except ValueError:
                    continue
            files_sorted = files_filtered

        for fname in files_sorted:
            try:
                n = int(fname.split('_')[1])
            except ValueError:
                continue

            fpath = os.path.join(self.data_dir, fname)
            if not os.path.isfile(fpath):
                continue

            print("Reading:", fname, '\n', fpath)
            epochs = read_function(fpath, preload=True)

            # Get the data from epochs
            data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

            # Initialize list to store LZC for each epoch
            lzc_list = []

            for epoch_data in data:
                # epoch_data shape: (n_channels, n_times)
                lzc_values = []
                for channel_data in epoch_data:
                    # Compute LZC for the channel data
                    lzc = lziv_complexity(
                        channel_data,
                        normalize=normalize
                    )
                    lzc_values.append(lzc)
                lzc_list.append(lzc_values)

            lzc_data = np.array(lzc_list)  # shape: (n_epochs, n_channels)

            # Create DataFrame
            lzc_df = pd.DataFrame(
                data=lzc_data,
                columns=epochs.ch_names
            )

            # Create a DatetimeIndex for each epoch
            subject_start_time = pd.Timestamp("2000-01-01 00:00:00") + self.time_delta
            datetime_index = [
                subject_start_time + i * self.time_delta for i in range(len(lzc_df))
            ]
            lzc_df.index = datetime_index
            lzc_df.index.name = 'timestamp'

            lzc_df['subject'] = n

            self.entropy_dfs.append(lzc_df)

        # Concatenate all LZC DataFrames into one
        if self.entropy_dfs:
            self.entropy_df = pd.concat(self.entropy_dfs)
            return self.entropy_df
        else:
            print("No LZC data extracted.")
            return None
    
    def save_lzc_hdf(self, fname):
        """
        Save the LZC DataFrame to an HDF5 file.

        Parameters:
            fname (str): Filename for the HDF5 file.
        """
        if self.entropy_df is not None and not self.entropy_df.empty:
            self.entropy_df.to_hdf(f"lzc_data_{fname}.h5", key='lzc_data', mode='w')
            print(f"LZC data saved to lzc_data_{fname}.h5")
        else:
            print("LZC DataFrame is empty. Please extract LZC features before saving.")


    def extract_psd_features(
        self, fmin=1, fmax=40, tmin=None, tmax=None, n_fft=256, subjects=None
    ):
        """
        Extract PSD features from preprocessed EEG data and create a DataFrame.

        Parameters:
            fmin (float): Minimum frequency of interest.
            fmax (float): Maximum frequency of interest.
            tmin (float): Start time for PSD calculation.
            tmax (float): End time for PSD calculation.
            n_fft (int): Number of FFT points.
            subjects (list): List of subject IDs to include. If None, include all subjects.

        Returns:
            pd.DataFrame: DataFrame containing PSD features for all subjects.
        """
        file_suffix = "_preproc_raw-epo.fif"
        read_function = mne.read_epochs

        files = [
            fname for fname in os.listdir(self.data_dir)
            if fname.startswith("sub_") and fname.endswith(file_suffix)
        ]
        files_sorted = sorted(files, key=lambda x: int(x.split('_')[1]))

        # Filter files based on the 'subjects' list
        if subjects is not None:
            files_filtered = []
            for fname in files_sorted:
                try:
                    n = int(fname.split('_')[1])
                    if n in subjects:
                        files_filtered.append(fname)
                except ValueError:
                    continue
            files_sorted = files_filtered

        for fname in files_sorted:
            try:
                n = int(fname.split('_')[1])
            except ValueError:
                continue

            fpath = os.path.join(self.data_dir, fname)
            if not os.path.isfile(fpath):
                continue

            print("Reading:", fname, '\n', fpath)
            epochs = read_function(fpath, preload=True)

            # Get the data from epochs
            data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
            sfreq = epochs.info['sfreq']

            # Initialize list to store PSD for each epoch
            psd_list = []
            freqs = None

            for epoch_data in data:
                # epoch_data shape: (n_channels, n_times)
                psd, freqs = psd_array_welch(
                    epoch_data,
                    sfreq=sfreq,
                    fmin=fmin,
                    fmax=fmax,
                    n_fft=n_fft,
                    average='mean',
                    verbose=False
                )
                psd_list.append(psd)

            psd_data = np.array(psd_list)  # shape: (n_epochs, n_channels, n_freqs)

            # Extract number of epochs, channels, and frequency bins
            n_epochs, n_channels, n_freqs = psd_data.shape

            # Create column MultiIndex
            column_index = pd.MultiIndex.from_product(
                [epochs.ch_names, freqs],
                names=['channel', 'frequency']
            )

            # Flatten the PSD data: reshape to (n_epochs, n_channels * n_freqs)
            psd_data_flat = psd_data.reshape((n_epochs, n_channels * n_freqs))

            # Create a DatetimeIndex for each epoch
            subject_start_time = pd.Timestamp(f"2000-01-01 00:00:00") + self.time_delta
            datetime_index = [
                subject_start_time + i * self.time_delta for i in range(n_epochs)
            ]

            # Create DataFrame
            psd_df = pd.DataFrame(
                data=psd_data_flat,
                index=datetime_index,
                columns=column_index
            )
            psd_df['subject'] = n

            self.psd_dfs.append(psd_df) 

        # Concatenate all PSD DataFrames into one
        if self.psd_dfs:
            self.psd_df = pd.concat(self.psd_dfs)
            self.psd_df.index.name = 'timestamp'
            return self.psd_df
        else:
            print("No PSD data extracted.")
            return None


    def save_psd_hdf(self, fname):
        """
        Save the PSD DataFrame to an HDF5 file.

        Parameters:
            fname (str): Filename for the HDF5 file.
        """
        if self.psd_df is not None and not self.psd_df.empty:
            self.psd_df.to_hdf(f"psd_data_{fname}.h5", key='psd_data', mode='w')
            print(f"PSD data saved to psd_data_{fname}.h5")
        else:
            print("PSD DataFrame is empty. Please extract PSD features before saving.")

   
    def cluster_eeg(
    self, n_clusters=5, n_jobs=-1, random_state=42, max_iter=1000,
    min_peak_distance=2, verbose=True, reject_by_annotation=True, raw=True,
    subjects=None  # Add subjects parameter
    ):
        """
        Cluster EEG data using Modified K-Means algorithm.

        Parameters:
            n_clusters (int): Number of clusters.
            n_jobs (int): The number of jobs to run in parallel. If -1, it is set to the number of CPU cores. Requires the joblib package.
            random_state (int): Random seed.
            max_iter (int): Maximum number of iterations.
            min_peak_distance (int): Minimum distance between peaks.
            verbose (bool): Verbosity flag.
            reject_by_annotation (bool): Whether to reject data based on annotations.
            raw (bool): If True, process raw data; else, process epochs.
            subjects (list): List of subject IDs to include. If None, include all subjects.
        
        Returns:
            ModKMeans: The fitted Modified K-Means model.
        """
        if raw:
            file_suffix = "_preproc_raw.fif"
            read_function = mne.io.read_raw_fif
        else:
            file_suffix = "_preproc_raw-epo.fif"
            read_function = mne.read_epochs

        files = [
            fname for fname in os.listdir(self.data_dir)
            if fname.startswith("sub_") and fname.endswith(file_suffix)
        ]
        files_sorted = sorted(files, key=lambda x: int(x.split('_')[1]))

        # Filter files based on the 'subjects' list
        if subjects is not None:
            files_filtered = []
            for fname in files_sorted:
                try:
                    n = int(fname.split('_')[1])
                    if n in subjects:
                        files_filtered.append(fname)
                except ValueError:
                    continue
            files_sorted = files_filtered

        for fname in files_sorted:
            try:
                n = int(fname.split('_')[1])
            except ValueError:
                continue

            fpath = os.path.join(self.data_dir, fname)
            if not os.path.isfile(fpath):
                continue

            print("Reading:", fname, '\n', fpath)
            data = read_function(fpath, preload=True)

            # Extract GFP peaks
            gfp_peaks = extract_gfp_peaks(
                data, min_peak_distance=min_peak_distance,
                verbose=verbose, reject_by_annotation=reject_by_annotation
            )

            if gfp_peaks.get_data().size > 0:
                modk = ModKMeans(
                    n_clusters=n_clusters, random_state=random_state, max_iter=max_iter
                )
                modk.fit(gfp_peaks, n_jobs=n_jobs)
                cluster_centers = modk.cluster_centers_

                if len(modk.info['ch_names']) == 64:
                    self.individual_cluster_centers.append(cluster_centers)
                    print(f"Cluster center shape: {cluster_centers.shape}")
                else:
                    print("Number of channels is not 64. Skipping.")
                    continue
            else:
                print("No valid data found. Continuing.")
                continue

        if not self.individual_cluster_centers:
            raise ValueError("No valid data found for clustering.")

        # Aggregate cluster centers and fit the group model
        self.group_cluster_centers = np.vstack(self.individual_cluster_centers)
        # Create a ChData object with the cluster centers
        chdata = ChData(self.group_cluster_centers.T, modk.info)


        self.ModK = ModKMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=1000, max_iter=max_iter
        )
        self.ModK.fit(chdata)

        return self.ModK

    def segment_eeg(
        self, reject_by_annotation=True, factor=10, half_window_size=10,
        min_segment_length=5, reject_edges=True, raw=True, subjects=None
    ):
        """
        Segment EEG data using the fitted Modified K-Means model.

        Parameters:
            reject_by_annotation (bool): Whether to reject data based on annotations.
            factor (int): Factor for rejection threshold.
            half_window_size (int): Half window size for segmentation.
            min_segment_length (int): Minimum segment length.
            reject_edges (bool): Whether to reject edges.
            raw (bool): If True, process raw data; else, process epochs.

        Returns:
           list: Segmentation objects for each subject, list: Segmentation labels for each subject.
        """
        if raw:
            file_suffix = "_preproc_raw.fif"
            read_function = mne.io.read_raw_fif
        else:
            file_suffix = "_preproc_raw-epo.fif"
            read_function = mne.read_epochs

        files = [
            fname for fname in os.listdir(self.data_dir)
            if fname.startswith("sub_") and fname.endswith(file_suffix)
        ]
        files_sorted = sorted(files, key=lambda x: int(x.split('_')[1]))

        # Filter files based on the 'subjects' list
        if subjects is not None:
            files_filtered = []
            for fname in files_sorted:
                try:
                    n = int(fname.split('_')[1])
                    if n in subjects:
                        files_filtered.append(fname)
                except ValueError:
                    continue
            files_sorted = files_filtered

        for fname in files_sorted:
            try:
                n = int(fname.split('_')[1])
            except ValueError:
                continue

            fpath = os.path.join(self.data_dir, fname)
            if not os.path.isfile(fpath):
                continue

            print("Reading:", fname, '\n', fpath)
            data = read_function(fpath, preload=True)

            if len(data.info['ch_names']) == 64:
                segmentation = self.ModK.predict(
                    data,
                    reject_by_annotation=reject_by_annotation,
                    factor=factor,
                    half_window_size=half_window_size,
                    min_segment_length=min_segment_length,
                    reject_edges=reject_edges,
                )
                self.segmentations.append((n, segmentation))
                self.segmentation_labels.append((n, segmentation.labels))
            else:
                print("Number of channels is not 64. Skipping.")
                continue

        return self.segmentations, self.segmentation_labels
    
    def plot_segmentation(self, subject, cmap='inferno', figsize = (45, 10), width_ratios=[70, 10]):
        """
        Plot microstate segmentation of one subject.
        
        Parameters:
            subject (int): Subject number to plot
            figsize (tuple): Size of the figure
            width_ratios (list): Ratio between segmentation and colorbar
        """
        if self.segmentations:

            figsize = figsize
            # Create a figure with GridSpec to allocate space for the colorbar
            fig = plt.figure(figsize=figsize, dpi=30)
            gs = GridSpec(1, 2, width_ratios=width_ratios)  

            # Create main axes and colorbar axes
            ax = fig.add_subplot(gs[0])
            cbar_ax = fig.add_subplot(gs[1])

            # Get the segmentation for a specific subject from the list of segmentations (subject segmentations)
            s = self.segmentations[subject][1]

            # Plot the segmentation with custom colormap on custom axes
            s.plot(cmap=cmap, axes=ax, cbar_axes=cbar_ax)

            # Adjust the layout
            fig.tight_layout()


        else:
            print("Data not segmented. No segmentation to plot.")


    def create_segmentation_df(self, remove_space=True):
        """
        Create a DataFrame from the segmentation labels for further analysis.
        """
        if not self.segmentation_labels:
            print("EEG data is not segmented. Please run 'segment_eeg' first.")
            return

        for n, labels in self.segmentation_labels:
            subject_start_time = pd.Timestamp("2000-01-01 00:00:00") + self.time_delta
            labels_flat = labels.flatten()
            print(f"Labels shape: {labels_flat.shape}")

            datetime_index = [
                subject_start_time + i * self.time_delta for i in range(len(labels_flat))
            ]
            subjects_array = np.repeat(n, len(labels_flat))
            print(f"Subject array shape: {subjects_array.shape}")

            df = pd.DataFrame(
                {'target': labels_flat, 'subject': subjects_array},
                index=datetime_index
            )
            if remove_space == True:
                # Remove rows with target == -1
                df = df[df['target'] != -1]

            self.microstate_dfs.append(df)

        self.microstate_df = pd.concat(self.microstate_dfs)
        self.microstate_df.index.name = 'timestamp'
        self.df = self.microstate_df.reset_index()
        self.df.set_index('timestamp', inplace=True)

        return self.microstate_df

    def save_segmentation_hdf(self, fname):
        """
        Save the segmentation DataFrame to an HDF5 file.

        Parameters:
            fname (str): Filename for the HDF5 file.
        """
        if self.df is not None and not self.df.empty:
            self.df.to_hdf(f"microstate_data_{fname}.h5", key='microstate_data', mode='w')
            print(f"Segmentation data saved to microstate_data_{fname}.h5")
        else:
            print("Segmentation DataFrame is empty. Please create the DataFrame before saving.")
    

    def plot_ModK(self, dpi=100, sizex=12, sizey=5):
        """
        Plot the cluster centers from the Modified K-Means model.

        Parameters: 
            dpi (int): DPI of the plot.
            sizex (int): Width in inches
            sizey (int): Height in inches
            
        """
        if self.ModK:
            fig = self.ModK.plot()
            fig.set_size_inches(sizex, sizey)
            fig.set_dpi(dpi)
        else:
            print("ModK is not initialized. Please run clustering first.")
