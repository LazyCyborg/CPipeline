import os
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from transformers import pipeline
from textacy.preprocessing import normalize, remove, replace, pipeline as pp
from functools import partial


class PreprocTranscribeAudio:
    """
    A class for preprocessing audio files and transcribing them using ASR models.
    """

    def __init__(self, audio_dir, model_name="openai/whisper-large-v3", whisper=True):
        """
        Initialize the PreprocTranscribeAudio class.

        Parameters:
        - audio_dir (str): Directory containing the audio files to process.
        - model_name (str): The name of the ASR model to use.
        - whisper (bool): Whether to use Whisper-specific parameters.
        """
        self.audio_dir = audio_dir
        self.files = sorted(os.listdir(audio_dir))
        self.model_name = model_name
        self.whisper = whisper
        self.asr_data = {}
        self.sentences = []
        self.device = self._get_device()
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            device=self.device
        )
        self.text_preprocessor = self._initialize_text_preprocessor()

    def _get_device(self):
        """
        Determine the appropriate device to use (MPS, CUDA, or CPU).

        Returns:
        - device (str or int): The device identifier for the pipeline.
        """
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 0  # First CUDA device
        else:
            device = -1  # CPU
        return device

    def load_audio(self, filename):
        """
        Load an audio file.

        Parameters:
        - filename (str): The name of the audio file to load.

        Returns:
        - speech_tensor (torch.Tensor): The loaded audio tensor.
        - sampling_rate (int): The sampling rate of the audio.
        """
        if filename.endswith('.wav'):
            file_path = os.path.join(self.audio_dir, filename)
            speech_tensor, sampling_rate = torchaudio.load(file_path)
            return speech_tensor, sampling_rate
        else:
            raise ValueError(f"Unsupported file format for file: {filename}")

    def apply_filters(self, speech_tensor, sampling_rate, lowpass_freq=300, highpass_freq=2000):
        """
        Apply lowpass and highpass filters to the audio.

        Parameters:
        - speech_tensor (torch.Tensor): The audio tensor.
        - sampling_rate (int): The sampling rate of the audio.
        - lowpass_freq (float): The cutoff frequency for the lowpass filter.
        - highpass_freq (float): The cutoff frequency for the highpass filter.

        Returns:
        - torch.Tensor: The filtered audio tensor.
        """
        speech_tensor = F.lowpass_biquad(speech_tensor, sampling_rate, lowpass_freq)
        speech_tensor = F.highpass_biquad(speech_tensor, sampling_rate, highpass_freq)
        return speech_tensor

    def resample_audio(self, speech_tensor, sampling_rate, target_rate=16000):
        """
        Resample audio to the target sample rate.

        Parameters:
        - speech_tensor (torch.Tensor): The audio tensor.
        - sampling_rate (int): The original sampling rate.
        - target_rate (int): The target sampling rate.

        Returns:
        - torch.Tensor: The resampled audio tensor.
        """
        resampler = T.Resample(orig_freq=sampling_rate, new_freq=target_rate)
        speech_tensor = resampler(speech_tensor)
        return speech_tensor

    def convert_to_numpy(self, speech_tensor):
        """
        Convert audio tensor to numpy array and handle mono audio.

        Parameters:
        - speech_tensor (torch.Tensor): The audio tensor.

        Returns:
        - numpy.ndarray: The audio data as a numpy array.
        """
        if speech_tensor.size(0) == 1:
            speech_tensor = speech_tensor.squeeze(0)
        else:
            # Convert to mono by averaging channels if stereo
            speech_tensor = torch.mean(speech_tensor, dim=0)
        return speech_tensor.numpy()

    def chunk_audio(self, audio_np, target_rate, chunk_duration=6):
        """
        Chunk the audio into segments of `chunk_duration` seconds.

        Parameters:
        - audio_np (numpy.ndarray): The audio data as a numpy array.
        - target_rate (int): The sampling rate of the audio.
        - chunk_duration (float): The duration of each chunk in seconds.

        Returns:
        - List[numpy.ndarray]: A list of audio chunks.
        """
        chunk_length = int(chunk_duration * target_rate)  # Chunk duration in samples
        num_chunks = (len(audio_np) + chunk_length - 1) // chunk_length  # Ceiling division
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_length
            end = start + chunk_length
            audio_chunk = audio_np[start:end]
            if len(audio_chunk) < chunk_length:
                # Pad the last chunk if necessary
                audio_chunk = np.pad(audio_chunk, (0, chunk_length - len(audio_chunk)), mode='constant')
            chunks.append(audio_chunk)
        return chunks

    def transcribe_chunked_audio(self, audio_chunks):
        """
        Transcribe the audio chunks using the specified model.

        Parameters:
        - audio_chunks (List[numpy.ndarray]): A list of audio chunks.

        Returns:
        - List[str]: A list of transcriptions for each chunk.
        """
        transcriptions = []
        for audio_chunk in audio_chunks:
            if self.whisper:
                transcription = self.transcriber(audio_chunk, generate_kwargs={"language": "swedish"})
            else:
                transcription = self.transcriber(audio_chunk)
            transcriptions.append(transcription['text'])
        return transcriptions

    def _initialize_text_preprocessor(self):
        """
        Initialize the text preprocessing pipeline.

        Returns:
        - Callable: A preprocessing function that can be applied to texts.
        """
        preproc = pp.make_pipeline(
            # Normalization Steps
            normalize.unicode,
            normalize.quotation_marks,
            normalize.hyphenated_words,
            #remove.punctuation,  # Remove all punctuation
            # partial(normalize.repeating_chars, chars=2),  # Truncate repeating chars to max 2
            normalize.bullet_points,
            normalize.whitespace,

            # Removal Steps
            remove.html_tags,
            remove.brackets,

            # Replacement Steps
            partial(replace.urls, repl="_URL_"),
            partial(replace.emails, repl="_EMAIL_"),
            partial(replace.phone_numbers, repl="_PHONE_"),
            partial(replace.user_handles, repl="_USER_"),
            partial(replace.hashtags, repl="_HASHTAG_"),
            partial(replace.emojis, repl="_EMOJI_"),
            partial(replace.numbers, repl="_NUMBER_"),
            partial(replace.currency_symbols, repl="_CURRENCY_"),
        )
        return preproc

    def preprocess_text(self, text):
        """
        Preprocess a single text string using the predefined pipeline.

        Parameters:
        - text (str): The text to preprocess.

        Returns:
        - str: The preprocessed text.
        """
        preproc = self.text_preprocessor(text)
        self.sentences = preproc.split('.')

        return self.sentences

    def preproc_transcribe_audio(self, speech_tensor, sampling_rate,
                                 lowpass_freq=300, highpass_freq=2000,
                                 target_rate=16000, chunk_duration=6):
        """
        Complete preprocessing and transcription pipeline for a single audio tensor.

        Parameters:
        - speech_tensor (torch.Tensor): The audio tensor.
        - sampling_rate (int): The sampling rate of the audio.
        - lowpass_freq (float): The cutoff frequency for the lowpass filter.
        - highpass_freq (float): The cutoff frequency for the highpass filter.
        - target_rate (int): The target sampling rate.
        - chunk_duration (float): The duration of each chunk in seconds.

        Returns:
        - str: The preprocessed transcription of the audio.
        """
        # Apply filters
        speech_tensor = self.apply_filters(speech_tensor, sampling_rate, lowpass_freq, highpass_freq)

        # Resample the audio
        speech_tensor = self.resample_audio(speech_tensor, sampling_rate, target_rate)

        # Convert to numpy
        audio_np = self.convert_to_numpy(speech_tensor)

        # Chunk the audio
        audio_chunks = self.chunk_audio(audio_np, target_rate, chunk_duration)

        # Transcribe the chunked audio
        transcriptions = self.transcribe_chunked_audio(audio_chunks)

        # Combine transcriptions
        full_transcription = " ".join(transcriptions)

        # Preprocess the transcription
        preprocessed_transcription = self.preprocess_text(full_transcription)

        return preprocessed_transcription

    def process_all_files(self, lowpass_freq=300, highpass_freq=2000,
                          target_rate=16000, chunk_duration=6):
        """
        Process and transcribe all audio files in the directory.

        Parameters:
        - lowpass_freq (float): The cutoff frequency for the lowpass filter.
        - highpass_freq (float): The cutoff frequency for the highpass filter.
        - target_rate (int): The target sampling rate.
        - chunk_duration (float): The duration of each chunk in seconds.

        Returns:
        - Dict[str, str]: A dictionary with filenames as keys and preprocessed transcriptions as values.
        """
        transcriptions = {}
        for filename in self.files:
            if filename.endswith('.wav'):
                try:
                    # Load audio
                    speech_tensor, sampling_rate = self.load_audio(filename)

                    # Preprocess and transcribe
                    preprocessed_transcription = self.preproc_transcribe_audio(
                        speech_tensor, sampling_rate,
                        lowpass_freq, highpass_freq,
                        target_rate, chunk_duration
                    )

                    transcriptions[filename] = preprocessed_transcription
                    print(f"Processed {filename}:")
                    print(preprocessed_transcription)
                    print("-" * 50)
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")
            else:
                print(f"Skipping unsupported file format: {filename}")
        return transcriptions
