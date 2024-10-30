import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from transformers import pipeline
from textacy.preprocessing import normalize, remove, replace, pipeline as pp
from functools import partial
from typing import Dict, List, Optional, Union, Tuple
import logging

from .configs import AudioConfig, PreprocessConfig

class PreprocTranscribeAudio:
    """
    A class for preprocessing audio files and transcribing them using ASR models.
    """

    def __init__(
        self,
        audio_dir: str,
        model_name: str = "openai/whisper-large-v3",
        whisper: bool = True,
        audio_config: Optional[AudioConfig] = None,
        preprocess_config: Optional[PreprocessConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the PreprocTranscribeAudio class.

        Args:
            audio_dir: Directory containing audio files
            model_name: Name of the ASR model to use
            whisper: Whether to use Whisper-specific parameters
            audio_config: Configuration for audio processing
            preprocess_config: Configuration for text preprocessing
            device: Computing device (auto-detected if None)
        """
        self.audio_dir = audio_dir
        self.files = sorted(os.listdir(audio_dir))
        self.model_name = model_name
        self.whisper = whisper
        
        # Use provided configs or defaults
        self.audio_config = audio_config or AudioConfig()
        self.preprocess_config = preprocess_config or PreprocessConfig()
        
        # Initialize other attributes
        self.device = self._get_device(device)
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            device=self.device if isinstance(self.device, int) else -1
        )
        self.text_preprocessor = self._initialize_text_preprocessor()
        self.transcriptions = {}
        self.processed_data = pd.DataFrame()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _get_device(self, device: Optional[str] = None) -> Union[str, int]:
        """Determine the appropriate device to use."""
        if device is not None:
            return device
        
        if torch.cuda.is_available():
            return 0
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return -1

    def _initialize_text_preprocessor(self) -> callable:
        """Initialize the text preprocessing pipeline based on config."""
        steps = [
            normalize.unicode,
            normalize.whitespace,
            normalize.bullet_points,
            remove.html_tags,
            remove.brackets,
            partial(replace.urls, repl="_URL_"),
            partial(replace.emails, repl="_EMAIL_"),
            partial(replace.phone_numbers, repl="_PHONE_"),
            partial(replace.user_handles, repl="_USER_"),
            partial(replace.hashtags, repl="_HASHTAG_"),
            partial(replace.emojis, repl="_EMOJI_"),
            partial(replace.numbers, repl="_NUMBER_"),
            partial(replace.currency_symbols, repl="_CURRENCY_"),
        ]

        if self.preprocess_config.normalize_chars:
            steps.extend([
                normalize.quotation_marks,
                normalize.hyphenated_words,
            ])

        if self.preprocess_config.remove_punctuation:
            steps.append(remove.punctuation)

        if self.preprocess_config.max_repeating_chars:
            steps.append(
                partial(normalize.repeating_chars, 
                       chars=self.preprocess_config.max_repeating_chars)
            )

        return pp.make_pipeline(*steps)

    def load_audio(self, filename: str) -> Tuple[torch.Tensor, int]:
        """Load an audio file and validate its format."""
        if not filename.endswith('.wav'):
            raise ValueError(f"Unsupported file format for file: {filename}")
            
        file_path = os.path.join(self.audio_dir, filename)
        try:
            speech_tensor, sampling_rate = torchaudio.load(file_path)
            self.logger.info(f"Loaded audio file: {filename}")
            return speech_tensor, sampling_rate
        except Exception as e:
            self.logger.error(f"Failed to load audio file {filename}: {str(e)}")
            raise

    def process_audio(
        self,
        speech_tensor: torch.Tensor,
        sampling_rate: int
    ) -> torch.Tensor:
        """Process audio with filtering and resampling."""
        # Apply filters
        speech_tensor = F.lowpass_biquad(
            speech_tensor,
            sampling_rate,
            self.audio_config.lowpass_freq
        )
        speech_tensor = F.highpass_biquad(
            speech_tensor,
            sampling_rate,
            self.audio_config.highpass_freq
        )

        # Resample audio
        speech_tensor = T.Resample(
            orig_freq=sampling_rate,
            new_freq=self.audio_config.target_rate
        )(speech_tensor)

        # Convert to mono if needed
        if speech_tensor.size(0) > 1:
            speech_tensor = torch.mean(speech_tensor, dim=0, keepdim=True)

        return speech_tensor

    def chunk_audio(
        self,
        audio: torch.Tensor
    ) -> List[np.ndarray]:
        """Split audio into chunks for processing."""
        audio_np = audio.numpy().squeeze()
        chunk_length = int(self.audio_config.chunk_duration * 
                         self.audio_config.target_rate)
        
        chunks = []
        for i in range(0, len(audio_np), chunk_length):
            chunk = audio_np[i:i + chunk_length]
            if len(chunk) < chunk_length:
                chunk = np.pad(
                    chunk,
                    (0, chunk_length - len(chunk)),
                    mode='constant'
                )
            chunks.append(chunk)
        
        return chunks

    def transcribe_chunks(
        self,
        chunks: List[np.ndarray]
    ) -> str:
        """Transcribe audio chunks and combine results."""
        transcriptions = []
        
        for chunk in chunks:
            try:
                if self.whisper:
                    result = self.transcriber(
                        chunk,
                        generate_kwargs={"language": "swedish"}
                    )
                else:
                    result = self.transcriber(chunk)
                
                transcriptions.append(result['text'])
                self.logger.debug(f"Transcribed chunk: {result['text']}")
                
            except Exception as e:
                self.logger.error(f"Failed to transcribe chunk: {str(e)}")
                continue

        return ' '.join(transcriptions)

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess transcribed text and split into sentences."""
        # Apply preprocessing pipeline
        processed_text = self.text_preprocessor(text)
        
        # Split into sentences and filter
        sentences = [s.strip() for s in processed_text.split('.')
                    if len(s.strip().split()) >= 
                    self.preprocess_config.min_sentence_length]

        return sentences

    def process_file(self, filename: str) -> Optional[List[str]]:
        """Process a single audio file through the complete pipeline."""
        try:
            # Load audio
            speech_tensor, sampling_rate = self.load_audio(filename)
            
            # Process audio
            processed_audio = self.process_audio(speech_tensor, sampling_rate)
            
            # Chunk audio
            chunks = self.chunk_audio(processed_audio)
            
            # Transcribe
            transcription = self.transcribe_chunks(chunks)
            
            # Preprocess text
            sentences = self.preprocess_text(transcription)
            
            self.logger.info(f"Successfully processed file: {filename}")
            return sentences
            
        except Exception as e:
            self.logger.error(f"Failed to process file {filename}: {str(e)}")
            return None

    def process_all_files(self) -> pd.DataFrame:
        """
        Process all audio files and return results as a DataFrame.
        """
        data = []
        
        for filename in self.files:
            if not filename.endswith('.wav'):
                self.logger.info(f"Skipping non-WAV file: {filename}")
                continue
                
            sentences = self.process_file(filename)
            if sentences:
                self.transcriptions[filename] = sentences
                
                for i, sentence in enumerate(sentences, 1):
                    data.append({
                        'Filename': filename,
                        'Sentence_Number': i,
                        'Sentence': sentence,
                        'Audio_Block': f"A{len(data) // 2 + 1}",
                        'Timestamp': f"{i * self.audio_config.chunk_duration:.1f}s"
                    })

        # Create DataFrame
        self.processed_data = pd.DataFrame(data)
        if not self.processed_data.empty:
            self.processed_data.set_index(['Filename', 'Sentence_Number'], inplace=True)
        
        return self.processed_data

    def save_results(
        self,
        output_path: str,
        format: str = 'csv'
    ) -> None:
        """
        Save processing results to file.

        Args:
            output_path: Path to save the results
            format: Output format ('csv' or 'hdf')
        """
        if self.processed_data.empty:
            self.logger.warning("No data to save")
            return

        try:
            if format == 'csv':
                self.processed_data.to_csv(output_path)
            elif format == 'hdf':
                self.processed_data.to_hdf(
                    output_path,
                    key='transcription_data',
                    mode='w'
                )
            self.logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")