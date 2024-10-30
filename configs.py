
from dataclasses import dataclass
from typing import Optional

@dataclass
class AudioConfig:
    """Configuration for audio processing parameters."""
    lowpass_freq: int = 300
    highpass_freq: int = 2000
    target_rate: int = 16000
    chunk_duration: int = 6

@dataclass
class PreprocessConfig:
    """Configuration for text preprocessing steps."""
    remove_punctuation: bool = False
    normalize_chars: bool = True
    max_repeating_chars: Optional[int] = None
    min_sentence_length: int = 5
    join_sentences: bool = True