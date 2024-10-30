import os
import numpy as np
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Optional, Tuple, Union
import logging

class TextFeatures:
    """
    Enhanced class for text feature extraction, using SentenceTransformer for embeddings
    and similarity computation, particularly optimized for Swedish text.
    """
    def __init__(
        self,
        translation_model: str = "Helsinki-NLP/opus-mt-sv-en",
        emotion_model: str = "j-hartmann/emotion-english-distilroberta-base",
        similarity_model: str = "KBLab/sentence-bert-swedish-cased",
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        batch_size: int = 32
    ):
        """
        Initialize the text feature extraction class.

        Args:
            translation_model: Model for Swedish to English translation
            emotion_model: Model for emotion classification
            similarity_model: SentenceTransformer model name
            device: Computing device ('cpu', 'cuda', or 'mps')
            batch_size: Batch size for processing
        """
        self.batch_size = batch_size
        self.device = self._get_device(device)
        
        # Initialize models
        self.translator = pipeline(
            "translation_sv_to_en",
            model=translation_model,
            device=self.device
        )
        
        self.classifier = pipeline(
            "text-classification",
            model=emotion_model,
            device=self.device
        )
        
        # Initialize sentence transformer
        self.sentence_transformer = SentenceTransformer(similarity_model)
        self.sentence_transformer.to(device)
        
        # Initialize storage
        self.embeddings_cache = {}
        self.processed_data = pd.DataFrame()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _get_device(self, device: str) -> int:
        """Convert device string to pipeline-compatible device ID."""
        if device == 'cpu':
            return -1
        elif device == 'cuda':
            return 0
        elif device == 'mps':
            return 0
        else:
            raise ValueError(f"Unsupported device: {device}")

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate a batch of texts from Swedish to English."""
        translations = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                results = self.translator(batch)
                translations.extend([r['translation_text'] for r in results])
            except Exception as e:
                self.logger.error(f"Translation error in batch {i}: {str(e)}")
                translations.extend([""] * len(batch))
        return translations

    def get_sentence_embeddings(self, sentences: List[str]) -> torch.Tensor:
        """
        Generate embeddings for sentences using SentenceTransformer.
        Implements caching for efficiency.
        """
        # First check cache for all sentences
        cached_embeddings = []
        sentences_to_encode = []
        indices_to_insert = []

        for i, sentence in enumerate(sentences):
            if sentence in self.embeddings_cache:
                cached_embeddings.append(self.embeddings_cache[sentence])
            else:
                sentences_to_encode.append(sentence)
                indices_to_insert.append(i)

        # Encode new sentences if any
        if sentences_to_encode:
            new_embeddings = self.sentence_transformer.encode(
                sentences_to_encode,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            # Cache new embeddings
            for sent, emb in zip(sentences_to_encode, new_embeddings):
                self.embeddings_cache[sent] = emb

        # Combine cached and new embeddings in correct order
        all_embeddings = torch.zeros(
            (len(sentences), cached_embeddings[0].shape[0] if cached_embeddings else new_embeddings.shape[1]),
            device=self.device
        )
        
        # Place cached embeddings
        for i, emb in enumerate(cached_embeddings):
            all_embeddings[i] = emb
            
        # Place new embeddings
        if sentences_to_encode:
            for i, idx in enumerate(indices_to_insert):
                all_embeddings[idx] = new_embeddings[i]

        return all_embeddings

    def compute_similarities(
        self,
        sentences: List[str],
        window_size: int = 3
    ) -> Tuple[torch.Tensor, pd.DataFrame]:
        """
        Compute similarities between sentences within a sliding window.
        
        Args:
            sentences: List of sentences to compare
            window_size: Size of the sliding window
        """
        # Get embeddings and normalize
        embeddings = self.get_sentence_embeddings(sentences)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        # Compute similarity matrix
        similarity_matrix = embeddings @ embeddings.t()
        
        # Compute windowed similarities
        n_sentences = len(sentences)
        window_similarities = []
        
        for i in range(n_sentences):
            start = max(0, i - window_size)
            end = min(n_sentences, i + window_size + 1)
            
            # Extract local similarities
            local_similarities = similarity_matrix[i, start:end].cpu().numpy()
            avg_similarity = np.mean(local_similarities)
            # Exclude self-similarity (1.0) when computing max
            local_similarities[local_similarities > 0.9999] = -1  # Handle numerical precision
            max_similarity = np.max(local_similarities)
            
            window_similarities.append({
                'sentence_idx': i,
                'sentence': sentences[i],
                'avg_similarity': float(avg_similarity),
                'max_similarity': float(max_similarity),
                'window_start': start,
                'window_end': end-1
            })
        
        return similarity_matrix.cpu(), pd.DataFrame(window_similarities)

    def classify_emotions(
        self,
        sentences: List[str]
    ) -> List[Dict[str, Union[str, float]]]:
        """Classify emotions in sentences using batched processing."""
        classifications = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i + self.batch_size]
            try:
                results = self.classifier(batch)
                classifications.extend(results)
            except Exception as e:
                self.logger.error(f"Classification error in batch {i}: {str(e)}")
                classifications.extend([{'label': 'error', 'score': 0.0}] * len(batch))
        return classifications

    def process_transcriptions(
        self,
        transcription_df: pd.DataFrame,
        window_size: int = 3,
        min_sentence_length: int = 10,
        translate: bool = True
    ) -> pd.DataFrame:
        """
        Process transcriptions from the PreprocTranscribeAudio output.
        
        Args:
            transcription_df: DataFrame from PreprocTranscribeAudio
            window_size: Window size for similarity computation
            min_sentence_length: Minimum sentence length to process
            translate: Whether to translate sentences to English for emotion classification
        """
        # Filter sentences by length
        mask = transcription_df['Sentence'].str.split().str.len() >= min_sentence_length
        filtered_df = transcription_df[mask].copy()
        
        if filtered_df.empty:
            self.logger.warning("No sentences meet the minimum length requirement")
            return pd.DataFrame()

        # Compute similarities on original Swedish text
        similarity_matrix, similarity_df = self.compute_similarities(
            filtered_df['Sentence'].tolist(),
            window_size
        )

        if translate:
            # Translate for emotion classification
            translated_sentences = self.translate_batch(filtered_df['Sentence'].tolist())
            filtered_df['Translated'] = translated_sentences
            
            # Classify emotions on translated text
            emotion_results = self.classify_emotions(translated_sentences)
            filtered_df['emotion'] = [r['label'] for r in emotion_results]
            filtered_df['emotion_score'] = [r['score'] for r in emotion_results]
        
        # Add similarity metrics
        filtered_df['avg_similarity'] = similarity_df['avg_similarity']
        filtered_df['max_similarity'] = similarity_df['max_similarity']
        
        # Maintain original index structure
        result_df = filtered_df.set_index(transcription_df.index.names)
        self.processed_data = result_df
        
        return result_df

    def save_results(self, output_path: str, include_similarities: bool = True):
        """
        Save processed results to file.
        
        Args:
            output_path: Path to save the results
            include_similarities: Whether to include similarity metrics
        """
        if not self.processed_data.empty:
            if include_similarities:
                self.processed_data.to_csv(output_path)
            else:
                # Save without similarity columns
                cols_to_save = [c for c in self.processed_data.columns 
                              if 'similarity' not in c.lower()]
                self.processed_data[cols_to_save].to_csv(output_path)
            self.logger.info(f"Results saved to {output_path}")
        else:
            self.logger.warning("No processed data available to save")