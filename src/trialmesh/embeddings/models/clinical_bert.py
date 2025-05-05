# src/trialmesh/embeddings/models/clinical_bert.py

import logging
import torch
from typing import List
from transformers import AutoModel, AutoTokenizer
from trialmesh.embeddings.base import BaseEmbeddingModel


class SapBERT(BaseEmbeddingModel):
    """Embedding model using cambridgeltl/SapBERT-from-PubMedBERT-fulltext."""

    def _load_model(self):
        """Load SapBERT model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            logging.info(f"Loaded SapBERT model from {self.model_path}")
        except Exception as e:
            logging.error(f"Error loading SapBERT model: {str(e)}")
            raise

    def _batch_encode(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts using SapBERT model."""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        # Generate embeddings
        outputs = self.model(**inputs)

        # SapBERT uses the [CLS] token embedding for semantic similarity
        embeddings = outputs.last_hidden_state[:, 0]

        # Normalize if requested
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings


class BioClinicalBERT(BaseEmbeddingModel):
    """Embedding model using emilyalsentzer/Bio_ClinicalBERT."""

    def _load_model(self):
        """Load Bio_ClinicalBERT model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            logging.info(f"Loaded Bio_ClinicalBERT model from {self.model_path}")
        except Exception as e:
            logging.error(f"Error loading Bio_ClinicalBERT model: {str(e)}")
            raise

    def _batch_encode(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts using Bio_ClinicalBERT model."""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        # Generate embeddings
        outputs = self.model(**inputs)

        # Use mean pooling for better representation of clinical text
        embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])

        # Normalize if requested
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings