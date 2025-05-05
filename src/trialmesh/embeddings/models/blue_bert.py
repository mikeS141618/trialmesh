# src/trialmesh/embeddings/models/blue_bert.py

import logging
import torch
from typing import List
from transformers import AutoModel, AutoTokenizer
from trialmesh.embeddings.base import BaseEmbeddingModel


class BlueBERT(BaseEmbeddingModel):
    """Embedding model using bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12."""

    def _load_model(self):
        """Load BlueBERT model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            logging.info(f"Loaded BlueBERT model from {self.model_path}")
        except Exception as e:
            logging.error(f"Error loading BlueBERT model: {str(e)}")
            raise

    def _batch_encode(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts using BlueBERT model."""
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

        # Use the [CLS] token embedding
        embeddings = outputs.last_hidden_state[:, 0]

        # Normalize if requested
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings