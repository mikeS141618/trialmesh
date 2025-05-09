# src/trialmesh/embeddings/models/e5.py

import logging
import torch
from typing import List
from transformers import AutoModel, AutoTokenizer
from trialmesh.embeddings.base import BaseEmbeddingModel


class E5LargeV2(BaseEmbeddingModel):
    """Embedding model using intfloat/e5-large-v2.

    This class implements the E5 embedding model which is optimized for
    semantic search and retrieval tasks. It uses mean pooling of the
    last hidden state as the document embedding.

    Note: E5 requires "passage: " prefix for document encoding.

    Attributes:
        Inherits all attributes from BaseEmbeddingModel
    """

    def _load_model(self):
        """Load E5 model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            logging.info(f"Loaded E5 model from {self.model_path}")
        except Exception as e:
            logging.error(f"Error loading E5 model: {str(e)}")
            raise

    def _batch_encode(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts using E5 model.

        This method converts text inputs to embedding vectors using:
        1. Adding the required "passage: " prefix to each text
        2. Tokenization with padding and truncation
        3. Forward pass through the E5 model
        4. Mean pooling of token embeddings weighted by attention mask
        5. Optional L2 normalization of embeddings

        Args:
            texts: List of texts to encode

        Returns:
            Tensor of embeddings, one per input text
        """
        # E5 requires "query: " or "passage: " prefix
        # Use "passage: " for document encoding
        prefixed_texts = [f"passage: {text}" for text in texts]

        # Tokenize
        inputs = self.tokenizer(
            prefixed_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        # Generate embeddings
        outputs = self.model(**inputs)

        # E5 uses mean pooling of last hidden state
        embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])

        # Normalize if requested
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings