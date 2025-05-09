# src/trialmesh/embeddings/models/bge.py

import logging
import torch
from typing import List
from transformers import AutoModel, AutoTokenizer
from trialmesh.embeddings.base import BaseEmbeddingModel


class BGELargeV15(BaseEmbeddingModel):
    """Embedding model using BAAI/bge-large-en-v1.5.

    This class implements the BGE embedding model which is optimized for
    semantic search and retrieval tasks. It uses the [CLS] token output
    as the document embedding.

    Attributes:
        Inherits all attributes from BaseEmbeddingModel
    """

    def _load_model(self):
        """Load BGE model and tokenizer.

        This method loads the model and tokenizer from disk,
        handling any errors that occur during loading.

        Raises:
            Exception: If the model or tokenizer cannot be loaded
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            logging.info(f"Loaded BGE model from {self.model_path}")
        except Exception as e:
            logging.error(f"Error loading BGE model: {str(e)}")
            raise

    def _batch_encode(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts using BGE model.

        This method converts text inputs to embedding vectors using:
        1. Tokenization with padding and truncation
        2. Forward pass through the BGE model
        3. Extraction of [CLS] token embeddings
        4. Optional L2 normalization of embeddings

        Args:
            texts: List of texts to encode

        Returns:
            Tensor of embeddings, one per input text
        """
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

        # BGE typically uses the [CLS] token embedding
        embeddings = outputs.last_hidden_state[:, 0]

        # Normalize if requested (BGE models typically require normalization)
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings