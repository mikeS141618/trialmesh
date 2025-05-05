# src/trialmesh/embeddings/__init__.py

from trialmesh.embeddings.base import BaseEmbeddingModel
from trialmesh.embeddings.factory import EmbeddingModelFactory

# Re-export key classes for cleaner imports
__all__ = ['BaseEmbeddingModel', 'EmbeddingModelFactory']