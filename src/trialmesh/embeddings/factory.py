# src/trialmesh/embeddings/factory.py

import os
import logging
from typing import Any, Dict, Type

from trialmesh.embeddings.base import BaseEmbeddingModel
from trialmesh.embeddings.models import MODEL_REGISTRY


class EmbeddingModelFactory:
    """Factory for creating embedding models based on model type.

    This factory class provides a centralized way to instantiate different
    embedding models based on type or path, handling auto-detection and
    initialization details.

    The factory supports multiple embedding model types including:
    - E5 models (e5-large-v2)
    - BGE models (bge-large-v1.5)
    - SapBERT models
    - BioClinicalBERT models
    - BlueBERT models
    """

    @staticmethod
    def create_model(model_type: str = None, model_path: str = None, **kwargs) -> BaseEmbeddingModel:
        """Create an embedding model instance.

        This method handles:
        1. Auto-detecting model type from path if not specified
        2. Instantiating the appropriate model class
        3. Preparing the model for use

        Args:
            model_type: Type of model to create (e5-large-v2, bge-large-v1.5, etc.)
            model_path: Path to model directory
            **kwargs: Additional arguments to pass to model constructor

        Returns:
            An instance of the requested embedding model

        Raises:
            ValueError: If model_type is unknown or cannot be auto-detected
        """
        # Auto-detect model type from path if not specified
        if (model_type not in MODEL_REGISTRY or model_type is None) and model_path:
            path_lower = model_path.lower()
            for key, model_pattern in [
                ("e5-large-v2", "e5"),
                ("bge-large-v1.5", "bge"),
                ("sapbert", ["sapbert", "pubmedbert"]),
                ("bio-clinicalbert", "clinicalbert"),
                ("bluebert", "bluebert"),
            ]:
                patterns = [model_pattern] if isinstance(model_pattern, str) else model_pattern
                if any(pattern in path_lower for pattern in patterns):
                    model_type = key
                    logging.info(f"Auto-detected model type as {model_type} from path")
                    break

        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}. Available types: {list(MODEL_REGISTRY.keys())}")

        model_class = MODEL_REGISTRY[model_type]
        model = model_class(model_path=model_path, **kwargs)

        # Prepare the model (load weights, move to device)
        model.prepare_model()

        return model

    @staticmethod
    def get_available_models():
        """Get a list of available model types.

        Returns:
            List of model type identifiers supported by the factory
        """
        return list(MODEL_REGISTRY.keys())