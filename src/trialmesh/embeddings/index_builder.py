# src/trialmesh/embeddings/index_builder.py

import os
import time
import logging
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable


class FaissIndexBuilder:
    """Builder for FAISS indices to enable efficient similarity search.

    This class encapsulates the creation and management of FAISS indices
    for vector similarity search, supporting multiple index types and
    distance metrics.

    Attributes:
        index_type (str): Type of index ("flat", "ivf", "hnsw")
        dimension (int): Embedding dimension
        metric (str): Distance metric ("cosine", "l2", "ip")
        nlist (int): Number of centroids for IVF indices
        m (int): Number of connections per layer for HNSW indices
        ef_construction (int): Size of dynamic candidate list for HNSW
        index: The FAISS index object
        id_map (dict): Mapping between internal FAISS IDs and document IDs
        requires_training (bool): Whether the index requires training (set for IVF indices)
    """

    def __init__(
            self,
            index_type: str = "flat",
            dimension: int = None,
            metric: str = "cosine",
            nlist: int = 100,  # For IVF indices
            m: int = 32,  # For HNSW indices
            ef_construction: int = 200,  # For HNSW indices
    ):
        """Initialize the FAISS index builder.

        Args:
            index_type: Type of index to build ("flat", "ivf", "hnsw")
            dimension: Embedding dimension (can be inferred from data)
            metric: Distance metric ("cosine", "l2", "ip")
            nlist: Number of centroids for IVF indices
            m: Number of connections per layer for HNSW indices
            ef_construction: Size of the dynamic candidate list for HNSW
        """
        self.index_type = index_type.lower()
        self.dimension = dimension
        self.metric = metric.lower()
        self.nlist = nlist
        self.m = m
        self.ef_construction = ef_construction
        self.index = None
        self.id_map = {}  # Maps internal FAISS IDs to document IDs

        # Validate index type
        valid_types = ["flat", "ivf", "hnsw"]
        if self.index_type not in valid_types:
            raise ValueError(f"Invalid index type: {index_type}. Choose from: {valid_types}")

        # Validate metric
        valid_metrics = ["cosine", "l2", "ip"]
        if self.metric not in valid_metrics:
            raise ValueError(f"Invalid metric: {metric}. Choose from: {valid_metrics}")

    def _create_index(self, dimension: int) -> faiss.Index:
        """Create a FAISS index based on configured parameters.

        This method instantiates the appropriate FAISS index type with
        the specified parameters and dimension.

        Args:
            dimension: Dimension of the embedding vectors

        Returns:
            A FAISS index object configured according to specifications
        """
        # Store the dimension
        self.dimension = dimension

        # Configure the metric
        if self.metric == "cosine":
            metric_param = faiss.METRIC_INNER_PRODUCT
            # For cosine, vectors should be normalized
            normalize = True
        elif self.metric == "ip":
            metric_param = faiss.METRIC_INNER_PRODUCT
            normalize = False
        else:  # l2
            metric_param = faiss.METRIC_L2
            normalize = False

        # Create the appropriate index
        if self.index_type == "flat":
            index = faiss.IndexFlatL2(dimension) if self.metric == "l2" else faiss.IndexFlatIP(dimension)

        elif self.index_type == "ivf":
            # Create a quantizer
            quantizer = faiss.IndexFlatL2(dimension) if self.metric == "l2" else faiss.IndexFlatIP(dimension)

            # Create the IVF index
            index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist, metric_param)

            # IVF indices need to be trained
            self.requires_training = True

        elif self.index_type == "hnsw":
            # Create HNSW index
            index = faiss.IndexHNSWFlat(dimension, self.m, metric_param)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = 128  # Default search depth

        # Wrap with IDMap to maintain document IDs
        index = faiss.IndexIDMap(index)

        logging.info(f"Created {self.index_type} index with {dimension} dimensions using {self.metric} metric")
        return index

    def build_from_dict(self, embeddings: Dict[str, np.ndarray],
                        normalize: bool = None) -> None:
        """Build a FAISS index from a dictionary of embeddings.

        Args:
            embeddings: Dictionary mapping document IDs to embedding vectors
            normalize: Whether to normalize vectors (for cosine similarity)
                       If None, defaults to True for cosine metric, False otherwise

        Raises:
            ValueError: If embeddings dictionary is empty or contains invalid data
        """
        if not embeddings:
            raise ValueError("Empty embeddings dictionary provided")

        # Extract document IDs and vectors
        doc_ids = list(embeddings.keys())
        vectors = np.array([embeddings[doc_id] for doc_id in doc_ids], dtype=np.float32)

        # Build the index
        self.build_from_vectors(vectors, doc_ids, normalize)

    def build_from_vectors(self, vectors: np.ndarray, doc_ids: List[str],
                           normalize: bool = None) -> None:
        """Build a FAISS index from vectors and document IDs.

        This is the core index building method that:
        1. Normalizes vectors if needed
        2. Creates the appropriate index type
        3. Trains the index if required (for IVF)
        4. Adds vectors with their IDs

        Args:
            vectors: Matrix of embedding vectors (n_docs × dimension)
            doc_ids: List of document IDs corresponding to vectors
            normalize: Whether to normalize vectors (for cosine similarity)

        Raises:
            ValueError: If vectors array is empty or if dimensions don't match
        """
        if len(vectors) == 0:
            raise ValueError("Empty vectors array provided")

        if len(vectors) != len(doc_ids):
            raise ValueError(f"Number of vectors ({len(vectors)}) must match number of IDs ({len(doc_ids)})")

        # Infer dimension if not provided
        if self.dimension is None:
            self.dimension = vectors.shape[1]

        # Ensure vectors are float32
        vectors = vectors.astype(np.float32)

        # Normalize vectors if using cosine similarity
        if normalize is None:
            normalize = (self.metric == "cosine")

        if normalize:
            faiss.normalize_L2(vectors)

        # Create index if not already created
        if self.index is None:
            self.index = self._create_index(vectors.shape[1])

        # Train index if needed (e.g., for IVF indices)
        if self.index_type == "ivf" and not self.index.is_trained:
            logging.info(f"Training IVF index with {len(vectors)} vectors")
            self.index.train(vectors)

        # Convert string IDs to numeric IDs for FAISS
        numeric_ids = np.arange(len(doc_ids), dtype=np.int64)

        # Store the ID mapping
        self.id_map = {int(numeric_id): doc_id for numeric_id, doc_id in zip(numeric_ids, doc_ids)}

        # Add vectors to the index
        start_time = time.time()
        self.index.add_with_ids(vectors, numeric_ids)
        elapsed = time.time() - start_time

        logging.info(f"Built index with {len(vectors)} vectors in {elapsed:.2f} seconds")

    def build_from_file(self, embeddings_file: str, normalize: bool = None) -> None:
        """Build a FAISS index from a saved embeddings file.

        Args:
            embeddings_file: Path to .npy file containing embeddings dictionary
            normalize: Whether to normalize vectors (for cosine similarity)

        Raises:
            ValueError: If file format is incorrect
        """
        logging.info(f"Loading embeddings from {embeddings_file}")

        try:
            # Load embeddings dictionary
            embeddings = np.load(embeddings_file, allow_pickle=True).item()

            if not isinstance(embeddings, dict):
                raise ValueError(f"Embeddings file should contain a dictionary, got {type(embeddings)}")

            # Build index from the loaded dictionary
            self.build_from_dict(embeddings, normalize)

        except Exception as e:
            logging.error(f"Error loading embeddings file: {str(e)}")
            raise

    def save_index(self, index_path: str) -> None:
        """Save the FAISS index and ID mapping to disk.

        Args:
            index_path: Path to save the index

        Raises:
            ValueError: If no index exists to save
        """
        if self.index is None:
            raise ValueError("No index to save. Build an index first.")

        # Ensure directory exists
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        # Save the FAISS index
        faiss.write_index(self.index, index_path)

        # Save the ID mapping
        mapping_path = index_path + ".mapping.npy"
        np.save(mapping_path, self.id_map)

        logging.info(f"Saved index to {index_path} and ID mapping to {mapping_path}")

    @classmethod
    def load_index(cls, index_path: str) -> 'FaissIndexBuilder':
        """Load a FAISS index and ID mapping from disk.

        Args:
            index_path: Path to the saved index

        Returns:
            A FaissIndexBuilder instance with the loaded index
        """
        # Create an empty builder
        builder = cls()

        # Load the FAISS index
        builder.index = faiss.read_index(index_path)

        # Load the ID mapping
        mapping_path = index_path + ".mapping.npy"
        if os.path.exists(mapping_path):
            builder.id_map = np.load(mapping_path, allow_pickle=True).item()
        else:
            logging.warning(f"ID mapping file not found at {mapping_path}. Search will return numeric IDs.")
            builder.id_map = {}

        # Infer dimension and type from the loaded index
        builder.dimension = builder.index.d

        if isinstance(builder.index, faiss.IndexIDMap):
            base_index = faiss.downcast_index(builder.index.index)
            if isinstance(base_index, faiss.IndexHNSWFlat):
                builder.index_type = "hnsw"
            elif isinstance(base_index, faiss.IndexIVFFlat):
                builder.index_type = "ivf"
            else:
                builder.index_type = "flat"

        logging.info(f"Loaded {builder.index_type} index with {builder.dimension} dimensions from {index_path}")
        return builder