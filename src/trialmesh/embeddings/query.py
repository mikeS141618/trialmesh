# src/trialmesh/embeddings/query.py

import logging
import numpy as np
import faiss
from typing import Dict, List, Optional, Union, Tuple, Any

from trialmesh.embeddings.index_builder import FaissIndexBuilder

class SearchResult:
    """Container for FAISS search results.

    This class encapsulates the results of a similarity search query,
    including document IDs, similarity scores, and query information.

    Attributes:
        query_id: ID of the query document
        doc_ids: List of retrieved document IDs
        distances: List of distances/scores for retrieved documents
        original_query: Original query vector (optional)
    """

    def __init__(
            self,
            query_id: str,
            doc_ids: List[str],
            distances: List[float],
            original_query: Optional[np.ndarray] = None,
    ):
        """Initialize search result.

        Args:
            query_id: ID of the query document
            doc_ids: List of retrieved document IDs
            distances: List of distances/scores for retrieved documents
            original_query: Original query vector (optional)
        """
        self.query_id = query_id
        self.doc_ids = doc_ids
        self.distances = distances
        self.original_query = original_query

    def __len__(self):
        return len(self.doc_ids)

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary.

        Returns:
            Dictionary representation of search results
        """
        return {
            "query_id": self.query_id,
            "results": [
                {"doc_id": doc_id, "score": float(score)}
                for doc_id, score in zip(self.doc_ids, self.distances)
            ]
        }

    def __str__(self):
        return f"SearchResult(query_id={self.query_id}, matches={len(self.doc_ids)})"


class FaissSearcher:
    """Search interface for FAISS indices.

    This class provides methods for performing similarity searches
    against FAISS indices, with support for different query formats
    and result processing.

    Attributes:
        index_builder: FaissIndexBuilder with the index
        index: The FAISS index object
        dimension: Embedding dimension
        metric: Distance metric used
        id_map: Mapping between internal FAISS IDs and document IDs
    """

    def __init__(
            self,
            index_builder: Optional[FaissIndexBuilder] = None,
            index_path: Optional[str] = None,
    ):
        """Initialize the searcher with an index builder or path.

        Args:
            index_builder: FaissIndexBuilder with a built index
            index_path: Path to a saved FAISS index
        """
        if index_builder is not None:
            self.index_builder = index_builder
        elif index_path is not None:
            self.index_builder = FaissIndexBuilder.load_index(index_path)
        else:
            raise ValueError("Either index_builder or index_path must be provided")

        # Extract the index and configuration
        self.index = self.index_builder.index
        self.dimension = self.index_builder.dimension
        self.metric = self.index_builder.metric
        self.id_map = self.index_builder.id_map

        # Set search parameters
        if self.index_builder.index_type == "hnsw":
            # Get the HNSW index
            base_index = faiss.downcast_index(self.index.index)
            base_index.hnsw.efSearch = 128  # Can be adjusted for search depth
        elif self.index_builder.index_type == "ivf":
            # Get the IVF index
            base_index = faiss.downcast_index(self.index.index)
            base_index.nprobe = 10  # Number of clusters to search

        logging.info(f"Initialized FAISS searcher with {self.index_builder.index_type} index")

    def search(self, query_vector: np.ndarray, query_id: str = "query",
               k: int = 10, normalize: bool = None) -> SearchResult:
        """Search the index for vectors similar to the query vector.

        Args:
            query_vector: Query embedding vector
            query_id: ID for the query (for result tracking)
            k: Number of results to return
            normalize: Whether to normalize the query vector

        Returns:
            SearchResult object with matches

        Raises:
            ValueError: If query dimension doesn't match index dimension
        """
        # Ensure query is 2D and float32
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        query_vector = query_vector.astype(np.float32)

        # Check dimension
        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension ({query_vector.shape[1]}) doesn't match index dimension ({self.dimension})")

        # Normalize if using cosine similarity
        if normalize is None:
            normalize = (self.metric == "cosine")

        if normalize:
            faiss.normalize_L2(query_vector)

        # Search the index
        distances, indices = self.index.search(query_vector, k)

        # Convert to 1D arrays
        distances = distances[0]
        indices = indices[0]

        # Convert numeric IDs to document IDs using the mapping
        doc_ids = [self.id_map.get(int(idx), str(idx)) for idx in indices if idx != -1]

        # Filter out any -1 indices (not found)
        valid_indices = [i for i, idx in enumerate(indices) if idx != -1]
        distances = distances[valid_indices]

        # Adjust distances for metric type
        if self.metric == "cosine" or self.metric == "ip":
            # Convert to similarity score (higher is better)
            # For inner product, distances are actually similarities
            scores = distances
        else:
            # Convert L2 distances to similarity scores (higher is better)
            scores = 1.0 / (1.0 + distances)

        return SearchResult(query_id, doc_ids, scores, query_vector)

    def batch_search(self, query_vectors: np.ndarray, query_ids: List[str],
                     k: int = 10, normalize: bool = None) -> List[SearchResult]:
        """Search the index for multiple query vectors in batch.

        Args:
            query_vectors: Matrix of query vectors (n_queries × dimension)
            query_ids: List of query IDs corresponding to vectors
            k: Number of results to return per query
            normalize: Whether to normalize the query vectors

        Returns:
            List of SearchResult objects

        Raises:
            ValueError: If number of vectors doesn't match number of IDs
                       or if dimensions don't match
        """
        if len(query_vectors) != len(query_ids):
            raise ValueError(
                f"Number of query vectors ({len(query_vectors)}) must match number of query IDs ({len(query_ids)})")

        # Ensure queries are float32
        query_vectors = query_vectors.astype(np.float32)

        # Check dimension
        if query_vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension ({query_vectors.shape[1]}) doesn't match index dimension ({self.dimension})")

        # Normalize if using cosine similarity
        if normalize is None:
            normalize = (self.metric == "cosine")

        if normalize:
            faiss.normalize_L2(query_vectors)

        # Search the index
        distances, indices = self.index.search(query_vectors, k)

        # Process results for each query
        results = []
        for i, query_id in enumerate(query_ids):
            query_distances = distances[i]
            query_indices = indices[i]

            # Convert numeric IDs to document IDs using the mapping
            doc_ids = [self.id_map.get(int(idx), str(idx)) for idx in query_indices if idx != -1]

            # Filter out any -1 indices (not found)
            valid_indices = [j for j, idx in enumerate(query_indices) if idx != -1]
            query_distances = query_distances[valid_indices]

            # Adjust distances for metric type
            if self.metric == "cosine" or self.metric == "ip":
                scores = query_distances
            else:
                scores = 1.0 / (1.0 + query_distances)

            results.append(SearchResult(query_id, doc_ids, scores, query_vectors[i]))

        return results

    def search_by_id(self, query_id: str, embeddings: Dict[str, np.ndarray],
                     k: int = 10, normalize: bool = None) -> SearchResult:
        """Search using a document ID as the query.

        Args:
            query_id: ID of the query document
            embeddings: Dictionary mapping IDs to embedding vectors
            k: Number of results to return
            normalize: Whether to normalize the query vector

        Returns:
            SearchResult object with matches

        Raises:
            ValueError: If query_id not found in embeddings
        """
        if query_id not in embeddings:
            raise ValueError(f"Query ID {query_id} not found in embeddings dictionary")

        query_vector = embeddings[query_id]
        return self.search(query_vector, query_id, k, normalize)

    def batch_search_by_id(self, query_ids: List[str], embeddings: Dict[str, np.ndarray],
                           k: int = 10, normalize: bool = None) -> List[SearchResult]:
        """Search for multiple document IDs in batch.

        Args:
            query_ids: List of query document IDs
            embeddings: Dictionary mapping IDs to embedding vectors
            k: Number of results to return per query
            normalize: Whether to normalize the query vectors

        Returns:
            List of SearchResult objects

        Raises:
            ValueError: If any query_id not found in embeddings
        """
        # Check that all query IDs exist in embeddings
        missing_ids = [qid for qid in query_ids if qid not in embeddings]
        if missing_ids:
            raise ValueError(f"Query IDs not found in embeddings dictionary: {missing_ids}")

        # Collect query vectors
        query_vectors = np.array([embeddings[qid] for qid in query_ids], dtype=np.float32)

        return self.batch_search(query_vectors, query_ids, k, normalize)