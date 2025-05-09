# src/trialmesh/embeddings/base.py

import os
import torch
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
from abc import ABC, abstractmethod
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from trialmesh.embeddings.dataset import EmbeddingDataset
from torch.utils.data import DataLoader, DistributedSampler


class BaseEmbeddingModel(ABC):
    """Abstract base class for all embedding models.

    This class defines the interface for embedding models used in TrialMesh.
    Embedding models convert text into dense vector representations that can be
    compared for semantic similarity. The class supports distributed processing
    across multiple GPUs for efficient embedding generation at scale.

    Attributes:
        model_path: Path to the embedding model
        max_length: Maximum sequence length for tokenization
        batch_size: Batch size for processing
        normalize_embeddings: Whether to L2 normalize embeddings
        device: Device to run the model on
        model: The actual model instance (initialized in prepare_model)
        tokenizer: The tokenizer for the model
        is_distributed: Whether running in distributed mode
        local_rank: Current process rank in distributed setting
        world_size: Total number of processes in distributed setting
    """

    def __init__(self, model_path: str, max_length: int = 512,
                 batch_size: int = 32, device: str = None,
                 use_multi_gpu: bool = False, normalize_embeddings: bool = True,
                 local_rank: int = -1):
        """Initialize the embedding model.

        Args:
            model_path: Path to the model directory on disk
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for processing
            device: Device to run the model on (cuda:0, cuda:1, cpu)
            use_multi_gpu: Whether to use distributed processing across GPUs
            normalize_embeddings: Whether to L2 normalize embeddings (recommended for cosine similarity)
        """

        self.model_path = model_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.use_multi_gpu = use_multi_gpu

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize distributed setup if needed
        self.is_distributed = False
        self.local_rank = local_rank  # Use passed local_rank instead of env var
        self.world_size = 1

        if use_multi_gpu:
            if self.local_rank >= 0:  # If local_rank is provided directly
                self.is_distributed = True
                self.world_size = int(os.environ.get("WORLD_SIZE", "1"))

                # Initialize the distributed process group
                if not dist.is_initialized():
                    dist.init_process_group(backend="nccl")

                # Update device to local rank
                self.device = f"cuda:{self.local_rank}"
                torch.cuda.set_device(self.local_rank)

                logging.info(f"Distributed setup initialized: rank {self.local_rank}/{self.world_size}")
            elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
                self.local_rank = int(os.environ["RANK"])
                self.world_size = int(os.environ["WORLD_SIZE"])
                self.is_distributed = True

                # Initialize the distributed process group
                if not dist.is_initialized():
                    dist.init_process_group(backend="nccl")

                # Update device to local rank
                self.device = f"cuda:{self.local_rank}"
                torch.cuda.set_device(self.local_rank)

                logging.info(f"Distributed setup initialized: rank {self.local_rank}/{self.world_size}")
            else:
                logging.warning("Multi-GPU requested but environment variables RANK and WORLD_SIZE not set")

        # Model and tokenizer will be initialized in _load_model
        self.model = None
        self.tokenizer = None

    def prepare_model(self):
        """Load model and move to device.

        This method initializes the model and tokenizer, moves them to the
        appropriate device, and sets up distributed processing if needed.
        """
        # Load the model and tokenizer
        self._load_model()

        # Move model to device
        if self.model is not None:
            self.model.to(self.device)

            # Wrap model with DDP if distributed
            if self.is_distributed:
                self.model = DDP(self.model, device_ids=[self.local_rank])

            # Set to evaluation mode
            self.model.eval()

    @abstractmethod
    def _load_model(self):
        """Load model and tokenizer from disk.

        This abstract method must be implemented by subclasses to load
        their specific model and tokenizer implementation.
        """
        pass

    @abstractmethod
    def _batch_encode(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts to embeddings.

        This abstract method must be implemented by subclasses to define
        how text is converted to embeddings using their specific model.

        Args:
            texts: List of texts to encode

        Returns:
            Tensor of embeddings, one per input text
        """
        pass

    def encode(self, texts: List[str], ids: Optional[List[str]] = None,
               show_progress: bool = True) -> Dict[str, np.ndarray]:
        """Encode texts to embeddings.

        This method handles the full embedding generation process including:
        1. Setting up datasets and dataloaders
        2. Processing batches efficiently
        3. Handling distributed processing
        4. Gathering results across processes

        Args:
            texts: List of texts to encode
            ids: List of IDs corresponding to texts (uses indices if not provided)
            show_progress: Whether to show progress bar

        Returns:
            Dictionary mapping IDs to embedding vectors
        """
        # Initialize IDs if not provided
        if ids is None:
            ids = [str(i) for i in range(len(texts))]

        # Handle empty input
        if len(texts) == 0:
            return {}

        # Create dataset and dataloader
        dataset = EmbeddingDataset(texts, ids, self.max_length)

        if self.is_distributed:
            sampler = DistributedSampler(dataset, shuffle=False)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

        # Process batches
        embeddings_dict = {}
        disable_progress = not show_progress or (self.is_distributed and self.local_rank != 0)

        with torch.no_grad():
            for batch in tqdm(dataloader, disable=disable_progress, desc="Generating embeddings"):
                batch_texts = batch["text"]
                batch_ids = batch["id"]

                # Get embeddings for this batch
                batch_embeddings = self._batch_encode(batch_texts)

                # Store results
                for i, doc_id in enumerate(batch_ids):
                    embeddings_dict[doc_id] = batch_embeddings[i].cpu().numpy()

        # If distributed, gather results from all processes
        if self.is_distributed:
            all_embeddings = [None for _ in range(self.world_size)]
            dist.gather_object(embeddings_dict, all_embeddings if self.local_rank == 0 else None, dst=0)

            # Combine results on rank 0
            if self.local_rank == 0:
                combined_dict = {}
                for emb_dict in all_embeddings:
                    if emb_dict is not None:  # Some might be None
                        combined_dict.update(emb_dict)
                return combined_dict
            else:
                return {}  # Non-zero ranks return empty dict
        else:
            return embeddings_dict

    def encode_corpus(self, jsonl_path: str, output_path: str,
                      text_field: str = "summary", id_field: str = "_id",
                      batch_size: Optional[int] = None):
        """Encode an entire corpus from a JSONL file.

        This method streamlines the process of encoding a corpus by:
        1. Loading documents from a JSONL file
        2. Extracting the text field from each document
        3. Generating embeddings for all documents
        4. Saving the embeddings to a NumPy file

        Args:
            jsonl_path: Path to input JSONL file with texts
            output_path: Path to save embeddings as .npy file
            text_field: Field containing the text to encode
            id_field: Field containing the document ID
            batch_size: Optional override for batch size
        """
        import json
        from pathlib import Path

        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get document count first (only on rank 0)
        if not self.is_distributed or self.local_rank == 0:
            # Count lines in file
            doc_count = 0
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        doc_count += 1

            logging.info(f"Found {doc_count} documents to embed")
        else:
            doc_count = 0

        # Broadcast document count to all processes
        if self.is_distributed:
            count_tensor = torch.tensor(doc_count, device=self.device)
            dist.broadcast(count_tensor, src=0)
            doc_count = count_tensor.item()

            if doc_count == 0:
                logging.warning("No documents to process")
                return

        # Calculate each rank's portion
        if self.is_distributed:
            docs_per_rank = doc_count // self.world_size
            remainder = doc_count % self.world_size

            # Each rank will process a specific range of documents
            start_idx = self.local_rank * docs_per_rank + min(self.local_rank, remainder)
            end_idx = start_idx + docs_per_rank + (1 if self.local_rank < remainder else 0)

            logging.info(f"Rank {self.local_rank}: Processing documents {start_idx} to {end_idx - 1}")
        else:
            # Non-distributed mode - process all
            start_idx = 0
            end_idx = doc_count

        # Each rank loads and processes only its assigned portion
        texts = []
        ids = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue

                # Skip documents not assigned to this rank
                if i < start_idx:
                    continue
                if i >= end_idx:
                    break

                try:
                    doc = json.loads(line)
                    text = doc.get(text_field, "")
                    doc_id = doc.get(id_field, "")

                    if text and doc_id:
                        texts.append(text)
                        ids.append(doc_id)
                except json.JSONDecodeError:
                    logging.warning(f"Error parsing JSON line {i}: {line[:100]}...")

        logging.info(f"Rank {self.local_rank if self.is_distributed else 0}: Loaded {len(texts)} documents")

        # Generate embeddings for this rank's documents
        embeddings = {}
        if texts:
            # Process in batches
            batch_size = batch_size or self.batch_size

            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_ids = ids[i:i + batch_size]

                    # Get embeddings for this batch
                    batch_embeddings = self._batch_encode(batch_texts)

                    # Store results
                    for j, doc_id in enumerate(batch_ids):
                        embeddings[doc_id] = batch_embeddings[j].cpu().numpy()

            logging.info(
                f"Rank {self.local_rank if self.is_distributed else 0}: Generated {len(embeddings)} embeddings")

        # In distributed mode, gather all embeddings to rank 0
        if self.is_distributed:
            # We need to gather information about how many embeddings each rank has
            num_embeddings = torch.tensor(len(embeddings), device=self.device)
            all_num_embeddings = [torch.zeros_like(num_embeddings) for _ in range(self.world_size)]
            dist.all_gather(all_num_embeddings, num_embeddings)

            # Use gather_object to collect all embedding dictionaries
            all_embeddings = [None] * self.world_size
            dist.gather_object(embeddings, all_embeddings if self.local_rank == 0 else None, dst=0)

            # Only rank 0 needs to save
            if self.local_rank == 0:
                # Combine all dictionaries
                combined_embeddings = {}
                for rank, rank_embeddings in enumerate(all_embeddings):
                    if rank_embeddings is not None:
                        logging.info(f"Rank 0: Combining {len(rank_embeddings)} embeddings from rank {rank}")
                        combined_embeddings.update(rank_embeddings)

                # Verify we got all expected embeddings
                total_expected = sum(n.item() for n in all_num_embeddings)
                logging.info(f"Rank 0: Expected {total_expected} embeddings, got {len(combined_embeddings)}")

                # Save the combined embeddings
                np.save(output_path, combined_embeddings)
                logging.info(f"Rank 0: Saved {len(combined_embeddings)} embeddings to {output_path}")
        else:
            # Non-distributed mode - save directly
            np.save(output_path, embeddings)
            logging.info(f"Saved {len(embeddings)} embeddings to {output_path}")

    def _mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling to create a single embedding vector.

        This utility method creates sentence embeddings by averaging token
        embeddings, weighted by the attention mask.

        Args:
            token_embeddings: Token-level embeddings from model
            attention_mask: Attention mask indicating valid tokens

        Returns:
            Pooled embeddings, one vector per input sequence
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)