# Project README

```
# TrialMesh

**TrialMesh** is an open-source, modular clinical trial matching system leveraging large language models (LLaMA via vLLM) and vector search (FAISS) to semantically match cancer patients to relevant clinical trials. The system emphasizes clinical reasoning, structured decision-making, and transparent, explainable logic.

## Overview

TrialMesh addresses the challenge of matching patients to appropriate clinical trials by transforming unstructured medical texts into structured, comparable formats. Using advanced language models, it extracts clinically relevant information from both patient records and trial protocols, enabling precise semantic matching that goes beyond simple keyword-based approaches.

## Features

- **Patient Summarization:** Extracts structured clinical data from unstructured patient records using LLaMA 3.3.
- **Trial Summarization:** Converts complex trial eligibility criteria into structured, machine-readable formats.
- **Semantic Matching:** Uses biomedical embedding models and FAISS for efficient similarity-based retrieval.
- **Multi-stage Filtering:** Applies rule-based pre-filtering followed by LLM-based clinical reasoning.
- **Robust Caching:** Hash-based caching system preserves prompts, tokens, and responses for reproducibility.
- **Explainable Decisions:** All matching decisions include transparent clinical reasoning and justifications.
- **Multiple Embedding Models:** Support for specialized biomedical embedding models including SapBERT, BioClinicalBERT, E5, BGE, and BlueBERT.
- **Distributed Processing:** Multi-GPU support for accelerated embedding generation.
- **Advanced Similarity Search:** FAISS indices for efficient retrieval with flat, IVF, and HNSW index types.
- **Comprehensive Evaluation:** Tools for evaluating retrieval performance against gold standard data.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/trialmesh.git
cd trialmesh

# Install the package in development mode
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

## Command-Line Tools

TrialMesh provides several command-line tools for data processing, trial matching, and evaluation:

### Data Acquisition

```bash
# Download the SIGIR2016 clinical trials dataset
trialmesh-download-sigir2016 --data-dir ./data

# Process XML trial documents into structured JSONL
trialmesh-process-sigir2016 --data-dir ./data --log-level INFO

# Download embedding and LLM models
trialmesh-download-models --embeddings  # Download just embedding models
trialmesh-download-models --llms        # Download just LLM models
trialmesh-download-models --all         # Download all models
trialmesh-download-models --list        # List available models
```

### LLM Summarization

```bash
# Generate LLM summaries for patients and trials
trialmesh-summarize --model-path /path/to/llama-model \
  --data-dir ./data \
  --dataset sigir2016/processed_cut \
  --cache-dir ./cache/llm_responses
```

### Embedding Generation

```bash
# Generate embeddings using SapBERT
trialmesh-embed \
  --model-path /path/to/SapBERT \
  --batch-size 32 \
  --normalize \
  --data-dir ./data \
  --dataset sigir2016/summaries

# Multi-GPU embedding with BGE
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
  -m trialmesh.embeddings.run_embedding \
  --model-path /path/to/BGE \
  --multi-gpu \
  --batch-size 64 \
  --normalize
```

### FAISS Index Building and Search

```bash
# Build a FAISS HNSW index (fast search)
trialmesh-index build \
  --embeddings ./data/sigir2016/summaries_embeddings/SapBERT/trial_embeddings.npy \
  --output ./data/sigir2016/indices/SapBERT_trials_hnsw.index \
  --index-type hnsw \
  --m 64

# Search for trials matching patients
trialmesh-index search \
  --index ./data/sigir2016/indices/SapBERT_trials_hnsw.index \
  --queries ./data/sigir2016/summaries_embeddings/SapBERT/patient_embeddings.npy \
  --output ./data/sigir2016/results/SapBERT_search_results.json \
  --k 100
```

### Retrieval Pipeline

```bash
# Run the complete retrieval pipeline (embedding, indexing, search) for all models
trialmesh-retrieval

# Run with specific models
trialmesh-retrieval --models SapBERT bge-large-en-v1.5

# Run with HNSW index type
trialmesh-retrieval --index-type hnsw --m-value 96 --ef-construction 300

# Skip certain stages
trialmesh-retrieval --skip-embeddings --skip-indexing
```

### Evaluation

```bash
# Evaluate all model search results
trialmesh-evaluate

# Evaluate specific models with visualization
trialmesh-evaluate --models SapBERT bge-large-en-v1.5 --visualize

# Evaluate only models using HNSW indices and save results
trialmesh-evaluate --index-type hnsw --output-file ./evaluation/hnsw_results.csv
```

### Maintenance

```bash
# Clean cached responses and outputs (with options for different types)
trialmesh-clean --clean-all

# Clean only embeddings for a specific model
trialmesh-clean --clean-embeddings --model-name SapBERT

# Preview what would be cleaned without removing anything
trialmesh-clean --clean-all --dry-run

# Generate code documentation
trialmesh-codemd
```

## Project Structure

The project follows a modern `src` layout:

```
trialmesh/
├── src/
│   └── trialmesh/         # Main package code
│       ├── cli/           # Command-line interfaces
│       │   ├── download_models.py       # Model download utility
│       │   └── run_retrieval_pipeline.py # Retrieval pipeline orchestration
│       ├── config/        # Configuration settings
│       ├── data/          # Data acquisition and processing
│       ├── embeddings/    # Vector embeddings and FAISS
│       │   ├── base.py    # Base embedding model class
│       │   ├── dataset.py # Dataset handling  
│       │   ├── factory.py # Model factory
│       │   ├── models/    # Embedding model implementations
│       │   ├── query.py   # FAISS search interface
│       │   └── index_builder.py  # FAISS index creation
│       ├── evaluation/    # Evaluation tools
│       │   └── evaluate_results.py  # Results evaluation 
│       ├── llm/           # LLM runners and processors
│       ├── match/         # Matching logic and pipeline
│       └── utils/         # Prompt registry and utilities
├── data/                  # Data storage
│   └── sigir2016/         # SIGIR clinical trials dataset
│       ├── processed/     # Processed trials and queries
│       ├── summaries/     # LLM-generated summaries
│       ├── summaries_embeddings/  # Vector embeddings
│       ├── indices/       # FAISS indices
│       └── results/       # Search and match results
├── cache/                 # Cache for LLM responses
├── tests/                 # Test suite
├── docs/                  # Documentation
└── notebooks/             # Analysis notebooks
```

## Data Processing Pipeline

TrialMesh processes clinical trial data through several stages:

1. **Data Acquisition:** Download and extract trial documents and patient queries
2. **Structured Extraction:** Parse XML into structured JSON with clinical fields
3. **LLM Summarization:** Create both detailed and condensed summaries using LLaMA
4. **Vector Embedding:** Generate semantic embeddings for efficient retrieval
5. **Index Building:** Create optimized FAISS indices for fast similarity search
6. **Matching:** Apply multi-stage filtering from vector retrieval to detailed LLM scoring
7. **Evaluation:** Assess retrieval performance against gold standard relevance judgments

## Supported Embedding Models

TrialMesh supports multiple embedding models optimized for different aspects of clinical text:

- **E5-large-v2**: General-purpose embedding model with 1024 dimensions
- **BGE-large-en-v1.5**: Powerful similarity embedding model with 1024 dimensions
- **SapBERT**: Specialized biomedical model trained on UMLS medical concepts
- **BioClinicalBERT**: Clinical BERT model trained on medical records
- **BlueBERT**: Model trained on PubMed abstracts and MIMIC clinical notes

All models can be loaded from local disk with full multi-GPU support for distributed processing.

## FAISS Index Types

TrialMesh supports multiple FAISS index types for different performance profiles:

- **Flat**: Exact search with no compression (slowest search, highest accuracy)
- **HNSW**: Hierarchical navigable small worlds (very fast search with high accuracy)
- **IVF**: Inverted file index (balanced speed and accuracy)

## Evaluation Capabilities

TrialMesh includes comprehensive evaluation tools for assessing retrieval performance:

- **Gold Standard Comparison**: Compare results against expert-annotated relevance judgments
- **Multiple Relevance Levels**: Separate evaluation for different relevance scores
- **Performance Metrics**: Calculate found/missing items and percentages
- **Visualizations**: Generate comparative plots of model performance
- **Reporting**: Export detailed performance statistics to CSV

## Current Status

The project has implemented:
- Complete data acquisition and processing pipeline
- LLM integration with vLLM for trial and patient summarization
- Structured and condensed summaries optimized for downstream tasks
- Robust caching and prompt management system
- Multi-model embedding generation with GPU acceleration
- FAISS indexing and similarity search
- Automated retrieval pipeline for multiple models
- Performance evaluation against gold standard data

Under development:
- Multi-stage matching pipeline
- Detailed clinical reasoning components

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Disclaimer

**TrialMesh is provided as a research and decision-support tool.**

- **No Warranty:** This software is provided "as is", without warranty of any kind, express or implied. The authors and contributors assume no responsibility for any clinical decisions, outcomes, or damages resulting from its use.
- **Professional Oversight Required:** If used in a clinical or patient-facing context, all outputs and recommendations must be independently reviewed and verified by qualified healthcare professionals. This software is not a substitute for professional medical judgment.
- **Use at Your Own Risk:** Users are solely responsible for ensuring that the software is used in compliance with all applicable laws, regulations, and institutional policies, and for validating its outputs in their specific context.
- **Data Privacy:** Users are responsible for ensuring compliance with all applicable data privacy and security regulations when handling patient data.


## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.


## Acknowledgments

- [SIGIR2016 Clinical Trials Collection](https://data.csiro.au/collection/csiro:17152)
- [vLLM Project](https://github.com/vllm-project/vllm)
- [FAISS Library](https://github.com/facebookresearch/faiss)
- [LLaMA Language Models](https://ai.meta.com/llama/)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [SapBERT](https://github.com/cambridgeltl/sapbert)
- [BioClinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT)

```

# pyproject.toml

```toml
# trialmesh/pyproject.toml
# For Poetry or modern build

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "trialmesh"
version = "1.0.0"
description = "A modular clinical trial matching system using LLMs and vector search"
readme = "README.md"
requires-python = ">=3.12,<3.13"
license = {text = "Apache-2.0"}
authors = [
    {name = "mikeS141618"}
]
keywords = ["clinical-trials", "healthcare", "llm", "vector-search", "oncology"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Healthcare Industry",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Medical Science Apps.",
]
dependencies = [
    "vllm==0.8.5",
    "faiss-cpu==1.11.0",
    "matplotlib==3.10.1",
    "pandas==2.2.3",
    "seaborn==0.13.2"
]

[project.optional-dependencies]
dev = [
    "pytest==8.3.3"
]

[project.urls]
"Homepage" = "https://github.com/mikeS141618/trialmesh"
"Bug Tracker" = "https://github.com/mikeS141618/trialmesh/issues"
"Documentation" = "https://github.com/mikeS141618/trialmesh#readme"

[project.scripts]
trialmesh-download-sigir2016 = "trialmesh.fetchers.pull_sigir2016:cli_main"
trialmesh-process-sigir2016 = "trialmesh.fetchers.processxml:cli_main"
trialmesh-summarize = "trialmesh.llm.summarizers:cli_main"
trialmesh-clean = "trialmesh.utils.clean_run:cli_main"
trialmesh-codemd = "trialmesh.utils.codemd:cli_main"
trialmesh-embed = "trialmesh.embeddings.run_embedding:cli_main"
trialmesh-index = "trialmesh.embeddings.build_index:cli_main"
trialmesh-retrieval = "trialmesh.cli.run_retrieval_pipeline:cli_main"
trialmesh-download-models = "trialmesh.cli.download_models:cli_main"
trialmesh-evaluate = "trialmesh.evaluation.evaluate_results:cli_main"
trialmesh-match = "trialmesh.cli.run_matcher:cli_main"


[tool.hatch.build.targets.wheel]
packages = ["src/trialmesh"]

[tool.black]
line-length = 100
target-version = ["py312"]

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
```

# src Directory Tree

```
src
trialmesh
    __init__.py
    embeddings
        __init__.py
        base.py
        build_index.py
        dataset.py
        factory.py
        index_builder.py
        query.py
        run_embedding.py
        models
            __init__.py
            bge.py
            blue_bert.py
            clinical_bert.py
            e5.py
            __pycache__
                __init__.cpython-312.pyc
                bge.cpython-312.pyc
                blue_bert.cpython-312.pyc
                clinical_bert.cpython-312.pyc
                e5.cpython-312.pyc
        __pycache__
            __init__.cpython-312.pyc
            base.cpython-312.pyc
            build_index.cpython-312.pyc
            dataset.cpython-312.pyc
            factory.cpython-312.pyc
            index_builder.cpython-312.pyc
            query.cpython-312.pyc
            run_embedding.cpython-312.pyc
    llm
        __init__.py
        llama_runner.py
        prompt_runner.py
        summarizers.py
        __pycache__
            __init__.cpython-312.pyc
            llama_runner.cpython-312.pyc
            prompt_runner.cpython-312.pyc
            summarizers.cpython-312.pyc
    match
        __init__.py
        matcher.py
        __pycache__
            __init__.cpython-312.pyc
            matcher.cpython-312.pyc
    utils
        __init__.py
        clean_run.py
        codemd.py
        prompt_registry.py
        __pycache__
            __init__.cpython-312.pyc
            clean_run.cpython-312.pyc
            codemd.cpython-312.pyc
            prompt_registry.cpython-312.pyc
    cli
        __init__.py
        download_models.py
        run_matcher.py
        run_retrieval_pipeline.py
        __pycache__
            __init__.cpython-312.pyc
            run_matcher.cpython-312.pyc
            run_retrieval_pipeline.cpython-312.pyc
    __pycache__
        __init__.cpython-312.pyc
    evaluation
        __init__.py
        evaluate_results.py
        __pycache__
            __init__.cpython-312.pyc
            evaluate_results.cpython-312.pyc
    fetchers
        __init__.py
        processxml.py
        pull_sigir2016.py
```

# Python Source Files

## src/trialmesh/__init__.py

```python
# src/trialmesh/__init__.py
```

## src/trialmesh/embeddings/__init__.py

```python
# src/trialmesh/embeddings/__init__.py

from trialmesh.embeddings.base import BaseEmbeddingModel
from trialmesh.embeddings.factory import EmbeddingModelFactory

# Re-export key classes for cleaner imports
__all__ = ['BaseEmbeddingModel', 'EmbeddingModelFactory']
```

## src/trialmesh/embeddings/base.py

```python
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
    """Abstract base class for all embedding models."""

    def __init__(
            self,
            model_path: str,
            max_length: int = 512,
            batch_size: int = 32,
            device: str = None,
            use_multi_gpu: bool = False,
            normalize_embeddings: bool = True,
    ):
        """Initialize the embedding model.

        Args:
            model_path: Path to the model directory on disk
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for processing
            device: Device to run the model on (cuda:0, cuda:1, cpu)
            use_multi_gpu: Whether to use distributed processing across GPUs
            normalize_embeddings: Whether to L2 normalize embeddings
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
        self.local_rank = -1
        self.world_size = 1

        if use_multi_gpu:
            if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
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
        """Load model and move to device."""
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
        """Load model and tokenizer from disk. Implemented by subclasses."""
        pass

    @abstractmethod
    def _batch_encode(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts to embeddings. Implemented by subclasses."""
        pass

    def encode(
            self,
            texts: List[str],
            ids: Optional[List[str]] = None,
            show_progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            ids: List of IDs corresponding to texts (uses indices if not provided)
            show_progress: Whether to show progress bar

        Returns:
            Dictionary mapping IDs to embeddings
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

    def encode_corpus(
            self,
            jsonl_path: str,
            output_path: str,
            text_field: str = "summary",
            id_field: str = "_id",
            batch_size: Optional[int] = None,
    ):
        """Encode an entire corpus from a JSONL file.

        Args:
            jsonl_path: Path to input JSONL file with texts
            output_path: Path to save embeddings as .npy files
            text_field: Field containing the text to encode
            id_field: Field containing the document ID
            batch_size: Optional override for batch size
        """
        import json
        from pathlib import Path

        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load documents with proper distributed handling
        texts = []
        ids = []

        if not self.is_distributed or self.local_rank == 0:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        doc = json.loads(line)
                        text = doc.get(text_field, "")
                        doc_id = doc.get(id_field, "")

                        if text and doc_id:
                            texts.append(text)
                            ids.append(doc_id)
                    except json.JSONDecodeError:
                        logging.warning(f"Error parsing JSON line: {line[:100]}...")

            logging.info(f"Loaded {len(texts)} documents for embedding")

        # Broadcast document count in distributed setting
        if self.is_distributed:
            if self.local_rank == 0:
                doc_count = torch.tensor(len(texts), device=self.device)
            else:
                doc_count = torch.tensor(0, device=self.device)

            dist.broadcast(doc_count, src=0)

            # Skip further processing if no documents
            if doc_count.item() == 0:
                logging.warning("No documents to process")
                return

        # Generate embeddings
        embeddings = self.encode(texts, ids)

        # Save embeddings
        if not self.is_distributed or self.local_rank == 0:
            logging.info(f"Saving {len(embeddings)} embeddings to {output_path}")
            np.save(output_path, embeddings)

    def _mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling to create a single embedding vector.

        This is a utility method shared by several model implementations.
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
```

## src/trialmesh/embeddings/build_index.py

```python
# src/trialmesh/embeddings/build_index.py

import argparse
import logging
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any

from trialmesh.embeddings.index_builder import FaissIndexBuilder
from trialmesh.embeddings.query import FaissSearcher


def setup_logging(log_level: str = "INFO"):
    """Configure logging."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build and search FAISS indices")

    # Common arguments
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")

    # Create subparsers for build and search commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build a FAISS index")

    # Required arguments for build
    build_parser.add_argument("--embeddings", type=str, required=True,
                              help="Path to patient or trial embeddings .npy file")
    build_parser.add_argument("--output", type=str, required=True,
                              help="Path to save the FAISS index")

    # Optional arguments for build
    build_parser.add_argument("--index-type", type=str, default="flat",
                              choices=["flat", "ivf", "hnsw"],
                              help="Type of FAISS index to build")
    build_parser.add_argument("--metric", type=str, default="cosine",
                              choices=["cosine", "l2", "ip"],
                              help="Distance metric to use")
    build_parser.add_argument("--nlist", type=int, default=100,
                              help="Number of clusters for IVF index")
    build_parser.add_argument("--m", type=int, default=32,
                              help="Number of connections per layer for HNSW index")
    build_parser.add_argument("--ef-construction", type=int, default=200,
                              help="Size of dynamic candidate list for HNSW index")
    build_parser.add_argument("--normalize", action="store_true",
                              help="Normalize vectors (implied for cosine metric)")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search a FAISS index")

    # Required arguments for search
    search_parser.add_argument("--index", type=str, required=True,
                               help="Path to the FAISS index")
    search_parser.add_argument("--queries", type=str, required=True,
                               help="Path to query embeddings .npy file")
    search_parser.add_argument("--output", type=str, required=True,
                               help="Path to save search results")

    # Optional arguments for search
    search_parser.add_argument("--k", type=int, default=100,
                               help="Number of results to return per query")
    search_parser.add_argument("--query-ids", type=str, default=None, nargs="+",
                               help="Specific query IDs to search for (default: all)")
    search_parser.add_argument("--normalize", action="store_true",
                               help="Normalize query vectors")

    args = parser.parse_args()

    # Validate command
    if args.command not in ["build", "search"]:
        parser.print_help()
        parser.error("Please specify a command: build or search")

    return args


def build_index(args):
    """Build a FAISS index from embeddings."""
    logging.info(f"Building {args.index_type} index with {args.metric} metric")

    # Create index builder
    builder = FaissIndexBuilder(
        index_type=args.index_type,
        metric=args.metric,
        nlist=args.nlist,
        m=args.m,
        ef_construction=args.ef_construction,
    )

    # Build index from embeddings file
    builder.build_from_file(args.embeddings, normalize=args.normalize)

    # Save the index
    builder.save_index(args.output)
    logging.info(f"Index saved to {args.output}")


def search_index(args):
    """Search a FAISS index with query embeddings."""
    logging.info(f"Searching index {args.index} with k={args.k}")

    # Load query embeddings
    logging.info(f"Loading query embeddings from {args.queries}")
    query_embeddings = np.load(args.queries, allow_pickle=True).item()

    if not isinstance(query_embeddings, dict):
        raise ValueError(f"Query embeddings file should contain a dictionary, got {type(query_embeddings)}")

    # Create searcher
    searcher = FaissSearcher(index_path=args.index)

    # Filter query IDs if specified
    if args.query_ids:
        query_ids = [qid for qid in args.query_ids if qid in query_embeddings]
        if not query_ids:
            raise ValueError("None of the specified query IDs were found in embeddings")
        logging.info(f"Searching for {len(query_ids)} specified queries")
    else:
        query_ids = list(query_embeddings.keys())
        logging.info(f"Searching for all {len(query_ids)} queries")

    # Perform search
    results = searcher.batch_search_by_id(
        query_ids=query_ids,
        embeddings=query_embeddings,
        k=args.k,
        normalize=args.normalize,
    )

    # Convert results to a list of dictionaries
    result_dicts = [result.to_dict() for result in results]

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result_dicts, f, indent=2)

    logging.info(f"Search results saved to {args.output}")


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    setup_logging(args.log_level)

    if args.command == "build":
        build_index(args)
    elif args.command == "search":
        search_index(args)


def cli_main():
    """Entry point for the console script"""
    main()


if __name__ == "__main__":
    main()
```

## src/trialmesh/embeddings/dataset.py

```python
# src/trialmesh/embeddings/dataset.py

from torch.utils.data import Dataset
from typing import List


class EmbeddingDataset(Dataset):
    """Dataset for efficient batching of text to embed."""

    def __init__(self, texts: List[str], ids: List[str], max_length: int = 512):
        """Initialize an embedding dataset.

        Args:
            texts: List of texts to encode
            ids: List of document IDs corresponding to texts
            max_length: Maximum sequence length (not used directly, but useful for reference)
        """
        self.texts = texts
        self.ids = ids
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "id": self.ids[idx]
        }
```

## src/trialmesh/embeddings/factory.py

```python
# src/trialmesh/embeddings/factory.py

import os
import logging
from typing import Any, Dict, Type

from trialmesh.embeddings.base import BaseEmbeddingModel
from trialmesh.embeddings.models import MODEL_REGISTRY


class EmbeddingModelFactory:
    """Factory for creating embedding models based on model type."""

    @staticmethod
    def create_model(
            model_type: str = None,
            model_path: str = None,
            **kwargs
    ) -> BaseEmbeddingModel:
        """Create an embedding model instance.

        Args:
            model_type: Type of model to create (e5-large-v2, bge-large-v1.5, etc.)
            model_path: Path to model directory
            **kwargs: Additional arguments to pass to model constructor

        Returns:
            An instance of the requested embedding model
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
        """Get a list of available model types."""
        return list(MODEL_REGISTRY.keys())
```

## src/trialmesh/embeddings/index_builder.py

```python
# src/trialmesh/embeddings/index_builder.py

import os
import time
import logging
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable


class FaissIndexBuilder:
    """Builder for FAISS indices to enable efficient similarity search."""

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

        Args:
            dimension: Dimension of the embedding vectors

        Returns:
            A FAISS index object
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

    def build_from_dict(
            self,
            embeddings: Dict[str, np.ndarray],
            normalize: bool = None,
    ) -> None:
        """Build a FAISS index from a dictionary of embeddings.

        Args:
            embeddings: Dictionary mapping document IDs to embedding vectors
            normalize: Whether to normalize vectors (for cosine similarity)
        """
        if not embeddings:
            raise ValueError("Empty embeddings dictionary provided")

        # Extract document IDs and vectors
        doc_ids = list(embeddings.keys())
        vectors = np.array([embeddings[doc_id] for doc_id in doc_ids], dtype=np.float32)

        # Build the index
        self.build_from_vectors(vectors, doc_ids, normalize)

    def build_from_vectors(
            self,
            vectors: np.ndarray,
            doc_ids: List[str],
            normalize: bool = None,
    ) -> None:
        """Build a FAISS index from vectors and document IDs.

        Args:
            vectors: Matrix of embedding vectors (n_docs × dimension)
            doc_ids: List of document IDs corresponding to vectors
            normalize: Whether to normalize vectors (for cosine similarity)
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

    def build_from_file(
            self,
            embeddings_file: str,
            normalize: bool = None,
    ) -> None:
        """Build a FAISS index from a saved embeddings file.

        Args:
            embeddings_file: Path to .npy file containing embeddings dictionary
            normalize: Whether to normalize vectors (for cosine similarity)
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
```

## src/trialmesh/embeddings/query.py

```python
# src/trialmesh/embeddings/query.py

import logging
import numpy as np
import faiss
from typing import Dict, List, Optional, Union, Tuple, Any

from trialmesh.embeddings.index_builder import FaissIndexBuilder


class SearchResult:
    """Container for FAISS search results."""

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
        """Convert results to a dictionary."""
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
    """Search interface for FAISS indices."""

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

    def search(
            self,
            query_vector: np.ndarray,
            query_id: str = "query",
            k: int = 10,
            normalize: bool = None,
    ) -> SearchResult:
        """Search the index for vectors similar to the query vector.

        Args:
            query_vector: Query embedding vector
            query_id: ID for the query (for result tracking)
            k: Number of results to return
            normalize: Whether to normalize the query vector

        Returns:
            SearchResult object with matches
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

    def batch_search(
            self,
            query_vectors: np.ndarray,
            query_ids: List[str],
            k: int = 10,
            normalize: bool = None,
    ) -> List[SearchResult]:
        """Search the index for multiple query vectors in batch.

        Args:
            query_vectors: Matrix of query vectors (n_queries × dimension)
            query_ids: List of query IDs corresponding to vectors
            k: Number of results to return per query
            normalize: Whether to normalize the query vectors

        Returns:
            List of SearchResult objects
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

    def search_by_id(
            self,
            query_id: str,
            embeddings: Dict[str, np.ndarray],
            k: int = 10,
            normalize: bool = None,
    ) -> SearchResult:
        """Search using a document ID as the query.

        Args:
            query_id: ID of the query document
            embeddings: Dictionary mapping IDs to embedding vectors
            k: Number of results to return
            normalize: Whether to normalize the query vector

        Returns:
            SearchResult object with matches
        """
        if query_id not in embeddings:
            raise ValueError(f"Query ID {query_id} not found in embeddings dictionary")

        query_vector = embeddings[query_id]
        return self.search(query_vector, query_id, k, normalize)

    def batch_search_by_id(
            self,
            query_ids: List[str],
            embeddings: Dict[str, np.ndarray],
            k: int = 10,
            normalize: bool = None,
    ) -> List[SearchResult]:
        """Search for multiple document IDs in batch.

        Args:
            query_ids: List of query document IDs
            embeddings: Dictionary mapping IDs to embedding vectors
            k: Number of results to return per query
            normalize: Whether to normalize the query vectors

        Returns:
            List of SearchResult objects
        """
        # Check that all query IDs exist in embeddings
        missing_ids = [qid for qid in query_ids if qid not in embeddings]
        if missing_ids:
            raise ValueError(f"Query IDs not found in embeddings dictionary: {missing_ids}")

        # Collect query vectors
        query_vectors = np.array([embeddings[qid] for qid in query_ids], dtype=np.float32)

        return self.batch_search(query_vectors, query_ids, k, normalize)
```

## src/trialmesh/embeddings/run_embedding.py

```python
# src/trialmesh/embeddings/run_embedding.py

import argparse
import logging
import os
import torch
from pathlib import Path
from typing import List, Dict, Optional

from trialmesh.embeddings.factory import EmbeddingModelFactory


def setup_logging(log_level: str = "INFO"):
    """Configure logging."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate embeddings using various models")

    # Model configuration
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the embedding model directory")
    parser.add_argument("--model-type", type=str, default=None,
                        help=f"Type of embedding model: {EmbeddingModelFactory.get_available_models()}")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for processing")
    parser.add_argument("--normalize", action="store_true",
                        help="L2 normalize embeddings")

    # GPU configuration
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda:0, cuda:1, cpu)")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use multiple GPUs with DistributedDataParallel")

    # Input/output options
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Base data directory")
    parser.add_argument("--dataset", type=str, default="sigir2016/processed_cut_summaries",
                        help="Dataset subdirectory under data-dir")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: {data-dir}/{dataset}_embeddings/{model-type})")

    # Processing options
    parser.add_argument("--skip-trials", action="store_true",
                        help="Skip trial embedding generation")
    parser.add_argument("--skip-patients", action="store_true",
                        help="Skip patient embedding generation")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")

    args = parser.parse_args()

    # Set default output directory if not specified
    if args.output_dir is None:
        model_name = os.path.basename(args.model_path.rstrip("/"))
        args.output_dir = os.path.join(args.data_dir, f"{args.dataset}_embeddings", model_name)

    return args


def main():
    """Generate embeddings for trials and patients."""
    args = parse_args()
    setup_logging(args.log_level)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Log GPU information
    logging.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logging.info(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logging.info(f"  {i}: {torch.cuda.get_device_name(i)}")
    else:
        logging.warning("No GPUs available, using CPU")

    # Create embedding model
    logging.info(f"Initializing embedding model from {args.model_path}")
    model = EmbeddingModelFactory.create_model(
        model_type=args.model_type,
        model_path=args.model_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        use_multi_gpu=args.multi_gpu,
        normalize_embeddings=args.normalize,
    )

    # Generate embeddings for trials
    if not args.skip_trials:
        trials_path = os.path.join(args.data_dir, args.dataset, "trial_condensed.jsonl")
        output_path = os.path.join(args.output_dir, "trial_embeddings.npy")
        logging.info(f"Generating trial embeddings from {trials_path}")

        model.encode_corpus(
            jsonl_path=trials_path,
            output_path=output_path,
            text_field="summary",
            id_field="_id",
        )

        if not model.is_distributed or model.local_rank == 0:
            logging.info(f"Trial embeddings saved to {output_path}")

    # Generate embeddings for patients
    if not args.skip_patients:
        patients_path = os.path.join(args.data_dir, args.dataset, "patient_condensed.jsonl")
        output_path = os.path.join(args.output_dir, "patient_embeddings.npy")
        logging.info(f"Generating patient embeddings from {patients_path}")

        model.encode_corpus(
            jsonl_path=patients_path,
            output_path=output_path,
            text_field="summary",
            id_field="_id",
        )

        if not model.is_distributed or model.local_rank == 0:
            logging.info(f"Patient embeddings saved to {output_path}")

    logging.info("Embedding generation complete!")


def cli_main():
    """Entry point for the console script"""
    main()


if __name__ == "__main__":
    main()
```

## src/trialmesh/embeddings/models/__init__.py

```python
# src/trialmesh/embeddings/models/__init__.py

from trialmesh.embeddings.models.e5 import E5LargeV2
from trialmesh.embeddings.models.bge import BGELargeV15
from trialmesh.embeddings.models.clinical_bert import SapBERT, BioClinicalBERT
from trialmesh.embeddings.models.blue_bert import BlueBERT

# Registry of all available models
MODEL_REGISTRY = {
    "e5-large-v2": E5LargeV2,
    "bge-large-v1.5": BGELargeV15,
    "sapbert": SapBERT,
    "bio-clinicalbert": BioClinicalBERT,
    "bluebert": BlueBERT,
}
```

## src/trialmesh/embeddings/models/bge.py

```python
# src/trialmesh/embeddings/models/bge.py

import logging
import torch
from typing import List
from transformers import AutoModel, AutoTokenizer
from trialmesh.embeddings.base import BaseEmbeddingModel


class BGELargeV15(BaseEmbeddingModel):
    """Embedding model using BAAI/bge-large-en-v1.5."""

    def _load_model(self):
        """Load BGE model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            logging.info(f"Loaded BGE model from {self.model_path}")
        except Exception as e:
            logging.error(f"Error loading BGE model: {str(e)}")
            raise

    def _batch_encode(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts using BGE model."""
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
```

## src/trialmesh/embeddings/models/blue_bert.py

```python
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
```

## src/trialmesh/embeddings/models/clinical_bert.py

```python
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
```

## src/trialmesh/embeddings/models/e5.py

```python
# src/trialmesh/embeddings/models/e5.py

import logging
import torch
from typing import List
from transformers import AutoModel, AutoTokenizer
from trialmesh.embeddings.base import BaseEmbeddingModel


class E5LargeV2(BaseEmbeddingModel):
    """Embedding model using intfloat/e5-large-v2."""

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
        """Encode a batch of texts using E5 model."""
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
```

## src/trialmesh/llm/__init__.py

```python
# src/trialmesh/llm/__init__.py
```

## src/trialmesh/llm/llama_runner.py

```python
# src/trialmesh/llm/llama_runner.py

import hashlib
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, NamedTuple

from vllm import LLM, EngineArgs, SamplingParams


class LlamaResponse(NamedTuple):
    """Container for LLaMA response with token counts and prompt info."""
    text: str
    input_tokens: int
    output_tokens: int
    user_prompt: str
    system_prompt: Optional[str] = None


class LlamaRunner:
    """Runner for LLaMA models using vLLM with caching support."""

    def __init__(
            self,
            model_path: str,
            cache_dir: Optional[str] = None,
            tensor_parallel_size: int = 4,
            max_tokens: int = 1024,
            max_model_len: int = 2048,
            max_batch_size: int = 8,
            use_cache: bool = True,
            temperature: float = 0.0,
            top_p: float = 1.0,
            top_k: int = -1,
    ):
        """Initialize the LlamaRunner with model and caching configuration.

        Args:
            model_path: Path to the LLaMA model
            cache_dir: Directory to store cached responses
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            max_tokens: Maximum sequence length for output
            max_model_len: Maximum model context length
            max_batch_size: Maximum batch size for inference
            use_cache: Whether to use caching
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
        """
        # Store config
        self.model_path = model_path
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/llm_responses")

        # Ensure cache dir exists
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize engine args
        engine_args = EngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_seq_len_to_capture=max_tokens,
            disable_custom_all_reduce=True,
            max_num_seqs=max_batch_size,
            max_model_len=max_model_len,
            enforce_eager=True
        )

        # Initialize default sampling params
        self.default_sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            min_tokens=10
        )

        # Initialize LLM engine
        self.llm = LLM(**vars(engine_args))

        logging.info(f"LlamaRunner initialized with model: {model_path}")

    def _get_cache_key(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a hash key for caching based on prompt input."""
        content_to_hash = prompt
        if system_prompt:
            content_to_hash = f"{system_prompt}|||{prompt}"

        # Create a deterministic hash
        key = hashlib.sha256(content_to_hash.encode('utf-8')).hexdigest()
        return key

    def _get_cached_response(self, cache_key: str) -> Optional[LlamaResponse]:
        """Retrieve cached response if it exists."""
        if not self.use_cache:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logging.debug(f"Cache hit for key: {cache_key}")
                return LlamaResponse(
                    text=data['text'],
                    input_tokens=data['input_tokens'],
                    output_tokens=data['output_tokens'],
                    user_prompt=data.get('user_prompt', ""),
                    system_prompt=data.get('system_prompt', "")
                )
        return None

    def _cache_response(self, cache_key: str, response: LlamaResponse) -> None:
        """Save response with token counts and prompts to cache."""
        if not self.use_cache:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        data = {
            'text': response.text,
            'input_tokens': response.input_tokens,
            'output_tokens': response.output_tokens,
            'user_prompt': response.user_prompt,
            'system_prompt': response.system_prompt if response.system_prompt else ""
        }

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)  # Pretty print for readability
        logging.debug(f"Cached response for key: {cache_key}")

    def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
    ) -> LlamaResponse:
        """Generate text from a prompt with optional system prompt."""
        # Try to get from cache first
        cache_key = self._get_cache_key(prompt, system_prompt)
        cached_response = self._get_cached_response(cache_key)
        if cached_response is not None:
            return cached_response

        # Create sampling params with any overrides
        sampling_params = SamplingParams(
            temperature=temperature if temperature is not None else self.default_sampling_params.temperature,
            top_p=self.default_sampling_params.top_p,
            top_k=self.default_sampling_params.top_k,
            max_tokens=max_tokens if max_tokens is not None else self.default_sampling_params.max_tokens,
            min_tokens=self.default_sampling_params.min_tokens,
        )

        # Run generation
        if system_prompt:
            # Use chat interface
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            outputs = self.llm.chat([messages], sampling_params)
            response_text = outputs[0].outputs[0].text
            input_tokens = len(outputs[0].prompt_token_ids) if outputs[0].prompt_token_ids is not None else 0
            output_tokens = len(outputs[0].outputs[0].token_ids) if outputs[0].outputs[0].token_ids is not None else 0
        else:
            # Use completion interface
            outputs = self.llm.generate([prompt], sampling_params)
            response_text = outputs[0].outputs[0].text
            input_tokens = len(outputs[0].prompt_token_ids) if outputs[0].prompt_token_ids is not None else 0
            output_tokens = len(outputs[0].outputs[0].token_ids) if outputs[0].outputs[0].token_ids is not None else 0

        # Create response object with prompt info
        response = LlamaResponse(
            text=response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            user_prompt=prompt,
            system_prompt=system_prompt
        )

        # Cache the response
        self._cache_response(cache_key, response)

        return response

    def generate_batch(
            self,
            prompts: List[str],
            system_prompt: Optional[str] = None,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
    ) -> List[LlamaResponse]:
        """Generate responses for a batch of prompts."""
        # Find which prompts are cached and which need generation
        uncached_prompts = []
        uncached_indices = []
        results = [None] * len(prompts)

        # Check cache for each prompt
        for i, prompt in enumerate(prompts):
            cache_key = self._get_cache_key(prompt, system_prompt)
            cached_response = self._get_cached_response(cache_key)

            if cached_response is not None:
                results[i] = cached_response
            else:
                uncached_prompts.append(prompt)
                uncached_indices.append(i)

        # If all prompts were cached, return early
        if not uncached_prompts:
            return results

        # Create sampling params with any overrides
        sampling_params = SamplingParams(
            temperature=temperature if temperature is not None else self.default_sampling_params.temperature,
            top_p=self.default_sampling_params.top_p,
            top_k=self.default_sampling_params.top_k,
            max_tokens=max_tokens if max_tokens is not None else self.default_sampling_params.max_tokens,
            min_tokens=self.default_sampling_params.min_tokens,
        )

        # Generate for uncached prompts
        if system_prompt:
            # Use chat interface
            conversations = []
            for prompt in uncached_prompts:
                conversations.append([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ])
            outputs = self.llm.chat(conversations, sampling_params)

            # Extract responses with token counts and prompts
            new_responses = []
            for i, output in enumerate(outputs):
                prompt = uncached_prompts[i]
                response_text = output.outputs[0].text
                input_tokens = len(output.prompt_token_ids) if output.prompt_token_ids is not None else 0
                output_tokens = len(output.outputs[0].token_ids) if output.outputs[0].token_ids is not None else 0
                new_responses.append(LlamaResponse(
                    text=response_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    user_prompt=prompt,
                    system_prompt=system_prompt
                ))
        else:
            # Use completion interface
            outputs = self.llm.generate(uncached_prompts, sampling_params)

            # Extract responses with token counts
            new_responses = []
            for output in outputs:
                response_text = output.outputs[0].text
                input_tokens = len(output.prompt_token_ids) if output.prompt_token_ids is not None else 0
                output_tokens = len(output.outputs[0].token_ids) if output.outputs[0].token_ids is not None else 0
                new_responses.append(LlamaResponse(
                    text=response_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                ))

        # Cache the new responses and fill in results
        for i, (prompt, response) in enumerate(zip(uncached_prompts, new_responses)):
            cache_key = self._get_cache_key(prompt, system_prompt)
            self._cache_response(cache_key, response)
            results[uncached_indices[i]] = response

        return results
```

## src/trialmesh/llm/prompt_runner.py

```python
# src/trialmesh/llm/prompt_runner.py

import logging
from typing import Dict, List, Optional, Union, Tuple

from trialmesh.utils.prompt_registry import PromptRegistry
from trialmesh.llm.llama_runner import LlamaRunner, LlamaResponse


class PromptRunner:
    """High-level interface for running prompts from the registry."""

    def __init__(
            self,
            llama_runner: LlamaRunner,
            prompt_registry: Optional[PromptRegistry] = None,
    ):
        self.llm = llama_runner
        self.prompts = prompt_registry or PromptRegistry()

    def run_prompt(
            self,
            prompt_name: str,
            variables: Dict[str, str],
            override_system_prompt: Optional[str] = None,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
    ) -> LlamaResponse:
        """Run a prompt from the registry with the given variables."""
        prompt_pair = self.prompts.get(prompt_name)
        if not prompt_pair or not prompt_pair.get("user"):
            raise ValueError(f"Prompt '{prompt_name}' not found in registry")

        # Get system and user prompts
        system_prompt = override_system_prompt or prompt_pair.get("system", "")
        user_prompt = prompt_pair.get("user", "")

        # Format the user prompt template with variables
        try:
            formatted_user_prompt = user_prompt.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing variable {e} for prompt '{prompt_name}'")

        # Run the formatted prompt with the system prompt
        return self.llm.generate(
            prompt=formatted_user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def run_prompt_batch(
            self,
            prompt_name: str,
            variables_list: List[Dict[str, str]],
            override_system_prompt: Optional[str] = None,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
    ) -> List[Optional[LlamaResponse]]:
        """Run a prompt from the registry with multiple sets of variables."""
        prompt_pair = self.prompts.get(prompt_name)
        if not prompt_pair or not prompt_pair.get("user"):
            raise ValueError(f"Prompt '{prompt_name}' not found in registry")

        # Get system and user prompts
        system_prompt = override_system_prompt or prompt_pair.get("system", "")
        user_prompt = prompt_pair.get("user", "")

        # Format each prompt with its variables
        formatted_prompts = []
        for variables in variables_list:
            try:
                formatted_prompt = user_prompt.format(**variables)
                formatted_prompts.append(formatted_prompt)
            except KeyError as e:
                logging.warning(f"Skipping prompt with missing variable {e}")
                formatted_prompts.append(None)

        # Filter out None prompts
        valid_prompts = [p for p in formatted_prompts if p is not None]
        valid_indices = [i for i, p in enumerate(formatted_prompts) if p is not None]

        # Run valid prompts
        if not valid_prompts:
            return []

        results = self.llm.generate_batch(
            prompts=valid_prompts,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Reconstruct full results list with None for invalid prompts
        full_results = [None] * len(variables_list)
        for i, result in zip(valid_indices, results):
            full_results[i] = result

        return full_results
```

## src/trialmesh/llm/summarizers.py

```python
# src/trialmesh/llm/summarizers.py

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

from trialmesh.llm.llama_runner import LlamaRunner, LlamaResponse
from trialmesh.llm.prompt_runner import PromptRunner
from trialmesh.utils.prompt_registry import PromptRegistry


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to a JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


class Summarizer:
    """Generate summaries for trials and patients using LLM."""

    def __init__(
            self,
            model_path: str,
            cache_dir: str = None,
            tensor_parallel_size: int = 4,
            max_model_len: int = 2048,
            batch_size: int = 8,
    ):
        """Initialize the summarizer with model configuration."""
        self.runner = LlamaRunner(
            model_path=model_path,
            cache_dir=cache_dir,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            max_batch_size=batch_size,
        )
        self.prompt_runner = PromptRunner(self.runner)

    def summarize_trials(
            self,
            trials_path: str,
            output_dir: str,
            batch_size: int = 8,
            max_tokens: int = 1024,
            condensed_trial_only: bool = True,  # New parameter trial
    ) -> None:
        """Generate summaries for clinical trials."""
        logging.info(f"Loading trials from {trials_path}")
        logging.info(f"condensed_trial_only {condensed_trial_only} batch_size {batch_size}")
        trials = load_jsonl(trials_path)
        logging.info(f"Loaded {len(trials)} trials")

        # Prepare output directory
        os.makedirs(output_dir, exist_ok=True)
        full_summary_path = os.path.join(output_dir, "trial_summaries.jsonl")
        condensed_summary_path = os.path.join(output_dir, "trial_condensed.jsonl")

        # Process in batches
        all_full_summaries = []
        all_condensed_summaries = []

        for i in tqdm(range(0, len(trials), batch_size), desc="Processing trials"):
            batch = trials[i:i + batch_size]

            # Prepare variables for prompts
            variables_list = [{"trial_text": self._format_trial(trial)} for trial in batch]

            # Run full summaries only if needed
            full_summary_responses = []
            if not condensed_trial_only:
                full_summary_responses = self.prompt_runner.run_prompt_batch(
                    prompt_name="trial_summary_sigir2016",
                    variables_list=variables_list,
                    max_tokens=max_tokens,
                )

            # Run condensed summaries for embedding (always needed)
            condensed_summary_responses = self.prompt_runner.run_prompt_batch(
                prompt_name="trial_condensed_sigir2016",
                variables_list=variables_list,
                max_tokens=min(max_tokens, 512),
            )

            # Process and save results
            for j, trial in enumerate(batch):
                trial_id = trial["_id"]

                # Full summary (if not skipped)
                if not condensed_trial_only and full_summary_responses and full_summary_responses[j]:
                    full_summary = {
                        "_id": trial_id,
                        "summary": full_summary_responses[j].text,
                        "input_tokens": full_summary_responses[j].input_tokens,
                        "output_tokens": full_summary_responses[j].output_tokens
                    }
                    all_full_summaries.append(full_summary)

                # Condensed summary for embedding
                if condensed_summary_responses[j]:
                    condensed_summary = {
                        "_id": trial_id,
                        "summary": condensed_summary_responses[j].text,
                        "input_tokens": condensed_summary_responses[j].input_tokens,
                        "output_tokens": condensed_summary_responses[j].output_tokens
                    }
                    all_condensed_summaries.append(condensed_summary)

        # Save results
        if not condensed_trial_only and all_full_summaries:
            save_jsonl(all_full_summaries, full_summary_path)
            logging.info(f"Saved {len(all_full_summaries)} trial summaries to {full_summary_path}")

        save_jsonl(all_condensed_summaries, condensed_summary_path)
        logging.info(f"Saved {len(all_condensed_summaries)} condensed trial summaries to {condensed_summary_path}")

    def summarize_patients(
            self,
            patients_path: str,
            output_dir: str,
            batch_size: int = 8,
            max_tokens: int = 1024,

    ) -> None:
        """Generate summaries for patient queries."""
        logging.info(f"Loading patients from {patients_path}")
        patients = load_jsonl(patients_path)
        logging.info(f"Loaded {len(patients)} patients")

        # Prepare output directory
        os.makedirs(output_dir, exist_ok=True)
        full_summary_path = os.path.join(output_dir, "patient_summaries.jsonl")
        condensed_summary_path = os.path.join(output_dir, "patient_condensed.jsonl")

        # Process in batches
        all_full_summaries = []
        all_condensed_summaries = []

        for i in tqdm(range(0, len(patients), batch_size), desc="Processing patients"):
            batch = patients[i:i + batch_size]

            # Prepare variables for prompts
            variables_list = [{"patient_text": patient["text"]} for patient in batch]

            # Run full summaries
            full_summary_responses = self.prompt_runner.run_prompt_batch(
                prompt_name="patient_summary_sigir2016",
                variables_list=variables_list,
                max_tokens=max_tokens,
            )

            # Run condensed summaries for embedding
            condensed_summary_responses = self.prompt_runner.run_prompt_batch(
                prompt_name="patient_condensed_sigir2016",
                variables_list=variables_list,
                max_tokens=min(max_tokens, 512),
            )

            # Process and save results
            for j, patient in enumerate(batch):
                patient_id = patient["_id"]

                # Full summary
                if full_summary_responses[j]:
                    full_summary = {
                        "_id": patient_id,
                        "summary": full_summary_responses[j].text,
                        "input_tokens": full_summary_responses[j].input_tokens,
                        "output_tokens": full_summary_responses[j].output_tokens,
                    }
                    all_full_summaries.append(full_summary)

                # Condensed summary for embedding
                if condensed_summary_responses[j]:
                    condensed_summary = {
                        "_id": patient_id,
                        "summary": condensed_summary_responses[j].text,
                        "input_tokens": condensed_summary_responses[j].input_tokens,
                        "output_tokens": condensed_summary_responses[j].output_tokens,
                    }
                    all_condensed_summaries.append(condensed_summary)

        # Save all results
        save_jsonl(all_full_summaries, full_summary_path)
        save_jsonl(all_condensed_summaries, condensed_summary_path)
        logging.info(f"Saved {len(all_full_summaries)} patient summaries to {full_summary_path}")
        logging.info(f"Saved {len(all_condensed_summaries)} condensed patient summaries to {condensed_summary_path}")

    def _format_trial(self, trial: Dict[str, Any]) -> str:
        """Format trial data for LLM input."""
        metadata = trial.get("metadata", {})

        formatted_text = f"Title: {trial.get('title', '')}\n\n"

        if metadata.get("brief_summary"):
            formatted_text += f"Summary: {metadata.get('brief_summary', '')}\n\n"

        if metadata.get("detailed_description"):
            formatted_text += f"Description: {metadata.get('detailed_description', '')}\n\n"

        if metadata.get("inclusion_criteria"):
            formatted_text += f"Inclusion Criteria: {metadata.get('inclusion_criteria', '')}\n\n"

        if metadata.get("exclusion_criteria"):
            formatted_text += f"Exclusion Criteria: {metadata.get('exclusion_criteria', '')}\n\n"

        if metadata.get("diseases_list"):
            diseases = ", ".join(metadata.get("diseases_list", []))
            formatted_text += f"Conditions: {diseases}\n\n"

        if metadata.get("drugs_list"):
            drugs = ", ".join(metadata.get("drugs_list", []))
            formatted_text += f"Interventions: {drugs}\n\n"

        if metadata.get("phase"):
            formatted_text += f"Phase: {metadata.get('phase', '')}\n\n"

        return formatted_text.strip()


def main():
    """Command-line interface for running summarizations."""
    parser = argparse.ArgumentParser(description="Generate summaries for trials and patients")

    # Model configuration
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the LLaMA model")
    parser.add_argument("--tensor-parallel-size", type=int, default=4,
                        help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Maximum sequence length for output")
    parser.add_argument("--max-model-len", type=int, default=2048,
                        help="Maximum context length for Input+output")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for processing")

    # Data paths
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Base data directory")
    parser.add_argument("--dataset", type=str, default="sigir2016/processed_cut",
                        help="Dataset subdirectory under data-dir")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: {data-dir}/{dataset}_summaries)")
    parser.add_argument("--cache-dir", type=str, default="./cache",
                        help="Cache directory for LLM responses")

    # Processing options
    parser.add_argument("--skip-trials", action="store_true",
                        help="Skip trial summarization")
    parser.add_argument("--skip-patients", action="store_true",
                        help="Skip patient summarization")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")
    parser.add_argument("--condensed-trial-only", action="store_true",
                        help="Generate only condensed trial summaries (skip full summaries)")

    args = parser.parse_args()

    # Set up logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, f"{args.dataset}_summaries")

    # Create summarizer
    summarizer = Summarizer(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
    )

    # Run summarizations
    data_path = os.path.join(args.data_dir, args.dataset)

    if not args.skip_trials:
        trials_path = os.path.join(data_path, "corpus.jsonl")
        summarizer.summarize_trials(
            trials_path=trials_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
        )

    if not args.skip_patients:
        patients_path = os.path.join(data_path, "queries.jsonl")
        summarizer.summarize_patients(
            patients_path=patients_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
        )

    logging.info("Summarization complete!")


def cli_main():
    """Entry point for the console script"""
    main()


if __name__ == "__main__":
    main()
```

## src/trialmesh/match/__init__.py

```python
# src/trialmesh/match/__init__.py
```

## src/trialmesh/match/matcher.py

```python
# src/trialmesh/match/matcher.py

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from tqdm import tqdm

from trialmesh.llm.llama_runner import LlamaRunner
from trialmesh.llm.prompt_runner import PromptRunner
from trialmesh.utils.prompt_registry import PromptRegistry


class TrialMatcher:
    """Pipeline for matching patients to trials using LLM evaluation."""

    def __init__(
            self,
            llm: LlamaRunner,
            data_dir: str,
            patient_summaries_path: str,
            trials_path: str,
            batch_size: int = 8,
    ):
        """Initialize the trial matcher.

        Args:
            llm: LlamaRunner instance
            data_dir: Base data directory
            patient_summaries_path: Path to patient summaries relative to data_dir
            trials_path: Path to trial corpus relative to data_dir
            batch_size: Batch size for processing
        """
        self.data_dir = data_dir
        self.patient_summaries_path = os.path.join(data_dir, patient_summaries_path)
        self.trials_path = os.path.join(data_dir, trials_path)
        self.batch_size = batch_size

        # Initialize LLM components
        self.llm = llm
        self.prompt_runner = PromptRunner(llm)

        # Load data
        self.patients = self._load_patients()
        self.trials = self._load_trials()

        logging.info(f"Loaded {len(self.patients)} patients and {len(self.trials)} trials")

    def _load_patients(self) -> Dict[str, Dict[str, Any]]:
        """Load patient summaries."""
        patients = {}
        with open(self.patient_summaries_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    patient = json.loads(line)
                    patient_id = patient.get("_id")
                    if patient_id:
                        patients[patient_id] = patient
                except json.JSONDecodeError:
                    logging.warning(f"Error parsing patient line: {line[:100]}...")

        return patients

    def _load_trials(self) -> Dict[str, Dict[str, Any]]:
        """Load trial data."""
        trials = {}
        with open(self.trials_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    trial = json.loads(line)
                    trial_id = trial.get("_id")
                    if trial_id:
                        trials[trial_id] = trial
                except json.JSONDecodeError:
                    logging.warning(f"Error parsing trial line: {line[:100]}...")

        return trials

    def _format_trial(self, trial: Dict[str, Any]) -> str:
        """Format trial data for LLM input."""
        metadata = trial.get("metadata", {})

        formatted_text = f"Title: {trial.get('title', '')}\n\n"

        if metadata.get("brief_summary"):
            formatted_text += f"Summary: {metadata.get('brief_summary', '')}\n\n"

        if metadata.get("detailed_description"):
            formatted_text += f"Description: {metadata.get('detailed_description', '')}\n\n"

        if metadata.get("inclusion_criteria"):
            formatted_text += f"Inclusion Criteria: {metadata.get('inclusion_criteria', '')}\n\n"

        if metadata.get("exclusion_criteria"):
            formatted_text += f"Exclusion Criteria: {metadata.get('exclusion_criteria', '')}\n\n"

        if metadata.get("diseases_list"):
            diseases = ", ".join(metadata.get("diseases_list", []))
            formatted_text += f"Conditions: {diseases}\n\n"

        if metadata.get("drugs_list"):
            drugs = ", ".join(metadata.get("drugs_list", []))
            formatted_text += f"Interventions: {drugs}\n\n"

        if metadata.get("phase"):
            formatted_text += f"Phase: {metadata.get('phase', '')}\n\n"

        return formatted_text.strip()

    def match(
            self,
            search_results: List[Dict[str, Any]],
            top_k: Optional[int] = None,
            skip_exclusion: bool = False,
            skip_inclusion: bool = False,
            skip_scoring: bool = False,
            include_all_trials: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run the matching pipeline.

        Args:
            search_results: List of search results from FAISS search
            top_k: Number of trials to process per patient (None for all trials)
            skip_exclusion: Skip exclusion filtering step
            skip_inclusion: Skip inclusion filtering step
            skip_scoring: Skip final scoring step
            include_all_trials: Include all trials in output, even those filtered out

        Returns:
            List of patient-trial matches with evaluation results
        """
        logging.info("Starting trial matching process")

        # Process patients in order
        all_patient_results = []

        for patient_result in tqdm(search_results, desc="Processing patients"):
            patient_id = patient_result["query_id"]
            patient_data = self.patients.get(patient_id)

            if not patient_data:
                logging.warning(f"Patient {patient_id} not found in summaries, skipping")
                continue

            patient_summary = patient_data.get("summary", "")

            # Get trial results for this patient
            if top_k is not None:
                trial_results = patient_result.get("results", [])[:top_k]
                logging.info(f"Using top {top_k} trials for patient {patient_id}")
            else:
                trial_results = patient_result.get("results", [])
                logging.info(f"Using all {len(trial_results)} trials for patient {patient_id}")

            trial_ids = [tr["doc_id"] for tr in trial_results]

            # Prepare patient result structure
            patient_match_result = {
                "patient_id": patient_id,
                "patient_summary": patient_summary,
                "trial_evaluations": []
            }

            # Get trial data
            valid_trials = []
            for trial_id in trial_ids:
                trial_data = self.trials.get(trial_id)
                if trial_data:
                    valid_trials.append((trial_id, trial_data))
                else:
                    logging.warning(f"Trial {trial_id} not found in corpus, skipping")

            # Dictionary to track all trials for complete results option
            all_trial_results = {}

            # 1. Exclusion Filter
            if not skip_exclusion:
                filtered_trials, excluded_trials = self._apply_exclusion_filter(
                    patient_summary,
                    valid_trials,
                    return_excluded=include_all_trials
                )

                # Track all trials if needed
                if include_all_trials:
                    for trial_result in filtered_trials + excluded_trials:
                        all_trial_results[trial_result["trial_id"]] = trial_result
            else:
                # Skip exclusion filter - pass all trials through with PASS verdict
                filtered_trials = []
                for trial_id, trial_data in valid_trials:
                    trial_result = {
                        "trial_id": trial_id,
                        "trial_data": trial_data,
                        "exclusion_result": {"verdict": "PASS", "reason": "Exclusion filter skipped"}
                    }
                    filtered_trials.append(trial_result)
                    if include_all_trials:
                        all_trial_results[trial_id] = trial_result

            # 2. Inclusion Filter (for trials that passed exclusion)
            if not skip_inclusion:
                inclusion_results, failed_inclusion = self._apply_inclusion_filter(
                    patient_summary,
                    filtered_trials,
                    return_failed=include_all_trials
                )

                # Track inclusion failures if needed
                if include_all_trials:
                    for trial_result in failed_inclusion:
                        all_trial_results[trial_result["trial_id"]] = trial_result
            else:
                # Skip inclusion filter - pass all trials through with UNDETERMINED verdict
                inclusion_results = []
                for trial_result in filtered_trials:
                    trial_result["inclusion_result"] = {
                        "verdict": "UNDETERMINED",
                        "missing_information": "None",
                        "unmet_criteria": "None",
                        "reasoning": "Inclusion filter skipped"
                    }
                    inclusion_results.append(trial_result)
                    if include_all_trials:
                        all_trial_results[trial_result["trial_id"]] = trial_result

            # 3. Final Scoring (for trials that didn't fail inclusion)
            if not skip_scoring:
                final_results = self._apply_scoring(patient_summary, inclusion_results)

                # Update tracked results with scores
                if include_all_trials:
                    for trial_result in final_results:
                        all_trial_results[trial_result["trial_id"]] = trial_result
            else:
                # Skip scoring - keep all trials with default score
                final_results = []
                for trial_result in inclusion_results:
                    trial_result["scoring_result"] = {
                        "score": "50",
                        "verdict": "POSSIBLE MATCH",
                        "reasoning": "Scoring skipped"
                    }
                    final_results.append(trial_result)
                    if include_all_trials:
                        all_trial_results[trial_result["trial_id"]] = trial_result

            # Add trial evaluations to patient result
            if include_all_trials:
                # Include all trials we've tracked
                for trial_id, trial_result in all_trial_results.items():
                    trial_data = trial_result["trial_data"]

                    evaluation = {
                        "trial_id": trial_id,
                        "trial_title": trial_data.get("title", ""),
                        "exclusion_result": trial_result.get("exclusion_result", {}),
                        "inclusion_result": trial_result.get("inclusion_result", {}),
                        "scoring_result": trial_result.get("scoring_result", {})
                    }

                    patient_match_result["trial_evaluations"].append(evaluation)
            else:
                # Include only trials that made it through all filters
                for trial_result in final_results:
                    trial_id = trial_result["trial_id"]
                    trial_data = trial_result["trial_data"]

                    evaluation = {
                        "trial_id": trial_id,
                        "trial_title": trial_data.get("title", ""),
                        "exclusion_result": trial_result.get("exclusion_result", {}),
                        "inclusion_result": trial_result.get("inclusion_result", {}),
                        "scoring_result": trial_result.get("scoring_result", {})
                    }

                    patient_match_result["trial_evaluations"].append(evaluation)

            # Add patient result to all results
            all_patient_results.append(patient_match_result)

        logging.info(f"Completed matching for {len(all_patient_results)} patients")
        return all_patient_results

    def _apply_exclusion_filter(
            self,
            patient_summary: str,
            trials: List[Tuple[str, Dict[str, Any]]],
            return_excluded: bool = False
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
        """Apply exclusion filter to trials.

        Args:
            patient_summary: Patient summary text
            trials: List of (trial_id, trial_data) tuples
            return_excluded: Whether to return excluded trials

        Returns:
            If return_excluded is False:
                List of dictionaries with trial data and exclusion results for trials that passed
            If return_excluded is True:
                Tuple of (passed_trials, excluded_trials)
        """
        logging.info(f"Running exclusion filter on {len(trials)} trials")

        # Prepare batches for processing
        batches = [trials[i:i + self.batch_size] for i in range(0, len(trials), self.batch_size)]

        # Process each batch
        passed_trials = []
        excluded_trials = []

        for batch in tqdm(batches, desc="Exclusion filtering"):
            variables_list = []

            for trial_id, trial_data in batch:
                # Extract exclusion criteria
                exclusion_criteria = trial_data.get("metadata", {}).get("exclusion_criteria", "")

                variables = {
                    "patient_summary": patient_summary,
                    "exclusion_criteria": exclusion_criteria or "No exclusion criteria specified."
                }

                variables_list.append(variables)

            # Run LLM for the batch
            responses = self.prompt_runner.run_prompt_batch(
                prompt_name="exclusion_filter",
                variables_list=variables_list
            )

            # Process responses
            for i, ((trial_id, trial_data), response) in enumerate(zip(batch, responses)):
                if response is None:
                    logging.warning(f"No response for trial {trial_id}, skipping")
                    continue

                # Extract verdict and reason using regex
                verdict_match = re.search(r"VERDICT:\s*(\w+)", response.text)
                reason_match = re.search(r"REASON:\s*(.*?)(?=\n\n|\Z)", response.text, re.DOTALL)

                verdict = verdict_match.group(1) if verdict_match else "PASS"
                reason = reason_match.group(1).strip() if reason_match else "No reason provided"

                # If exclusion criteria is empty, automatically PASS
                if not trial_data.get("metadata", {}).get("exclusion_criteria", "").strip():
                    verdict = "PASS"
                    reason = "No exclusion criteria specified"

                # Create trial result with exclusion data
                trial_result = {
                    "trial_id": trial_id,
                    "trial_data": trial_data,
                    "exclusion_result": {"verdict": verdict, "reason": reason}
                }

                # If verdict is not EXCLUDE, add trial to passed list
                if verdict != "EXCLUDE":
                    passed_trials.append(trial_result)
                elif return_excluded:
                    excluded_trials.append(trial_result)

        logging.info(f"Exclusion filter passed {len(passed_trials)} of {len(trials)} trials")

        if return_excluded:
            return passed_trials, excluded_trials
        return passed_trials

    def _apply_inclusion_filter(
            self,
            patient_summary: str,
            trials: List[Dict[str, Any]],
            return_failed: bool = False
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
        """Apply inclusion filter to trials that passed exclusion.

        Args:
            patient_summary: Patient summary text
            trials: List of dictionaries with trial data and exclusion results
            return_failed: Whether to return trials that failed inclusion

        Returns:
            If return_failed is False:
                List of dictionaries with trial data, exclusion and inclusion results
            If return_failed is True:
                Tuple of (included_trials, failed_trials)
        """
        logging.info(f"Running inclusion filter on {len(trials)} trials")

        # Prepare batches for processing
        batches = [trials[i:i + self.batch_size] for i in range(0, len(trials), self.batch_size)]

        # Process each batch
        included_trials = []
        failed_trials = []

        for batch in tqdm(batches, desc="Inclusion filtering"):
            variables_list = []

            for trial_result in batch:
                trial_data = trial_result["trial_data"]

                # Extract inclusion criteria
                inclusion_criteria = trial_data.get("metadata", {}).get("inclusion_criteria", "")

                variables = {
                    "patient_summary": patient_summary,
                    "inclusion_criteria": inclusion_criteria or "No inclusion criteria specified."
                }

                variables_list.append(variables)

            # Run LLM for the batch
            responses = self.prompt_runner.run_prompt_batch(
                prompt_name="inclusion_filter",
                variables_list=variables_list
            )

            # Process responses
            for i, (trial_result, response) in enumerate(zip(batch, responses)):
                if response is None:
                    logging.warning(f"No response for trial {trial_result['trial_id']}, skipping")
                    continue

                # Extract information using regex
                verdict_match = re.search(r"VERDICT:\s*(\w+)", response.text)
                missing_match = re.search(r"MISSING INFORMATION:\s*(.*?)(?=\n\n|\Z)", response.text, re.DOTALL)
                unmet_match = re.search(r"UNMET CRITERIA:\s*(.*?)(?=\n\n|\Z)", response.text, re.DOTALL)
                reasoning_match = re.search(r"REASONING:\s*(.*?)(?=\n\n|\Z)", response.text, re.DOTALL)

                verdict = verdict_match.group(1) if verdict_match else "UNDETERMINED"
                missing = missing_match.group(1).strip() if missing_match else "None"
                unmet = unmet_match.group(1).strip() if unmet_match else "None"
                reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"

                # If inclusion criteria is empty, automatically UNDETERMINED
                trial_data = trial_result["trial_data"]
                if not trial_data.get("metadata", {}).get("inclusion_criteria", "").strip():
                    verdict = "UNDETERMINED"
                    reasoning = "No inclusion criteria specified"

                # Add inclusion result to trial data
                trial_result["inclusion_result"] = {
                    "verdict": verdict,
                    "missing_information": missing,
                    "unmet_criteria": unmet,
                    "reasoning": reasoning
                }

                # Add to appropriate list based on verdict
                if verdict != "FAIL":
                    included_trials.append(trial_result)
                elif return_failed:
                    failed_trials.append(trial_result)

        logging.info(f"Inclusion filter kept {len(included_trials)} of {len(trials)} trials")

        if return_failed:
            return included_trials, failed_trials
        return included_trials

    def _apply_scoring(
            self,
            patient_summary: str,
            trials: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply final scoring to trials that passed inclusion filter.

        Args:
            patient_summary: Patient summary text
            trials: List of dictionaries with trial data, exclusion and inclusion results

        Returns:
            List of dictionaries with trial data and all evaluation results
        """
        logging.info(f"Running final scoring on {len(trials)} trials")

        # Prepare batches for processing
        batches = [trials[i:i + self.batch_size] for i in range(0, len(trials), self.batch_size)]

        # Process each batch
        scored_trials = []

        for batch in tqdm(batches, desc="Final scoring"):
            variables_list = []

            for trial_result in batch:
                trial_data = trial_result["trial_data"]

                # Format trial text
                trial_summary = self._format_trial(trial_data)

                variables = {
                    "patient_summary": patient_summary,
                    "trial_summary": trial_summary
                }

                variables_list.append(variables)

            # Run LLM for the batch
            responses = self.prompt_runner.run_prompt_batch(
                prompt_name="final_match_scoring",
                variables_list=variables_list
            )

            # Process responses
            for i, (trial_result, response) in enumerate(zip(batch, responses)):
                if response is None:
                    logging.warning(f"No response for trial {trial_result['trial_id']}, skipping")
                    continue

                # Extract information using regex
                score_match = re.search(r"SCORE:\s*(\d+)", response.text)
                verdict_match = re.search(r"VERDICT:\s*(.*?)(?=\n|\Z)", response.text)
                reasoning_match = re.search(r"REASONING:\s*(.*?)(?=\n\n|\Z)", response.text, re.DOTALL)

                score = score_match.group(1) if score_match else "0"
                verdict = verdict_match.group(1).strip() if verdict_match else "UNSUITABLE"
                reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"

                # Add scoring result to trial data
                trial_result["scoring_result"] = {
                    "score": score,
                    "verdict": verdict,
                    "reasoning": reasoning
                }

                scored_trials.append(trial_result)

        logging.info(f"Completed scoring for {len(scored_trials)} trials")
        return scored_trials
```

## src/trialmesh/utils/__init__.py

```python
# src/trialmesh/utils/__init__.py
```

## src/trialmesh/utils/clean_run.py

```python
#!/usr/bin/env python
# src/trialmesh/utils/clean_run.py

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict


def setup_logging(log_level: str = "INFO"):
    """Configure logging."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def clean_directory(directory: Path, dry_run: bool = False) -> bool:
    """Remove a directory and its contents."""
    if not directory.exists():
        logging.info(f"Directory does not exist (skipping): {directory}")
        return False

    try:
        if dry_run:
            logging.info(f"Would remove directory: {directory}")
            return True
        else:
            shutil.rmtree(directory)
            logging.info(f"Removed directory: {directory}")
            return True
    except PermissionError:
        logging.error(f"Permission denied when removing: {directory}")
        return False
    except Exception as e:
        logging.error(f"Error removing directory {directory}: {e}")
        return False


def confirm_action(message: str) -> bool:
    """Ask for user confirmation."""
    response = input(f"{message} (y/N): ").strip().lower()
    return response in ('y', 'yes')


def resolve_directories(args) -> Dict[str, List[Path]]:
    """Resolve directories to clean based on arguments."""
    dirs_by_category = {
        "cache": [],
        "summaries": [],
        "embeddings": [],
        "indices": [],
        "results": [],
        "matches": [],  # Add new category for matcher results
    }

    # Base data directory
    data_dir = Path(args.data_dir)

    # Cache directories
    if args.clean_cache or args.clean_all:
        # LLM response cache
        llm_cache_dir = Path(args.llm_cache_dir)

        # Matcher cache
        matcher_cache_dir = Path(args.matcher_cache_dir)

        if args.all_caches:
            # If all_caches is True, we'll clean the parent cache directory
            # Get the parent cache directory (typically ./cache)
            parent_cache_dir = llm_cache_dir.parent
            dirs_by_category["cache"].append(parent_cache_dir)
        else:
            # Add specific cache directories
            dirs_by_category["cache"].append(llm_cache_dir)
            dirs_by_category["cache"].append(matcher_cache_dir)

    # Summaries directories
    if args.clean_summaries or args.clean_all:
        if args.dataset:
            summaries_dir = data_dir / f"{args.dataset}_summaries"
        else:
            summaries_dir = data_dir / "sigir2016/summaries"
        dirs_by_category["summaries"].append(summaries_dir)

    # Embeddings directories
    if args.clean_embeddings or args.clean_all:
        if args.dataset:
            embeddings_dir = data_dir / f"{args.dataset}_embeddings"
            if args.model_name:
                embeddings_dir = embeddings_dir / args.model_name
        else:
            embeddings_dir = data_dir / "sigir2016/summaries_embeddings"
        dirs_by_category["embeddings"].append(embeddings_dir)

    # Indices directories
    if args.clean_indices or args.clean_all:
        indices_dir = data_dir / "sigir2016/indices"
        if args.model_name:
            # Look for specific model indices
            indices_pattern = f"{args.model_name}_*.index*"
            # We'll handle these files individually
            for index_file in data_dir.glob(f"**/indices/{indices_pattern}"):
                dirs_by_category["indices"].append(index_file)
        else:
            dirs_by_category["indices"].append(indices_dir)

    # Results directories
    if args.clean_results or args.clean_all:
        results_dir = data_dir / "sigir2016/results"
        if args.model_name:
            # Look for specific model results
            results_pattern = f"{args.model_name}_*.json"
            # We'll handle these files individually
            for result_file in data_dir.glob(f"**/results/{results_pattern}"):
                dirs_by_category["results"].append(result_file)
        else:
            dirs_by_category["results"].append(results_dir)

    # Matcher results directories
    if args.clean_matches or args.clean_all:
        matches_dir = data_dir / "sigir2016/matched"
        if args.patient_id:
            # Look for specific patient matches
            matches_pattern = f"*{args.patient_id}*.json"
            # We'll handle these files individually
            for match_file in data_dir.glob(f"**/matched/{matches_pattern}"):
                dirs_by_category["matches"].append(match_file)
        else:
            dirs_by_category["matches"].append(matches_dir)

    return dirs_by_category


def clean_run(args):
    """Clean directories based on arguments."""
    # Resolve directories to clean
    dirs_by_category = resolve_directories(args)

    # Flatten all directories to clean
    all_dirs = []
    for category, dirs in dirs_by_category.items():
        all_dirs.extend(dirs)

    if not all_dirs:
        logging.warning("No directories selected for cleaning.")
        return

    # Confirm action if not forced
    if not args.force and not args.dry_run:
        print("\nDirectories to clean:")
        for category, dirs in dirs_by_category.items():
            if dirs:
                print(f"\n{category.upper()}:")
                for d in dirs:
                    print(f"  - {d}")

        if not confirm_action("\nDo you want to continue?"):
            logging.info("Operation cancelled by user.")
            return

    # Clean directories
    for category, dirs in dirs_by_category.items():
        if dirs:
            logging.info(f"Cleaning {category} directories...")
            for directory in dirs:
                if directory.is_file():
                    # Handle individual files (like indices and results)
                    try:
                        if args.dry_run:
                            logging.info(f"Would remove file: {directory}")
                        else:
                            directory.unlink()
                            logging.info(f"Removed file: {directory}")
                    except Exception as e:
                        logging.error(f"Error removing file {directory}: {e}")
                else:
                    # Handle directories
                    clean_directory(directory, args.dry_run)


def main():
    """Command-line interface for cleaning runs."""
    parser = argparse.ArgumentParser(
        description="Clean TrialMesh directories for cache, summaries, embeddings, indices, results, and matches"
    )

    # Base directories
    parser.add_argument(
        "--data-dir", type=str, default="./data",help="Base data directory (default: ./data)")
    parser.add_argument("--llm-cache-dir",type=str,default="./cache/llm_responses",
                        help="LLM responses cache directory (default: ./cache/llm_responses)")
    parser.add_argument("--matcher-cache-dir",type=str,default="./cache/matcher",
                        help="Matcher cache directory (default: ./cache/matcher)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (e.g., sigir2016/processed_cut) ")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Embedding model name for selective cleaning (e.g., SapBERT) ")
    parser.add_argument("--patient-id", type=str, default=None, help="Patient ID for selective match cleaning ")

    # Cleaning categories
    parser.add_argument("--clean-cache", action="store_true", help="Clean LLM and matcher response caches ")
    parser.add_argument("--clean-summaries", action="store_true", help="Clean LLM-generated summaries ")
    parser.add_argument("--clean-embeddings", action="store_true", help="Clean vector embeddings ")
    parser.add_argument("--clean-indices", action="store_true", help="Clean FAISS indices ")
    parser.add_argument("--clean-results", action="store_true", help="Clean search results ")
    parser.add_argument("--clean-matches", action="store_true", help="Clean trial matcher results ")
    parser.add_argument("--clean-all", action="store_true",
                        help="Clean all categories (cache, summaries, embeddings, indices, results, matches) ")

    # General options
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without actually removing ")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt ")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level ")

    args = parser.parse_args()
    setup_logging(args.log_level)

    # If no cleaning option specified, default to basic clean
    if not (args.clean_cache or args.clean_summaries or args.clean_embeddings or
            args.clean_indices or args.clean_results or args.clean_matches or args.clean_all):
        args.clean_cache = True
        args.clean_summaries = True
        logging.info("No specific cleaning option selected, defaulting to clean cache and summaries.")

    clean_run(args)


def cli_main():
    """Entry point for the console script"""
    main()


if __name__ == "__main__":
    main()
```

## src/trialmesh/utils/codemd.py

```python
import os
import sys

def read_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Could not read {path}: {e}"

def get_src_root():
    # src/trialmesh/utils -> src
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    return src_dir

def get_project_root():
    # src/trialmesh/utils -> project root (parent of src)
    return os.path.abspath(os.path.join(get_src_root(), '..'))

def build_tree(root):
    tree_lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_path = os.path.relpath(dirpath, root)
        indent = '    ' * (0 if rel_path == '.' else rel_path.count(os.sep))
        base = '.' if rel_path == '.' else rel_path
        tree_lines.append(f"{indent}{os.path.basename(base)}" if base != '.' else f"{indent}{os.path.basename(root)}")
        for fname in sorted(filenames):
            tree_lines.append(f"{indent}    {fname}")
    return '\n'.join(tree_lines)

def collect_python_files(root):
    py_files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            if fname.endswith('.py'):
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, os.path.dirname(root))
                py_files.append((rel_path, read_file(full_path)))
    return py_files

def generate_codemd():
    src_dir = get_src_root()
    project_root = get_project_root()
    readme_path = os.path.join(project_root, 'README.md')
    pyproject_path = os.path.join(project_root, 'pyproject.toml')
    output_path = os.path.join(project_root, 'codecomplete.md')

    readme = read_file(readme_path)
    pyproject = read_file(pyproject_path)
    tree = build_tree(src_dir)
    py_files = collect_python_files(src_dir)

    with open(output_path, 'w', encoding='utf-8') as out:
        out.write("# Project README\n\n")
        out.write(f"```\n{readme}\n```\n\n")

        out.write("# pyproject.toml\n\n")
        out.write(f"```toml\n{pyproject}\n```\n\n")

        out.write("# src Directory Tree\n\n")
        out.write(f"```\n{tree}\n```\n\n")

        out.write("# Python Source Files\n\n")
        for rel_path, content in py_files:
            out.write(f"## {rel_path}\n\n")
            out.write(f"```python\n{content}\n```\n\n")

def cli_main():
    """Entry point for trialmesh-codemd CLI tool."""
    generate_codemd()
    print("codecomplete.md generated at project root.")

if __name__ == "__main__":
    cli_main()
```

## src/trialmesh/utils/prompt_registry.py

```python
# src/trialmesh/utils/prompt_registry.py

class PromptRegistry:
    def __init__(self):
        self.prompts = {
            # Existing prompts
            "patient_summary": self._patient_summary(),
            "trial_summary": self._trial_summary(),
            "patient_condensed": self._patient_condensed(),
            "trial_condensed": self._trial_condensed(),
            "exclusion_filter": self._exclusion_filter(),
            "inclusion_filter": self._inclusion_filter(),
            "final_match_scoring": self._final_match_scoring(),

            # New SIGIR2016-specific prompts
            "patient_summary_sigir2016": self._patient_summary_sigir2016(),
            "patient_condensed_sigir2016": self._patient_condensed_sigir2016(),
            "trial_summary_sigir2016": self._trial_summary_sigir2016(),
            "trial_condensed_sigir2016": self._trial_condensed_sigir2016(),
        }

    def get(self, name: str) -> dict:
        """Get prompt pair (system and user) by name."""
        return self.prompts.get(name, {"system": "", "user": ""})

    def get_system(self, name: str) -> str:
        """Get just the system prompt for a template."""
        prompt_pair = self.prompts.get(name, {})
        return prompt_pair.get("system", "")

    def get_user(self, name: str) -> str:
        """Get just the user prompt for a template."""
        prompt_pair = self.prompts.get(name, {})
        return prompt_pair.get("user", "")

    @staticmethod
    def _patient_summary():
        return {
            "system": "You are a clinical assistant specializing in extracting structured patient information for clinical trial matching. Provide clear, concise, and medically accurate summaries.",
            "user": """
    Given the following patient medical history, organize the key information into CLEARLY LABELED SECTIONS.
    
    Patient Text:
    \"\"\"
    {patient_text}
    \"\"\"
    
    For each section, include all relevant information from the patient record:
    
    DIAGNOSIS:
    - List each cancer diagnosis with dates
    - Include stage, grade, and anatomical location
    
    BIOMARKERS:
    - List all genomic mutations (e.g., BRAF V600E, EGFR exon 19 deletion)
    - Include expression status (e.g., PD-L1 80%, HER2 3+)
    
    TREATMENT HISTORY:
    - List all cancer therapies with approximate dates
    - Include response and reason for discontinuation when available
    
    RELEVANT MEDICAL HISTORY:
    - Major comorbidities or conditions
    - Organ function issues
    - History that might affect trial eligibility
    
    CURRENT STATUS:
    - Performance status if mentioned
    - Current symptoms
    - Disease progression status
    
    Provide concise, bulleted information without adding details not present in the original text.
    """
        }

    @staticmethod
    def _patient_condensed():
        return {
            "system": "You are a clinical data specialist creating concise cancer patient summaries for AI-based trial matching systems. Focus on positive patient characteristics.",
            "user": """
    Generate a CONDENSED SUMMARY (UNDER 300 WORDS) of this cancer patient that contains ONLY information relevant for matching to appropriate clinical trials.

    Patient Record:
    \"\"\"
    {patient_text}
    \"\"\"

    Focus EXCLUSIVELY on:
    1. Primary and secondary cancer diagnoses with staging and histology
    2. Biomarkers and genetic mutations with expression levels
    3. Patient characteristics (gender, performance status)
    4. Prior cancer treatments with responses
    5. Current disease status and treatment needs
    6. Relevant organ function status affecting eligibility

    IMPORTANT INSTRUCTIONS:
    - Focus on POSITIVE attributes that would qualify the patient for trials
    - DO NOT include specific numeric thresholds - instead use descriptive terms
      * Instead of "52 years old" use "adult"
      * Instead of "PD-L1 expression 80%" use "high PD-L1 expression"
      * Instead of "creatinine 1.1 mg/dL" use "normal kidney function"
      * Instead of "ECOG 1" use "good performance status"
    - Include clinical significance rather than raw measurements
    - Focus on actionable cancer characteristics for trial matching

    Write as a continuous informative yet concise paragraph optimized for semantic embedding model matching with clinical trial criteria.
    """
        }

    @staticmethod
    def _trial_summary():
        return {
            "system": "You are a clinical research coordinator specializing in translating complex trial protocols into structured eligibility criteria.",
            "user": """
    Given the following trial description, organize the key information into CLEARLY LABELED SECTIONS to match patient profiles.
    
    Trial Text:
    \"\"\"
    {trial_text}
    \"\"\"
    
    Summarize the trial's target population and requirements:
    
    DIAGNOSIS:
    - Target cancer types
    - Required stage or disease status
    - Specific histology requirements
    
    BIOMARKERS:
    - Required biomarker status
    - Genetic mutations of interest
    - Expression level thresholds
    
    TREATMENT HISTORY:
    - Required prior therapies
    - Line of therapy requirements
    - Treatment-free interval requirements
    
    EXCLUSION CRITERIA:
    - Key medical conditions that disqualify patients
    - Prohibited concurrent medications
    - Organ function requirements
    
    INCLUSION CRITERIA:
    - Age and performance status requirements
    - Laboratory value thresholds
    - Other key eligibility factors
    
    Use clear, concise bullets focusing on objective criteria. Follow the exact section structure provided.
    """
        }

    @staticmethod
    def _trial_condensed():
        return {
            "system": "You are a trial eligibility specialist creating concise summaries for AI-based oncology matching systems. Focus exclusively on positive eligibility characteristics.",
            "user": """
    Generate a CONDENSED SUMMARY (UNDER 300 WORDS) of this clinical trial that focuses ONLY on what makes an ideal cancer patient match.

    Trial Description:
    \"\"\"
    {trial_text}
    \"\"\"

    Focus EXCLUSIVELY on:
    1. Target cancer type(s) with specific histology and staging requirements
    2. Required biomarkers and genetic mutations with expression levels
    3. Patient characteristics (gender, performance status)
    4. Prior cancer therapy requirements or preferences
    5. Essential inclusion criteria that qualify patients
    6. Unique eligibility factors specific to this oncology trial

    IMPORTANT INSTRUCTIONS:
    - DO NOT mention any exclusion criteria or negative characteristics
    - DO NOT include specific numeric thresholds - instead use descriptive terms
      * Instead of "age > 18" use "adults"
      * Instead of "ANC > 1.5 × 10⁹/L" use "adequate neutrophil count"
      * Instead of "creatinine < 1.5 mg/dL" use "normal kidney function"
      * Instead of "ECOG 0-1" use "good performance status"
    - Focus only on the positive attributes of ideal candidates

    Write as a continuous informative yet concise paragraph optimized for semantic embedding model matching with cancer patient records.
    """
        }

    @staticmethod
    def _exclusion_filter():
        return {
            "system": "You are a clinical trial screening specialist with expertise in evaluating exclusion criteria. Your task is to determine if a patient is definitively excluded from a trial based on the exclusion criteria. Be thorough in your assessment but only exclude when clear evidence exists.",
            "user": """
    Carefully review the trial's exclusion criteria against the patient's profile and determine if the patient should be excluded.

    Trial Exclusion Criteria:
    \"\"\"
    {exclusion_criteria}
    \"\"\"

    Patient Summary:
    \"\"\"
    {patient_summary}
    \"\"\"

    INSTRUCTIONS:
    1. Review each exclusion criterion carefully
    2. Compare against the patient's information
    3. Only mark as EXCLUDE if there is clear evidence the patient meets an exclusion criterion
    4. If information is missing to determine exclusion, give the patient the benefit of the doubt

    Return your analysis in this EXACT format:

    VERDICT: [EXCLUDE or PASS]

    REASON:
    [If EXCLUDE, explain specifically which exclusion criterion the patient meets and why]
    [If PASS, explain "Patient does not meet any exclusion criteria" or "Insufficient evidence to confirm any exclusion criteria"]
    """
        }

    @staticmethod
    def _inclusion_filter():
        return {
            "system": "You are a clinical trial eligibility specialist evaluating inclusion criteria. Your task is to determine if a patient is included to a trial based on the inclusion criteria. Be thorough in your assessment.",
            "user": """
    Compare the trial inclusion criteria to the patient's profile to determine if required inclusion conditions are met.
    
    Trial Inclusion Criteria:
    \"\"\"
    {inclusion_criteria}
    \"\"\"
    
    Patient Summary:
    \"\"\"
    {patient_summary}
    \"\"\"
    
    INSTRUCTIONS:
    1. Review each inclusion criterion carefully
    2. Compare against the patient's information
    3. Only mark as FAIL if there is clear evidence
    4. If information is missing to determine inclusion, give the patient the benefit of the doubt

    Return your analysis in this EXACT format:
    
    VERDICT: [INCLUDE, UNDETERMINED, or FAIL]
    - INCLUDE: All inclusion criteria are definitively met
    - UNDETERMINED: Cannot determine eligibility due to missing information
    - FAIL: One or more inclusion criteria are definitively not met
    
    MISSING INFORMATION:
    [If UNDETERMINED, list specific information needed to make a determination]
    [If INCLUDE or FAIL, write "None"]
    
    UNMET CRITERIA:
    [If FAIL, list which specific inclusion criteria are not met]
    [If INCLUDE or UNDETERMINED, write "None" or list potential concerns]
    
    REASONING:
    [Brief explanation of your decision, highlighting key factors]
    """
        }

    @staticmethod
    def _final_match_scoring():
        return {
            "system": "You are a clinical match evaluator with extensive experience in oncology trials. Provide balanced, evidence-based assessments of patient-trial compatibility focused on clinical significance rather than numerical values.",
            "user": """
    Carefully analyze how well the patient matches the trial requirements, considering diagnosis alignment, biomarkers, prior treatments, and inclusion/exclusion criteria.

    Patient Profile:
    \"\"\"
    {patient_summary}
    \"\"\"

    Trial Description:
    \"\"\"
    {trial_summary}
    \"\"\"

    Evaluate the match and provide your assessment in this EXACT format:

    SCORE: [0-100]
    [0-29: Unsuitable - Major disqualifying factors]
    [30-59: Weak match - Significant concerns or missing requirements]
    [60-79: Possible match - Generally good fit with minor issues]
    [80-100: Strong match - Excellent alignment with requirements]

    VERDICT: [STRONG MATCH, POSSIBLE MATCH, WEAK MATCH, or UNSUITABLE]

    MATCHING FACTORS:
    - List 2-3 key factors where the patient clearly meets trial requirements
    - Focus on the most clinically significant matching elements

    CONCERNS:
    - List any factors where the patient may not meet requirements
    - Include potential exclusion criteria that might apply
    - Mention any critical missing information needed for assessment

    CLINICAL REASONING:
    Provide a concise clinical explanation of your overall assessment, focusing on:
    1. Disease alignment (type, stage, histology)
    2. Biomarker/genetic requirements
    3. Prior treatment compatibility
    4. Performance status and organ function eligibility

    Base your assessment on clinical significance rather than raw numerical values. Consider the totality of evidence and prioritize factors that would most impact the patient's eligibility and potential benefit.
    """
        }

    @staticmethod
    def _patient_summary_sigir2016():
        return {
            "system": "You are a clinical assistant specializing in extracting structured information from patient cases across all medical specialties. Provide clear, concise, and medically accurate summaries.",
            "user": """
    Given the following patient medical history, organize the key information into CLEARLY LABELED SECTIONS.

    Patient Text:
    \"\"\"
    {patient_text}
    \"\"\"

    For each section, include all relevant information from the patient record:

    CHIEF COMPLAINT:
    - Main reason for visit/primary symptoms
    - Duration and severity

    VITAL SIGNS:
    - Any measured vitals (temperature, blood pressure, heart rate, respiratory rate, etc.)

    KEY FINDINGS:
    - Significant physical examination findings
    - Relevant laboratory or imaging results

    MEDICAL HISTORY:
    - Existing conditions
    - Previous surgeries or hospitalizations
    - Risk factors (smoking, travel history, exposures)

    CURRENT MEDICATIONS:
    - Prescribed medications
    - Over-the-counter medications

    DEMOGRAPHICS:
    - Age, gender, and other relevant demographic information

    Provide concise, bulleted information without adding details not present in the original text.
    """
        }

    @staticmethod
    def _patient_condensed_sigir2016():
        return {
            "system": "You are a clinical data specialist creating concise patient summaries for AI-based trial matching systems. Focus on positive patient characteristics.",
            "user": """
    Generate a CONDENSED SUMMARY (UNDER 300 WORDS) of this patient that contains ONLY information relevant for matching to appropriate clinical trials.

    Patient Record:
    \"\"\"
    {patient_text}
    \"\"\"

    Focus EXCLUSIVELY on:
    1. Primary medical condition(s) with specific subtypes and staging
    2. Patient characteristics (gender, performance status)
    3. Relevant clinical findings and diagnostic results
    4. Prior treatments received and current therapy status
    5. Significant medical history that affects eligibility for interventions

    IMPORTANT INSTRUCTIONS:
    - Focus on POSITIVE attributes that would qualify the patient for trials
    - DO NOT include specific numeric thresholds - instead use descriptive terms
      * Instead of "57 years old" use "adult"
      * Instead of "hemoglobin 9.2 g/dL" use "mild anemia" or "slightly low hemoglobin"
      * Instead of "creatinine 1.2 mg/dL" use "near-normal kidney function"
      * Instead of "ECOG 1" use "good performance status"
    - Include clinical significance rather than raw measurements

    Write as a continuous informative yet concise paragraph optimized for semantic embedding model matching with clinical trial criteria.
    """
        }

    @staticmethod
    def _trial_summary_sigir2016():
        return {
            "system": "You are a clinical research coordinator specializing in translating complex trial protocols into structured eligibility criteria for all medical specialties.",
            "user": """
    Given the following trial description, organize the key information into CLEARLY LABELED SECTIONS to match potential participants.

    Trial Text:
    \"\"\"
    {trial_text}
    \"\"\"

    Summarize the trial's target population and requirements:

    TARGET CONDITION:
    - Primary condition being studied
    - Disease stage or severity requirements

    DEMOGRAPHICS:
    - Age range requirements
    - Gender specifications
    - Other demographic criteria

    INCLUSION REQUIREMENTS:
    - Key symptoms or diagnostic criteria
    - Laboratory or test result thresholds
    - Specific medical history requirements

    EXCLUSION CRITERIA:
    - Disqualifying comorbidities
    - Prohibited medications or therapies
    - Other factors that would exclude patients

    INTERVENTION DETAILS:
    - Brief description of the intervention
    - Dosing, frequency, or administration details if relevant

    Use clear, concise bullets focusing on objective criteria. Follow the exact section structure provided.
    """
        }

    @staticmethod
    def _trial_condensed_sigir2016():
        return {
            "system": "You are a clinical trial eligibility specialist. Focus exclusively on the positive eligibility characteristics.",
            "user": """
    Generate a CONDENSED SUMMARY (UNDER 300 WORDS) of this clinical trial that focuses ONLY on what makes an ideal candidate match.

    Trial Description:
    \"\"\"
    {trial_text}
    \"\"\"

    Focus EXCLUSIVELY on:
    1. Target medical condition(s) with specific subtypes and staging
    2. Required patient characteristics (gender, performance status)
    3. Essential inclusion criteria that most participants must meet
    4. Prior treatments that are required or preferred
    5. Any unique eligibility factors specific to this trial

    IMPORTANT INSTRUCTIONS:
    - DO NOT mention any exclusion criteria or negative characteristics
    - DO NOT include specific numeric thresholds - instead use descriptive terms
      * Instead of "age > 18" use "adults"
      * Instead of "hemoglobin > 9 g/dL" use "adequate hemoglobin levels"
      * Instead of "creatinine < 1.5 mg/dL" use "normal kidney function"
      * Instead of "ECOG 0-2" use "good performance status"
    - Focus only on the positive attributes of ideal candidates

    Write as a continuous informative yet concise paragraph optimized for semantic embedding model matching with patient records.
    """
        }


```

## src/trialmesh/cli/__init__.py

```python
# src/trialmesh/cli/__init__.py
```

## src/trialmesh/cli/download_models.py

```python
#!/usr/bin/env python3
# src/trialmesh/cli/download_models.py

import argparse
import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import concurrent.futures
import sys

# Define model groups
LLM_MODELS = {
    "llama-4-scout-17b-quantized": {
        "repo_id": "RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16",
        "local_dir": "Llama-4-Scout-17B-16E-Instruct-quantized.w4a16"
    },
    "llama-4-scout-17b-fp8": {
        "repo_id": "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",
        "local_dir": "Llama-4-Scout-17B-16E-Instruct-FP8-dynamic"
    },
    "llama-3.3-70b": {
        "repo_id": "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
        "local_dir": "Llama-3.3-70B-Instruct-FP8-dynamic"
    },
    "llama-3.2-3b": {
        "repo_id": "RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic",
        "local_dir": "Llama-3.2-3B-Instruct-FP8-dynamic"
    }
}

EMBEDDING_MODELS = {
    "e5-large-v2": {
        "repo_id": "intfloat/e5-large-v2",
        "local_dir": "e5-large-v2"
    },
    "bge-large-v1.5": {
        "repo_id": "BAAI/bge-large-en-v1.5",
        "local_dir": "bge-large-en-v1.5"
    },
    "sapbert": {
        "repo_id": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "local_dir": "SapBERT"
    },
    "bio_clinicalbert": {
        "repo_id": "emilyalsentzer/Bio_ClinicalBERT",
        "local_dir": "Bio_ClinicalBERT"
    },
    "bluebert": {
        "repo_id": "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16",
        "local_dir": "bluebert"
    }
}

# Combine all models
ALL_MODELS = {**LLM_MODELS, **EMBEDDING_MODELS}


def setup_logging(log_level: str = "INFO"):
    """Configure logging."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download models for TrialMesh from HuggingFace"
    )

    # Output directory
    parser.add_argument( "--output-dir", type=str, default="./models",
                         help="Base directory to store downloaded models (default: ./models) ")

    # Model selection groups
    group = parser.add_mutually_exclusive_group()
    group.add_argument( "--all", action="store_true",
                        help="Download all models (LLMs and embedding models) ")
    group.add_argument( "--llms", action="store_true", help="Download only LLM models ")
    group.add_argument( "--embeddings", action="store_true", help="Download only embedding models ")
    group.add_argument( "--models", type=str, nargs="+",
                        help="Specific models to download (space separated list) ")

    # List available models
    parser.add_argument( "--list", action="store_true",
                         help="List available models without downloading ")

    # Concurrent downloads
    parser.add_argument( "--workers", type=int, default=1,
                         help="Number of concurrent downloads (default: 1) ")

    # Force download even if exists
    parser.add_argument( "--force", action="store_true",
                         help="Force download even if model directory already exists ")

    # Resume incomplete downloads
    parser.add_argument( "--resume", action="store_true",
                         help="Resume incomplete downloads (where possible) ")

    # General options
    parser.add_argument( "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                         help="Set the logging level ")

    args = parser.parse_args()

    # Default to all models if no selection is made
    if not (args.all or args.llms or args.embeddings or args.models or args.list):
        args.all = True
        logging.info("No specific model group selected, defaulting to download all models")

    return args


def list_available_models():
    """Display available models in a formatted table."""
    print("\nAvailable Models:\n")

    print("LLM Models:")
    print("-" * 80)
    print(f"{'Short Name':<25} {'Repository':<50}")
    print("-" * 80)
    for name, details in LLM_MODELS.items():
        print(f"{name:<25} {details['repo_id']:<50}")

    print("\nEmbedding Models:")
    print("-" * 80)
    print(f"{'Short Name':<25} {'Repository':<50}")
    print("-" * 80)
    for name, details in EMBEDDING_MODELS.items():
        print(f"{name:<25} {details['repo_id']:<50}")

    print("\nTo download specific models, use: --models model1 model2 ...")


def download_model(
        model_name: str,
        repo_id: str,
        output_dir: str,
        local_dir: str = None,
        force: bool = False,
        resume: bool = False
) -> Tuple[str, bool]:
    """Download a model from HuggingFace."""
    local_dir = local_dir or model_name
    model_dir = os.path.join(output_dir, local_dir)

    # Check if model already exists
    if os.path.exists(model_dir) and not force:
        logging.info(f"Model directory {model_dir} already exists, skipping download")
        return model_name, True

    # Create parent directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Build command
    cmd = ["huggingface-cli", "download", repo_id, "--local-dir", model_dir]

    # Add resume flag if requested
    if resume:
        cmd.append("--resume-download")

    try:
        logging.info(f"Downloading {model_name} from {repo_id} to {model_dir}")
        process = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        logging.debug(process.stdout)
        logging.info(f"Successfully downloaded {model_name}")
        return model_name, True

    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to download {model_name}: {e}")
        logging.error(f"Error output: {e.stderr}")
        return model_name, False

    except Exception as e:
        logging.error(f"Unexpected error downloading {model_name}: {e}")
        return model_name, False


def download_models(args):
    """Download selected models based on arguments."""
    # Handle --list flag first
    if args.list:
        list_available_models()
        return

    # Determine which models to download
    models_to_download = {}

    if args.models:
        # Download specific models by name
        invalid_models = []
        for model_name in args.models:
            if model_name in ALL_MODELS:
                models_to_download[model_name] = ALL_MODELS[model_name]
            else:
                invalid_models.append(model_name)

        if invalid_models:
            logging.error(f"Unknown model(s): {', '.join(invalid_models)}")
            logging.error("Use --list to see available models")
            return

    elif args.all:
        # Download all models
        models_to_download = ALL_MODELS

    elif args.llms:
        # Download only LLMs
        models_to_download = LLM_MODELS

    elif args.embeddings:
        # Download only embedding models
        models_to_download = EMBEDDING_MODELS

    if not models_to_download:
        logging.error("No models selected for download")
        return

    logging.info(f"Preparing to download {len(models_to_download)} models to {args.output_dir}")

    # Download models (potentially in parallel)
    successful = []
    failed = []

    if args.workers > 1 and len(models_to_download) > 1:
        # Parallel downloads
        logging.info(f"Using {args.workers} workers for parallel downloads")

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    download_model,
                    model_name,
                    details["repo_id"],
                    args.output_dir,
                    details["local_dir"],
                    args.force,
                    args.resume
                ): model_name
                for model_name, details in models_to_download.items()
            }

            for future in concurrent.futures.as_completed(futures):
                model_name, success = future.result()
                if success:
                    successful.append(model_name)
                else:
                    failed.append(model_name)

                # Progress report
                completed = len(successful) + len(failed)
                total = len(models_to_download)
                logging.info(f"Progress: {completed}/{total} models processed")

    else:
        # Sequential downloads
        for i, (model_name, details) in enumerate(models_to_download.items(), 1):
            logging.info(f"Processing model {i}/{len(models_to_download)}: {model_name}")

            _, success = download_model(
                model_name,
                details["repo_id"],
                args.output_dir,
                details["local_dir"],
                args.force,
                args.resume
            )

            if success:
                successful.append(model_name)
            else:
                failed.append(model_name)

    # Print summary
    logging.info("=" * 80)
    logging.info("Download Summary")
    logging.info("=" * 80)

    if successful:
        logging.info(f"Successfully downloaded {len(successful)} models:")
        for model in successful:
            logging.info(f"  - {model}")

    if failed:
        logging.error(f"Failed to download {len(failed)} models:")
        for model in failed:
            logging.error(f"  - {model}")

    logging.info(f"Models are available in: {args.output_dir}")


def check_huggingface_cli():
    """Check if huggingface-cli is available."""
    try:
        subprocess.run(
            ["huggingface-cli", "--help"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def main():
    """Main entry point for the script."""
    args = parse_args()
    setup_logging(args.log_level)

    # Check for huggingface-cli
    if not check_huggingface_cli():
        logging.error("huggingface-cli not found. Please install huggingface_hub:")
        logging.error("  pip install huggingface_hub")
        sys.exit(1)

    download_models(args)


def cli_main():
    """Entry point for console script."""
    main()


if __name__ == "__main__":
    main()
```

## src/trialmesh/cli/run_matcher.py

```python
#!/usr/bin/env python3
# src/trialmesh/cli/run_matcher.py

import argparse
import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from trialmesh.match.matcher import TrialMatcher
from trialmesh.llm.llama_runner import LlamaRunner


def setup_logging(log_level: str = "INFO"):
    """Configure logging."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run trial-patient matching with LLM evaluation"
    )

    # Model configuration
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the LLaMA model")
    parser.add_argument("--tensor-parallel-size", type=int, default=4,
                        help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Maximum sequence length for output")
    parser.add_argument("--max-model-len", type=int, default=2048,
                        help="Maximum context length for Input+output")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for processing")

    # Data paths
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Base data directory")
    parser.add_argument("--search-results", type=str,
                        default="sigir2016/results/bge-large-en-v1.5_hnsw_search_results.json",
                        help="Path to search results JSON relative to data-dir")
    parser.add_argument("--patient-summaries", type=str,
                        default="sigir2016/summaries/patient_summaries.jsonl",
                        help="Path to patient summaries JSONL relative to data-dir")
    parser.add_argument("--trials-path", type=str,
                        default="sigir2016/processed_cut/corpus.jsonl",
                        help="Path to trials JSONL relative to data-dir")
    parser.add_argument("--output-file", type=str,
                        default="sigir2016/matched/trial_matches.json",
                        help="Output file path relative to data-dir")
    parser.add_argument("--cache-dir", type=str, default="./cache/matcher",
                        help="Cache directory for LLM responses")

    # Processing options
    parser.add_argument("--include-all-trials", action="store_true",
                        help="Include all trials in output, even those filtered out at early stages")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Maximum number of trials to evaluate per patient (None for all trials)")
    parser.add_argument("--skip-exclusion", action="store_true",
                        help="Skip exclusion filter step")
    parser.add_argument("--skip-inclusion", action="store_true",
                        help="Skip inclusion filter step")
    parser.add_argument("--skip-scoring", action="store_true",
                        help="Skip final scoring step")
    parser.add_argument("--patient-ids", type=str, nargs="+",
                        help="Specific patient IDs to process")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")

    args = parser.parse_args()
    return args


def main():
    """Main entry point for the script."""
    args = parse_args()
    setup_logging(args.log_level)

    # Create output directories
    output_path = os.path.join(args.data_dir, args.output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize LLM
    llm = LlamaRunner(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        max_batch_size=args.batch_size,
    )

    # Initialize the matcher
    matcher = TrialMatcher(
        llm=llm,
        data_dir=args.data_dir,
        patient_summaries_path=args.patient_summaries,
        trials_path=args.trials_path,
        batch_size=args.batch_size,
    )

    # Load search results
    search_results_path = os.path.join(args.data_dir, args.search_results)
    logging.info(f"Loading search results from {search_results_path}")
    with open(search_results_path, 'r') as f:
        search_results = json.load(f)

    # Filter patients if specified
    if args.patient_ids:
        search_results = [r for r in search_results if r["query_id"] in args.patient_ids]
        if not search_results:
            logging.error(f"No matching patients found for IDs: {args.patient_ids}")
            return
        logging.info(f"Processing {len(search_results)} patients based on provided IDs")
    else:
        logging.info(f"Processing all {len(search_results)} patients from search results")

    # Run the matching process
    results = matcher.match(
        search_results=search_results,
        top_k=args.top_k,
        skip_exclusion=args.skip_exclusion,
        skip_inclusion=args.skip_inclusion,
        skip_scoring=args.skip_scoring,
        include_all_trials=args.include_all_trials,
    )

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"Saved matching results to {output_path}")


def cli_main():
    """Entry point for the console script"""
    main()


if __name__ == "__main__":
    main()
```

## src/trialmesh/cli/run_retrieval_pipeline.py

```python
#!/usr/bin/env python3
# src/trialmesh/cli/run_retrieval_pipeline.py

import argparse
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# Define default parameters that match the shell script
DEFAULT_MODELS = [
    "SapBERT",
    "bge-large-en-v1.5",
    "Bio_ClinicalBERT",
    "bluebert",
    "e5-large-v2"
]

# Model-specific batch sizes
DEFAULT_BATCH_SIZES = {
    "SapBERT": 256,
    "bge-large-en-v1.5": 128,
    "Bio_ClinicalBERT": 256,
    "bluebert": 256,
    "e5-large-v2": 128
}

# HNSW parameters
DEFAULT_M_VALUE = 64
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_K_VALUE = 300


def setup_logging(log_level: str = "INFO"):
    """Configure logging."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the complete TrialMesh pipeline (embedding, indexing, search)")

    # Base directories
    parser.add_argument( "--data-dir", type=str, default="./data", help="Base data directory (default: ./data) ")
    parser.add_argument( "--models-dir", type=str, default=None, help="Directory containing embedding models ")
    parser.add_argument( "--dataset", type=str, default="sigir2016/summaries",
                         help="Dataset to process (default: sigir2016/summaries) ")

    # Model selection
    parser.add_argument( "--models", type=str, nargs="+", default=None, help=f"Models to process (default: {DEFAULT_MODELS}) ")
    parser.add_argument( "--single-model", type=str, default=None, help="Run only one model (overrides --models) ")

    # Index configuration
    parser.add_argument( "--index-type", type=str, choices=["flat", "hnsw"], default="hnsw",
                         help="Type of FAISS index to build (default: hnsw) ")

    # HNSW parameters
    parser.add_argument( "--m-value", type=int, default=DEFAULT_M_VALUE,
                         help=f"M parameter for HNSW index (default: {DEFAULT_M_VALUE}) ")
    parser.add_argument( "--ef-construction", type=int, default=DEFAULT_EF_CONSTRUCTION,
                         help=f"EF construction parameter for HNSW index (default: {DEFAULT_EF_CONSTRUCTION}) ")
    parser.add_argument( "--k-value", type=int, default=DEFAULT_K_VALUE,
                         help=f"Number of results to return per query (default: {DEFAULT_K_VALUE}) ")

    # Processing options
    parser.add_argument( "--skip-embeddings", action="store_true", help="Skip embedding generation (use existing embeddings) ")
    parser.add_argument( "--skip-indexing", action="store_true", help="Skip index building (use existing indices) ")
    parser.add_argument( "--skip-search", action="store_true", help="Skip search execution ")
    parser.add_argument( "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                         help="Set the logging level ")

    args = parser.parse_args()

    # Handle models selection
    if args.single_model:
        args.models = [args.single_model]
    elif args.models is None:
        args.models = DEFAULT_MODELS

    # Check if models directory is provided or can be inferred
    if not args.models_dir:
        # Try common locations
        potential_dirs = [
            os.path.expanduser("~/models"),
            os.path.expanduser("~/deepNets/models"),
            "/models",
            "/data/models"
        ]
        for dir_path in potential_dirs:
            if os.path.isdir(dir_path):
                args.models_dir = dir_path
                logging.info(f"Inferred models directory: {args.models_dir}")
                break

        if not args.models_dir:
            parser.error("Could not infer models directory. Please specify --models-dir.")

    # Create output directories
    os.makedirs(os.path.join(args.data_dir, "sigir2016", "indices"), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, "sigir2016", "results"), exist_ok=True)

    return args


def run_embedding(
        model_name: str,
        model_path: str,
        data_dir: str,
        dataset: str,
        batch_size: int,
) -> bool:
    """Run embedding generation for a single model."""
    logging.info(f"Generating embeddings for {model_name}...")

    cmd = [
        "trialmesh-embed",
        "--model-path", model_path,
        "--batch-size", str(batch_size),
        "--normalize",
        "--data-dir", data_dir,
        "--dataset", dataset
    ]

    try:
        # Don't capture stdout/stderr - let them flow to console
        process = subprocess.run(
            cmd,
            check=True,
            text=True
        )

        # Log command output at debug level
        logging.debug(process.stdout)

        # Check if the expected output files exist
        embeddings_dir = os.path.join(data_dir, f"{dataset}_embeddings", model_name)
        trial_embeddings = os.path.join(embeddings_dir, "trial_embeddings.npy")
        patient_embeddings = os.path.join(embeddings_dir, "patient_embeddings.npy")

        if not os.path.exists(trial_embeddings) or not os.path.exists(patient_embeddings):
            logging.error(f"Embedding files not found for {model_name}")
            return False

        logging.info(f"Successfully generated embeddings for {model_name}")
        return True

    except subprocess.CalledProcessError as e:
        logging.error(f"Embedding generation failed for {model_name}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during embedding generation for {model_name}: {e}")
        return False


def build_index(
        model_name: str,
        data_dir: str,
        dataset: str,
        index_type: str,
        m_value: int = DEFAULT_M_VALUE,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
) -> bool:
    """Build FAISS index for a model."""
    logging.info(f"Building {index_type} index for {model_name}...")

    # Set paths
    embeddings_dir = os.path.join(data_dir, f"{dataset}_embeddings", model_name)
    trial_embeddings = os.path.join(embeddings_dir, "trial_embeddings.npy")
    index_file = os.path.join(data_dir, "sigir2016", "indices", f"{model_name}_trials_{index_type}.index")

    # Build the command based on index type
    cmd = [
        "trialmesh-index", "build",
        "--embeddings", trial_embeddings,
        "--output", index_file,
        "--index-type", index_type
    ]

    # Add HNSW-specific parameters if needed
    if index_type == "hnsw":
        cmd.extend([
            "--m", str(m_value),
            "--ef-construction", str(ef_construction)
        ])

    try:
        # Execute the command
        process = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Log command output at debug level
        logging.debug(process.stdout)

        # Check if the index file exists
        if not os.path.exists(index_file):
            logging.error(f"Index file not found after building: {index_file}")
            return False

        logging.info(f"Successfully built {index_type} index for {model_name}")
        return True

    except subprocess.CalledProcessError as e:
        logging.error(f"Index building failed for {model_name}: {e}")
        logging.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during index building for {model_name}: {e}")
        return False


def run_search(
        model_name: str,
        data_dir: str,
        dataset: str,
        index_type: str,
        k_value: int = DEFAULT_K_VALUE,
) -> bool:
    """Run vector search for a model."""
    logging.info(f"Running search for {model_name} using {index_type} index...")

    # Set paths
    embeddings_dir = os.path.join(data_dir, f"{dataset}_embeddings", model_name)
    patient_embeddings = os.path.join(embeddings_dir, "patient_embeddings.npy")
    index_file = os.path.join(data_dir, "sigir2016", "indices", f"{model_name}_trials_{index_type}.index")
    results_file = os.path.join(data_dir, "sigir2016", "results", f"{model_name}_{index_type}_search_results.json")

    cmd = [
        "trialmesh-index", "search",
        "--index", index_file,
        "--queries", patient_embeddings,
        "--output", results_file,
        "--k", str(k_value)
    ]

    try:
        # Execute the command
        process = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Log command output at debug level
        logging.debug(process.stdout)

        # Check if the results file exists
        if not os.path.exists(results_file):
            logging.error(f"Results file not found after search: {results_file}")
            return False

        logging.info(f"Successfully ran search for {model_name} with {index_type} index")
        return True

    except subprocess.CalledProcessError as e:
        logging.error(f"Search failed for {model_name}: {e}")
        logging.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during search for {model_name}: {e}")
        return False


def run_pipeline(args):
    """Run the complete pipeline for all specified models."""
    successful_models = []
    failed_models = []

    # Print banner
    logging.info("=" * 80)
    logging.info(f"Starting TrialMesh pipeline with INDEX_TYPE={args.index_type}")
    logging.info("=" * 80)

    for model_name in args.models:
        logging.info("=" * 80)
        logging.info(f"Processing model: {model_name}")
        logging.info("=" * 80)

        model_path = os.path.join(args.models_dir, model_name)
        batch_size = DEFAULT_BATCH_SIZES.get(model_name, 128)

        # 1. Generate embeddings (if not skipped)
        if not args.skip_embeddings:
            if not run_embedding(model_name, model_path, args.data_dir, args.dataset, batch_size):
                logging.error(f"Skipping {model_name} due to embedding generation failure")
                failed_models.append(model_name)
                continue

        # 2. Build index (if not skipped)
        if not args.skip_indexing:
            if not build_index(
                    model_name,
                    args.data_dir,
                    args.dataset,
                    args.index_type,
                    args.m_value,
                    args.ef_construction
            ):
                logging.error(f"Skipping {model_name} due to index building failure")
                failed_models.append(model_name)
                continue

        # 3. Run search (if not skipped)
        if not args.skip_search:
            if not run_search(
                    model_name,
                    args.data_dir,
                    args.dataset,
                    args.index_type,
                    args.k_value
            ):
                logging.error(f"Search failed for {model_name}")
                failed_models.append(model_name)
                continue

        # If we got here, everything succeeded for this model
        successful_models.append(model_name)
        logging.info(f"Pipeline completed successfully for {model_name} with {args.index_type} index!")
        logging.info("")

    # Print summary
    logging.info("=" * 80)
    logging.info(f"All processing complete! Used index type: {args.index_type}")
    logging.info("=" * 80)

    if successful_models:
        logging.info(f"Successful models ({len(successful_models)}): {', '.join(successful_models)}")

    if failed_models:
        logging.warning(f"Failed models ({len(failed_models)}): {', '.join(failed_models)}")

    logging.info("Results available in:")
    logging.info(f"{args.data_dir}/sigir2016/results/")


def main():
    """Main entry point for the script."""
    args = parse_args()
    setup_logging(args.log_level)
    run_pipeline(args)


def cli_main():
    """Entry point for console script."""
    main()


if __name__ == "__main__":
    main()
```

## src/trialmesh/evaluation/__init__.py

```python
# src/trialmesh/evaluation/__init__.py
"""Evaluation tools for TrialMesh."""
```

## src/trialmesh/evaluation/evaluate_results.py

```python
#!/usr/bin/env python3
# src/trialmesh/evaluation/evaluate_results.py

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logging(log_level: str = "INFO"):
    """Configure logging."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate search results against SIGIR2016 gold standard"
    )

    # Data directories
    parser.add_argument( "--data-dir", type=str, default="./data",
                         help="Base data directory (default: ./data) ")
    parser.add_argument( "--dataset", type=str, default="sigir2016/processed_cut",
                         help="Dataset path relative to data-dir (default: sigir2016/processed_cut) ")
    parser.add_argument( "--results-dir", type=str, default=None,
                         help="Results directory (default: {data-dir}/sigir2016/results) ")

    # Models to evaluate
    parser.add_argument( "--models", type=str, nargs="+", default=None,
                         help="Specific models to evaluate (default: all files in results directory) ")

    # Index type (for filtering result files)
    parser.add_argument( "--index-type", type=str, default=None, choices=["flat", "hnsw"],
                         help="Index type to evaluate (optional filter) ")

    # Output options
    parser.add_argument( "--output-file", type=str, default=None, help="Save evaluation results to CSV file ")
    parser.add_argument( "--visualize", action="store_true", help="Generate visualization plots ")

    # General options
    parser.add_argument( "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                         help="Set the logging level ")

    args = parser.parse_args()

    # Set default results directory if not specified
    if args.results_dir is None:
        args.results_dir = os.path.join(args.data_dir, "sigir2016", "results")

    return args


def load_gold_data(data_dir: str, dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load gold standard data files."""
    dataset_path = os.path.join(data_dir, dataset)

    # Load relevance judgments
    tsv_file = os.path.join(dataset_path, "test.tsv")
    logging.info(f"Loading relevance judgments from {tsv_file}")
    df_tsv = pd.read_csv(tsv_file, sep='\t')

    # Load queries
    queries_file = os.path.join(dataset_path, "queries.jsonl")
    logging.info(f"Loading queries from {queries_file}")
    df_queries = pd.read_json(queries_file, lines=True)

    # Load corpus
    corpus_file = os.path.join(dataset_path, "corpus.jsonl")
    logging.info(f"Loading corpus from {corpus_file}")
    df_corpus = pd.read_json(corpus_file, lines=True)

    logging.info(f"df_tsv len:{len(df_tsv)} df_queries len:{len(df_queries)} df_corpus len:{len(df_corpus)}")

    return df_tsv, df_queries, df_corpus


def load_search_results(results_file: str) -> pd.DataFrame:
    """Load search results from a JSON file."""
    logging.info(f"Loading search results from {results_file}")
    try:
        return pd.read_json(results_file)
    except Exception as e:
        logging.error(f"Error loading results file {results_file}: {e}")
        return pd.DataFrame()


def flatten_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the nested results into a long DataFrame."""
    dfs = []
    for _, row in results_df.iterrows():
        for result in row['results']:
            dfs.append({
                'query-id': row['query_id'],
                'corpus-id': result['doc_id'],
                'score': result['score']
            })
    logging.info(f"results_df len:{len(dfs)}")
    return pd.DataFrame(dfs)


def per_query_stats(df_tsv: pd.DataFrame, df_results_long: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-query statistics."""
    df_tsv = df_tsv.copy()
    df_tsv['label_1'] = (df_tsv['score'] == 1).astype(int)
    df_tsv['label_2'] = (df_tsv['score'] == 2).astype(int)

    # Create a set of predicted pairs for fast lookup
    pred_pairs = set(zip(df_results_long['query-id'], df_results_long['corpus-id']))

    # Mark which relevant pairs were found in search results
    df_tsv['found'] = df_tsv.apply(
        lambda row: (row['query-id'], row['corpus-id']) in pred_pairs,
        axis=1
    )

    # Group by query and calculate statistics
    per_query = (
        df_tsv
        .groupby('query-id', as_index=True)
        .agg(
            total_score_1=('label_1', 'sum'),
            total_score_2=('label_2', 'sum'),
            found_score_1=('found', lambda x: int(((x) & (df_tsv.loc[x.index, 'label_1'] == 1)).sum())),
            found_score_2=('found', lambda x: int(((x) & (df_tsv.loc[x.index, 'label_2'] == 1)).sum()))
        )
    )

    # Calculate missing items and percentages
    per_query['missing_score_1'] = per_query['total_score_1'] - per_query['found_score_1']
    per_query['missing_score_2'] = per_query['total_score_2'] - per_query['found_score_2']

    # Calculate percentages, handling division by zero
    per_query['percent_missing_1'] = np.where(
        per_query['total_score_1'] == 0,
        0.0,
        100 * per_query['missing_score_1'] / per_query['total_score_1']
    ).round(1)

    per_query['percent_missing_2'] = np.where(
        per_query['total_score_2'] == 0,
        0.0,
        100 * per_query['missing_score_2'] / per_query['total_score_2']
    ).round(1)

    return per_query


def calculate_metrics(df_tsv: pd.DataFrame, df_results_long: pd.DataFrame) -> Dict:
    """Calculate various evaluation metrics."""
    # Get per-query statistics first
    per_query = per_query_stats(df_tsv, df_results_long)

    # Calculate overall statistics
    total_1 = int(per_query['total_score_1'].sum())
    total_2 = int(per_query['total_score_2'].sum())
    missing_1 = int(per_query['missing_score_1'].sum())
    missing_2 = int(per_query['missing_score_2'].sum())
    found_1 = total_1 - missing_1
    found_2 = total_2 - missing_2

    # Calculate percentages, handling division by zero
    percent_missing_1 = (missing_1 / total_1 * 100) if total_1 > 0 else 0
    percent_missing_2 = (missing_2 / total_2 * 100) if total_2 > 0 else 0
    percent_found_1 = (found_1 / total_1 * 100) if total_1 > 0 else 0
    percent_found_2 = (found_2 / total_2 * 100) if total_2 > 0 else 0

    # Create overall metrics
    metrics = {
        "Total Score 1": total_1,
        "Total Score 2": total_2,
        "Found Score 1": found_1,
        "Found Score 2": found_2,
        "Missing Score 1": missing_1,
        "Missing Score 2": missing_2,
        "Percent Found 1": round(percent_found_1, 1),
        "Percent Found 2": round(percent_found_2, 1),
        "Percent Missing 1": round(percent_missing_1, 1),
        "Percent Missing 2": round(percent_missing_2, 1),
    }

    return metrics


def analyze_model(
        model_name: str,
        results_file: str,
        df_tsv: pd.DataFrame,
        df_queries: pd.DataFrame,
        df_corpus: pd.DataFrame
) -> Dict:
    """Analyze a single model's search results."""
    logging.info(f"Analyzing model: {model_name}")

    # Load and process results
    results_df = load_search_results(results_file)
    if results_df.empty:
        logging.warning(f"Skipping {model_name} due to empty results")
        return None

    df_results_long = flatten_results(results_df)
    metrics = calculate_metrics(df_tsv, df_results_long)

    # Log key metrics
    logging.info(f"  Total Score 1: {metrics['Total Score 1']}")
    logging.info(f"  Total Score 2: {metrics['Total Score 2']}")
    logging.info(f"  Found Score 1: {metrics['Found Score 1']} ({metrics['Percent Found 1']}%)")
    logging.info(f"  Found Score 2: {metrics['Found Score 2']} ({metrics['Percent Found 2']}%)")

    # Return complete results
    return {
        "Model": model_name,
        **metrics
    }


def find_result_files(results_dir: str, models: List[str] = None, index_type: str = None) -> Dict[str, str]:
    """Find result files to evaluate."""
    result_files = {}

    # List all JSON files in the results directory
    for file in os.listdir(results_dir):
        if not file.endswith(".json"):
            continue

        # Skip if index type doesn't match
        if index_type and index_type not in file:
            continue

        # Extract model name from filename
        # Initialize model_name to None first
        model_name = None

        # Format: modelname_indextype_search_results.json or modelname_search_results.json
        file_parts = file.split("_")

        # Try to extract model name based on common patterns
        if len(file_parts) >= 3:
            if "search" in file_parts and "results" in file_parts:
                # Handle case with index type in filename
                if len(file_parts) >= 4 and file_parts[1] in ["flat", "hnsw"]:
                    model_name = file_parts[0]
                else:
                    # No index type in filename
                    model_name = file_parts[0]

        # If we couldn't determine model name, use filename without extension as fallback
        if not model_name:
            model_name = file.rsplit(".", 1)[0]

        # Skip if not in requested models
        if models and model_name not in models:
            continue

        result_files[model_name] = os.path.join(results_dir, file)

    return result_files


def generate_visualizations(summary_df: pd.DataFrame, output_prefix: str = None):
    """Generate visualization plots."""
    try:
        # Set style
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 8))

        # Sort by performance on Score 2 (more relevant)
        plot_df = summary_df.sort_values(by='Percent Found 2', ascending=False)

        # Create bar plot for percentage found
        ax = sns.barplot(
            x='Model',
            y='Percent Found 2',
            data=plot_df,
            palette='viridis',
            label='Score 2 (High Relevance)'
        )

        # Add bars for score 1
        sns.barplot(
            x='Model',
            y='Percent Found 1',
            data=plot_df,
            palette='muted',
            label='Score 1 (Relevant)',
            alpha=0.6
        )

        # Add labels and title
        plt.title('Model Performance: Percent of Relevant Trials Retrieved', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Percent Found (%)', fontsize=14)
        plt.ylim(0, 100)

        # Add legend
        plt.legend()

        # Rotate x labels for readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save or display
        if output_prefix:
            plt.savefig(f"{output_prefix}_performance.png", dpi=300, bbox_inches='tight')
            logging.info(f"Saved visualization to {output_prefix}_performance.png")
        else:
            plt.show()

    except Exception as e:
        logging.error(f"Error generating visualizations: {e}")


def evaluate_models(args):
    """Evaluate all models based on command line arguments."""
    # Load gold standard data
    df_tsv, df_queries, df_corpus = load_gold_data(args.data_dir, args.dataset)

    # Find result files to evaluate
    result_files = find_result_files(args.results_dir, args.models, args.index_type)

    if not result_files:
        logging.error(f"No result files found in {args.results_dir}")
        if args.models:
            logging.error(f"Requested models: {args.models}")
        if args.index_type:
            logging.error(f"Requested index type: {args.index_type}")
        return

    logging.info(f"Found {len(result_files)} result files to evaluate")

    # Analyze each model
    summary_list = []
    for model_name, results_file in result_files.items():
        result = analyze_model(model_name, results_file, df_tsv, df_queries, df_corpus)
        if result:
            summary_list.append(result)

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_list)

    # Sort by performance on Score 2 (more relevant)
    summary_df = summary_df.sort_values(by='Percent Found 2', ascending=False)

    # Display summary table
    print("\n=== Model Performance Summary ===")
    print(summary_df.to_string(index=False))

    # Save to CSV if requested
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        summary_df.to_csv(args.output_file, index=False)
        logging.info(f"Saved evaluation results to {args.output_file}")

    # Generate visualizations if requested
    if args.visualize:
        output_prefix = args.output_file.rsplit('.', 1)[0] if args.output_file else None
        generate_visualizations(summary_df, output_prefix)

    return summary_df


def main():
    """Main entry point for the script."""
    args = parse_args()
    setup_logging(args.log_level)
    evaluate_models(args)


def cli_main():
    """Entry point for console script."""
    main()


if __name__ == "__main__":
    main()
```

## src/trialmesh/fetchers/__init__.py

```python
# src/trialmesh/data/__init__.py
```

## src/trialmesh/fetchers/processxml.py

```python
#!/usr/bin/env python3
# src/trialmesh/data/processxml.py

import argparse
import glob
import json
import logging
import os
import re
import shutil
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from multiprocessing import Pool, cpu_count
import csv
from pathlib import Path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Process SIGIR2016 clinical trial XML files")
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Base data directory (default: ./data)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: {data-dir}/sigir2016/processed)')
    parser.add_argument('--log-level', default='WARNING',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level (default: WARNING)')
    args = parser.parse_args()

    # Set default output dir if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, 'sigir2016', 'processed')

    # Ensure output directory exists for log file
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def setup_logging(args):
    """Set up logging configuration based on user arguments"""
    # Ensure log directory exists first
    log_dir = Path(args.data_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"processxml_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    console_level = getattr(logging, args.log_level)

    # Set up file handler (always DEBUG level)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Set up console handler (level based on user input)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # This ensures all messages are processed
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info(f"Console logging level set to: {logging.getLevelName(console_level)}")
    logging.debug(f"File logging level set to: DEBUG, log file: {log_file}")


def time_function(func_name, start_time):
    """Log the execution time of a function if in debug mode"""
    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
        end_time = time.time()
        logging.debug(f"{func_name} took {end_time - start_time:.2f} seconds to execute.")


def clean_text(text):
    """Clean and normalize text content"""
    if text is None:
        return ""
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Replace Unicode characters with ASCII equivalents
    text = text.replace('\u2264', '<=').replace('\u2265', '>=')
    return text


def process_xml_file(file_path):
    """Extract relevant data from a clinical trial XML file"""
    # Skip macOS metadata files
    if os.path.basename(file_path).startswith('._'):
        return None

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        nct_id = root.find('id_info/nct_id').text
        title = clean_text(root.find('brief_title').text)

        brief_summary = root.find('brief_summary/textblock')
        brief_summary_text = clean_text(brief_summary.text) if brief_summary is not None else ""

        detailed_description = root.find('detailed_description/textblock')
        detailed_description_text = clean_text(detailed_description.text) if detailed_description is not None else ""

        criteria = root.find('.//criteria/textblock')
        criteria_text = clean_text(criteria.text) if criteria is not None else ""

        # Split criteria text into inclusion and exclusion
        inclusion_text = ""
        exclusion_text = ""
        if "Inclusion Criteria:" in criteria_text and "Exclusion Criteria:" in criteria_text:
            parts = criteria_text.split("Exclusion Criteria:")
            inclusion_text = parts[0].split("Inclusion Criteria:")[-1].strip()
            exclusion_text = parts[1].strip()
        elif "Inclusion Criteria:" in criteria_text:
            inclusion_text = criteria_text.split("Inclusion Criteria:")[-1].strip()
        elif "Exclusion Criteria:" in criteria_text:
            exclusion_text = criteria_text.split("Exclusion Criteria:")[-1].strip()

        # Clean up inclusion and exclusion criteria
        inclusion_text = clean_text(inclusion_text)
        exclusion_text = clean_text(exclusion_text)

        enrollment = root.find('enrollment')
        enrollment_value = enrollment.text if enrollment is not None else "0"

        intervention_elements = root.findall('.//intervention/intervention_name')
        drugs_list = [clean_text(elem.text) for elem in intervention_elements if elem.text]

        condition_elements = root.findall('condition')
        diseases_list = [clean_text(elem.text) for elem in condition_elements if elem.text]

        phase_element = root.find('phase')
        phase = clean_text(phase_element.text) if phase_element is not None else ""

        data = {
            "_id": nct_id,
            "title": title,
            "metadata": {
                "phase": phase,
                "drugs": str(drugs_list),
                "drugs_list": drugs_list,
                "diseases_list": diseases_list,
                "enrollment": enrollment_value,
                "inclusion_criteria": inclusion_text,
                "exclusion_criteria": exclusion_text,
                "brief_summary": brief_summary_text,
                "detailed_description": detailed_description_text
            }
        }

        return data

    except ET.ParseError:
        logging.error(f"Invalid XML file: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {str(e)}")
        return None


def copy_and_transform_files(data_dir, output_dir):
    """Transform and copy auxiliary files from the dataset"""
    start_time = time.time()

    source_dir = os.path.join(data_dir, "sigir2016", "data")
    files_to_copy = [
        ("adhoc-queries.json", "queries.jsonl"),
        ("qrels-clinical_trials.txt", "test.tsv")
    ]

    for source_file, dest_file in files_to_copy:
        source_path = os.path.join(source_dir, source_file)
        destination_path = os.path.join(output_dir, dest_file)

        try:
            if source_file == "adhoc-queries.json":
                # Read the original JSON file
                with open(source_path, 'r', encoding='utf-8') as f:
                    queries = json.load(f)

                # Transform the queries and write to JSONL format
                with open(destination_path, 'w', encoding='utf-8') as f:
                    for query in queries:
                        transformed_query = {
                            "_id": f"sigir-{query['qId'][4:].replace('-', '')}",
                            "text": query['description']
                        }
                        json.dump(transformed_query, f, ensure_ascii=False)
                        f.write('\n')

                logging.info(f"Successfully transformed {source_file} and saved as {dest_file} in {output_dir}")

            elif source_file == "qrels-clinical_trials.txt":
                # Read the original file and transform
                with open(source_path, 'r', encoding='utf-8') as infile, \
                        open(destination_path, 'w', encoding='utf-8', newline='') as outfile:

                    tsv_writer = csv.writer(outfile, delimiter='\t')

                    # Write the header
                    tsv_writer.writerow(['query-id', 'corpus-id', 'score'])

                    # Process and write the data
                    for line in infile:
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            query_id = f"sigir-{parts[0]}"
                            corpus_id = parts[2]
                            score = parts[3]
                            tsv_writer.writerow([query_id, corpus_id, score])

                logging.info(f"Successfully transformed {source_file} and saved as {dest_file} in {output_dir}")

            else:
                # For any other files, just copy without modification
                shutil.copy2(source_path, destination_path)
                logging.info(f"Successfully copied {source_file} to {output_dir} as {dest_file}")

        except FileNotFoundError:
            logging.error(f"Source file not found: {source_path}")
        except PermissionError:
            logging.error(f"Permission denied when copying {source_file}")
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON in {source_file}")
        except Exception as e:
            logging.error(f"Error processing {source_file}: {str(e)}")

    time_function("copy_and_transform_files", start_time)


def main():
    """Main entry point for the script"""
    main_start_time = time.time()

    args = parse_args()
    setup_logging(args)

    # Convert string paths to Path objects for easier manipulation
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Clean and recreate output directory
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Removed and recreated directory: {output_dir}")

    # Find the XML files in the extracted dataset directory
    trials_dir = data_dir / "sigir2016" / "data" / "clinicaltrials.gov-16_dec_2015"

    # If the first path doesn't exist, try the nested structure that might be generated by extraction
    if not trials_dir.exists():
        nested_trials_dir = trials_dir / "clinicaltrials.gov-16_dec_2015"
        if nested_trials_dir.exists():
            trials_dir = nested_trials_dir
        else:
            logging.error(f"Trial directory not found at {trials_dir} or {nested_trials_dir}")
            logging.error("Please ensure the SIGIR2016 dataset has been downloaded and extracted correctly.")
            return

    # Filter out the macOS metadata files (._*) when collecting XML files
    xml_files = [f for f in trials_dir.glob("*.xml") if not f.name.startswith('._')]

    if not xml_files:
        logging.error(
            f"No XML files found in {trials_dir}. Please ensure the SIGIR2016 dataset has been downloaded correctly.")
        return

    logging.info(f"Found {len(xml_files)} XML files to process")

    # Process XML files in parallel
    processing_start_time = time.time()
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_xml_file, xml_files)
    time_function("XML processing", processing_start_time)

    valid_results = [result for result in results if result is not None]

    # Write the processed trials to a JSONL file
    write_start_time = time.time()
    output_file = output_dir / "corpus.jsonl"
    with open(output_file, 'w') as f:
        for result in valid_results:
            json.dump(result, f)
            f.write('\n')
    time_function("Writing results", write_start_time)

    logging.info(f"Processed {len(valid_results)} files successfully. Output saved to {output_file}")

    # Copy and transform auxiliary files
    copy_and_transform_files(data_dir, output_dir)

    time_function("Total processing", main_start_time)


def cli_main():
    """Entry point for the console script"""
    main()


if __name__ == "__main__":
    main()
```

## src/trialmesh/fetchers/pull_sigir2016.py

```python
#!/usr/bin/env python3
# src/trialmesh/data/pull_sigir2016.py

import os
import sys
import requests
import tarfile
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_file(url, destination_path):
    """
    Download a file from url to destination_path with progress bar
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise exception for HTTP errors

    # Get file size for progress bar if available
    file_size = int(response.headers.get('content-length', 0))
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up progress bar
    desc = f"Downloading {destination_path.name}"
    progress = tqdm(total=file_size, unit='B', unit_scale=True, desc=desc)

    # Download the file
    with open(destination_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive chunks
                file.write(chunk)
                progress.update(len(chunk))
    progress.close()

    return destination_path


def download_paper(data_dir):
    """Download the reference paper"""
    paper_url = "https://bevankoopman.github.io/papers/sigir2016_clinicaltrials_collection.pdf"
    paper_path = data_dir / "sigir2016_clinicaltrials_collection.pdf"

    print("Downloading reference paper...")
    try:
        download_file(paper_url, paper_path)
        print(f"Paper successfully downloaded to {paper_path}")
    except Exception as e:
        print(f"Error downloading paper: {e}")
        return False

    return True


def process_download_links_file(links_file_path, data_dir):
    """
    Parse the download links file and return a list of URL and destination pairs
    """
    if not links_file_path.exists():
        print(f"Error: Download links file does not exist at {links_file_path}")
        return []

    downloads = []

    with open(links_file_path, 'r') as f:
        url = None
        directory = None

        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue

            if line.startswith('http'):
                url = line
            elif line.startswith('dir='):
                directory = line.replace('dir=', '').strip()

                if url and directory:
                    # Get the filename from content-disposition in URL
                    filename = url.split('filename%3D%22')[1].split('%22')[0] if 'filename%3D%22' in url else \
                    url.split('/')[-1].split('?')[0]

                    # Create destination path relative to data_dir
                    rel_dir = Path(directory.lstrip('./')).relative_to(
                        '000017152v004') if '000017152v004' in directory else Path(directory.lstrip('./'))
                    dest_dir = data_dir / "sigir2016" / rel_dir
                    dest_path = dest_dir / filename

                    downloads.append((url, dest_path))
                    url = None
                    directory = None

    return downloads


def download_dataset(links_file_path, data_dir):
    """Download all files from the dataset using the links file"""
    print(f"Processing download links from {links_file_path}...")
    downloads = process_download_links_file(links_file_path, data_dir)

    if not downloads:
        print("No valid download links found.")
        return False

    print(f"Found {len(downloads)} files to download.")

    successful_downloads = 0

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_file, url, dest_path): (url, dest_path) for url, dest_path in downloads}

        for future in as_completed(futures):
            url, dest_path = futures[future]
            try:
                future.result()
                successful_downloads += 1
            except Exception as e:
                print(f"Error downloading {dest_path.name}: {e}")

    print(f"Successfully downloaded {successful_downloads} of {len(downloads)} files.")

    # Extract the tarball if it exists
    tarball_path = list(data_dir.glob("**/clinicaltrials.gov-*.tgz"))
    if tarball_path:
        tarball_path = tarball_path[0]
        print(f"Extracting {tarball_path}...")
        try:
            with tarfile.open(tarball_path, 'r:gz') as tar:
                tar.extractall(path=tarball_path.parent)
            print(f"Successfully extracted {tarball_path}")
        except Exception as e:
            print(f"Error extracting {tarball_path}: {e}")

    return successful_downloads > 0


def main():
    parser = argparse.ArgumentParser(description='Download SIGIR 2016 Clinical Trials dataset')
    parser.add_argument('--links_file', type=str, help='Path to the download links file (e.g., 17152v004.txt)')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to store the downloaded data')
    args = parser.parse_args()

    # Create data directory at the specified location (or default to ./data)
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using data directory: {data_dir}")

    print("SIGIR 2016 Clinical Trials Dataset Downloader")
    print("---------------------------------------------")

    # Download reference paper
    download_paper(data_dir)

    # Handle the dataset download
    if args.links_file:
        links_file_path = Path(args.links_file)
        download_dataset(links_file_path, data_dir)
    else:
        print("\nTo download the dataset, please follow these steps:")
        print("1. Go to https://data.csiro.au/collection/csiro:17152")
        print("2. Click 'Download all files' on the webpage")
        print("3. Download the generated text file (will be named something like '17152v004.txt')")
        print("4. Run this script again with the path to the downloaded file:")
        print(f"   trialmesh-download-sigir2016 --links_file=/path/to/17152v004.txt")


def cli_main():
    """Entry point for the console script"""
    main()


if __name__ == "__main__":
    main()
```

