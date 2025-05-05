# TrialMesh API Documentation

This document provides a comprehensive reference for the TrialMesh API, organized by core modules. It explains how to programmatically use TrialMesh for trial matching, embedding generation, and evaluation in your own applications.

## Table of Contents

- [LLM Module](#llm-module)
- [Embeddings Module](#embeddings-module)
- [Matching Module](#matching-module)
- [Utils Module](#utils-module)
- [Integration Examples](#integration-examples)
- [Extension Points](#extension-points)

## LLM Module

The LLM module provides interfaces for interacting with large language models.

### LlamaRunner

The core class for LLM inference with caching.

```python
from trialmesh.llm.llama_runner import LlamaRunner, LlamaResponse

# Initialize LlamaRunner with model configuration
llm = LlamaRunner(
    model_path="/path/to/llama-model",
    cache_dir="./cache/llm_responses",
    tensor_parallel_size=4,
    max_tokens=1024,
    max_model_len=2048
)

# Generate text from a prompt
response = llm.generate(
    prompt="Describe the patient's symptoms.",
    system_prompt="You are a helpful clinical assistant.",
    max_tokens=500,
    temperature=0.0
)

# Access response attributes
generated_text = response.text
input_token_count = response.input_tokens
output_token_count = response.output_tokens
```

### PromptRunner

A higher-level interface that connects the LLM with prompt templates.

```python
from trialmesh.llm.prompt_runner import PromptRunner
from trialmesh.utils.prompt_registry import PromptRegistry

# Initialize with a LlamaRunner
prompt_runner = PromptRunner(llm)

# Run a prompt from the registry
response = prompt_runner.run_prompt(
    prompt_name="patient_summary",
    variables={"patient_text": "Patient presents with..."},
    max_tokens=1024
)

# Process a batch of documents
responses = prompt_runner.run_prompt_batch(
    prompt_name="trial_condensed",
    variables_list=[
        {"trial_text": "Trial 1 details..."},
        {"trial_text": "Trial 2 details..."}
    ],
    max_tokens=512
)
```

### Summarizer

Handles the generation of structured clinical summaries.

```python
from trialmesh.llm.summarizers import Summarizer

# Initialize with model configuration
summarizer = Summarizer(
    model_path="/path/to/llama-model",
    cache_dir="./cache/llm_responses",
    tensor_parallel_size=4
)

# Generate summaries for trials
summarizer.summarize_trials(
    trials_path="./data/sigir2016/processed_cut/corpus.jsonl",
    output_dir="./data/sigir2016/summaries",
    batch_size=8,
    max_tokens=1024,
    condensed_trial_only=True
)

# Generate summaries for patients
summarizer.summarize_patients(
    patients_path="./data/sigir2016/processed_cut/queries.jsonl",
    output_dir="./data/sigir2016/summaries",
    batch_size=8,
    max_tokens=1024
)
```

## Embeddings Module

The embeddings module handles vector representations of text and similarity search.

### EmbeddingModelFactory

Factory for creating embedding models of different types.

```python
from trialmesh.embeddings.factory import EmbeddingModelFactory

# List available models
models = EmbeddingModelFactory.get_available_models()
print(f"Available models: {models}")

# Create a model with auto-detection
model = EmbeddingModelFactory.create_model(
    model_path="/path/to/bge-large-en-v1.5",
    max_length=512,
    batch_size=32,
    normalize_embeddings=True
)

# Prepare the model (loads weights, moves to device)
model.prepare_model()
```

### BaseEmbeddingModel

Base interface for all embedding models.

```python
# Generate embeddings for a list of texts
embeddings = model.encode(
    texts=["Patient with stage IV lung cancer", "History of breast cancer"],
    show_progress=True
)

# Process a corpus from a JSONL file
model.encode_corpus(
    jsonl_path="./data/sigir2016/summaries/trial_condensed.jsonl",
    output_path="./data/sigir2016/embeddings/trial_embeddings.npy",
    text_field="summary",
    id_field="_id"
)
```

### FaissIndexBuilder

Creates FAISS indices for efficient similarity search.

```python
from trialmesh.embeddings.index_builder import FaissIndexBuilder
import numpy as np

# Create an index builder for HNSW index type
builder = FaissIndexBuilder(
    index_type="hnsw",
    metric="cosine",
    m=64,
    ef_construction=200
)

# Build from embeddings dictionary
embeddings = np.load("./embeddings.npy", allow_pickle=True).item()
builder.build_from_dict(embeddings, normalize=True)

# Save the index
builder.save_index("./faiss_index.index")

# Load an existing index
loaded_builder = FaissIndexBuilder.load_index("./faiss_index.index")
```

### FaissSearcher

Performs similarity searches using FAISS indices.

```python
from trialmesh.embeddings.query import FaissSearcher
import numpy as np

# Create a searcher from an index file
searcher = FaissSearcher(index_path="./faiss_index.index")

# Search by vector
query_vector = np.random.rand(768).astype(np.float32)  # Example vector
results = searcher.search(query_vector, k=10)

# Search by ID (using a dictionary of embeddings)
embeddings = np.load("./embeddings.npy", allow_pickle=True).item()
results = searcher.search_by_id("patient_123", embeddings, k=10)

# Batch search for multiple queries
batch_results = searcher.batch_search_by_id(
    query_ids=["patient_1", "patient_2", "patient_3"],
    embeddings=embeddings,
    k=10
)

# Access results
for result in batch_results:
    print(f"Query: {result.query_id}")
    for i, (doc_id, score) in enumerate(zip(result.doc_ids, result.distances)):
        print(f"  {i+1}. {doc_id}: {score:.4f}")
```

## Matching Module

The matching module implements clinical trial-patient matching logic.

### TrialMatcher

The main class for matching patients to appropriate trials.

```python
from trialmesh.match.matcher import TrialMatcher
from trialmesh.llm.llama_runner import LlamaRunner
import json

# Initialize LLM for reasoning
llm = LlamaRunner(
    model_path="/path/to/llama-model",
    cache_dir="./cache/matcher",
    tensor_parallel_size=4
)

# Create the matcher
matcher = TrialMatcher(
    llm=llm,
    data_dir="./data",
    patient_summaries_path="sigir2016/summaries/patient_summaries.jsonl",
    trials_path="sigir2016/processed_cut/corpus.jsonl",
    batch_size=8
)

# Load search results
with open("./data/sigir2016/results/search_results.json", "r") as f:
    search_results = json.load(f)

# Run the matching pipeline
match_results = matcher.match(
    search_results=search_results,
    top_k=50,                    # Limit to top 50 trials per patient
    skip_exclusion=False,        # Enable exclusion filtering
    skip_inclusion=False,        # Enable inclusion filtering
    skip_scoring=False,          # Enable detailed scoring
    include_all_trials=True      # Include trials that were filtered out
)

# Save the results
with open("./data/sigir2016/matched/trial_matches.json", "w") as f:
    json.dump(match_results, f, indent=2)
```

## Utils Module

The utils module provides supporting functionality.

### PromptRegistry

Registry of prompt templates for LLM interactions.

```python
from trialmesh.utils.prompt_registry import PromptRegistry

# Create a registry
registry = PromptRegistry()

# Get a prompt pair by name
prompt_pair = registry.get("patient_summary")
system_prompt = prompt_pair["system"]
user_prompt = prompt_pair["user"]

# Get individual components
system_prompt = registry.get_system("trial_condensed")
user_prompt = registry.get_user("trial_condensed")

# Format a prompt with variables
formatted_prompt = user_prompt.format(trial_text="This trial studies...")
```

### Cleaning Utilities

Functions for managing cache and intermediate files.

```python
from trialmesh.utils.clean_run import clean_directory, resolve_directories
from pathlib import Path
import argparse

# Clean a specific directory
result = clean_directory(Path("./cache/llm_responses"), dry_run=True)

# Resolve directories based on arguments
args = argparse.Namespace()
args.data_dir = "./data"
args.clean_cache = True
args.model_name = "SapBERT"
dirs_by_category = resolve_directories(args)
```

## Integration Examples

### Complete Trial Matching Pipeline

```python
from trialmesh.llm.llama_runner import LlamaRunner
from trialmesh.llm.summarizers import Summarizer
from trialmesh.embeddings.factory import EmbeddingModelFactory
from trialmesh.embeddings.index_builder import FaissIndexBuilder
from trialmesh.embeddings.query import FaissSearcher
from trialmesh.match.matcher import TrialMatcher
import json
import os

# Define paths
data_dir = "./data"
dataset = "sigir2016/processed_cut"
summaries_dir = os.path.join(data_dir, "sigir2016/summaries")
embeddings_dir = os.path.join(data_dir, "sigir2016/summaries_embeddings/bge-large-en-v1.5")
index_path = os.path.join(data_dir, "sigir2016/indices/bge-large-en-v1.5_trials_hnsw.index")
results_path = os.path.join(data_dir, "sigir2016/results/search_results.json")
match_output_path = os.path.join(data_dir, "sigir2016/matched/trial_matches.json")

# 1. Initialize LLM
llm = LlamaRunner(
    model_path="/path/to/llama-model",
    cache_dir="./cache/llm_responses",
    tensor_parallel_size=4
)

# 2. Generate summaries
summarizer = Summarizer(llm.model_path, cache_dir="./cache/llm_responses")
summarizer.summarize_trials(
    trials_path=os.path.join(data_dir, dataset, "corpus.jsonl"),
    output_dir=summaries_dir,
    condensed_trial_only=True
)
summarizer.summarize_patients(
    patients_path=os.path.join(data_dir, dataset, "queries.jsonl"),
    output_dir=summaries_dir
)

# 3. Generate embeddings
model = EmbeddingModelFactory.create_model(
    model_path="/path/to/bge-large-en-v1.5",
    batch_size=32,
    normalize_embeddings=True
)
model.encode_corpus(
    jsonl_path=os.path.join(summaries_dir, "trial_condensed.jsonl"),
    output_path=os.path.join(embeddings_dir, "trial_embeddings.npy")
)
model.encode_corpus(
    jsonl_path=os.path.join(summaries_dir, "patient_condensed.jsonl"),
    output_path=os.path.join(embeddings_dir, "patient_embeddings.npy")
)

# 4. Build FAISS index
builder = FaissIndexBuilder(index_type="hnsw", m=64, ef_construction=200)
builder.build_from_file(
    embeddings_file=os.path.join(embeddings_dir, "trial_embeddings.npy"),
    normalize=True
)
builder.save_index(index_path)

# 5. Perform search
searcher = FaissSearcher(index_path=index_path)
embeddings = model.encode_corpus(
    jsonl_path=os.path.join(summaries_dir, "patient_condensed.jsonl"),
    output_path=os.path.join(embeddings_dir, "patient_embeddings.npy")
)
batch_results = searcher.batch_search_by_id(
    query_ids=list(embeddings.keys()),
    embeddings=embeddings,
    k=100
)

# Convert results to JSON format
search_results = [result.to_dict() for result in batch_results]
with open(results_path, "w") as f:
    json.dump(search_results, f, indent=2)

# 6. Run trial matcher
matcher = TrialMatcher(
    llm=llm,
    data_dir=data_dir,
    patient_summaries_path="sigir2016/summaries/patient_summaries.jsonl",
    trials_path=dataset + "/corpus.jsonl"
)
match_results = matcher.match(search_results=search_results, top_k=50)

# Save match results
with open(match_output_path, "w") as f:
    json.dump(match_results, f, indent=2)
```

### Custom Embedding Model Implementation

Example of extending TrialMesh with a custom embedding model:

```python
from trialmesh.embeddings.base import BaseEmbeddingModel
import torch
from transformers import AutoModel, AutoTokenizer

class CustomEmbeddingModel(BaseEmbeddingModel):
    """Custom embedding model implementation."""
    
    def _load_model(self):
        """Load model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path)
        
    def _batch_encode(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts to embeddings."""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use CLS token embedding
        embeddings = outputs.last_hidden_state[:, 0]
        
        # Normalize if requested
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        return embeddings

# Register the custom model
from trialmesh.embeddings.models import MODEL_REGISTRY
MODEL_REGISTRY["custom-model"] = CustomEmbeddingModel

# Use the custom model
model = EmbeddingModelFactory.create_model(
    model_type="custom-model",
    model_path="/path/to/custom/model"
)
```

## Extension Points

TrialMesh is designed to be extended in several ways:

### 1. Custom Embedding Models

Add new embedding models by:

1. Creating a class that inherits from `BaseEmbeddingModel`
2. Implementing the `_load_model` and `_batch_encode` methods
3. Registering the model in `MODEL_REGISTRY`

### 2. Custom Prompts

Add new prompt templates by:

1. Extending the `PromptRegistry` class
2. Adding new methods that return prompt dictionaries
3. Using your custom registry with `PromptRunner`

```python
class CustomPromptRegistry(PromptRegistry):
    def __init__(self):
        super().__init__()
        self.prompts.update({
            "custom_prompt": self._custom_prompt()
        })
        
    @staticmethod
    def _custom_prompt():
        return {
            "system": "Custom system prompt",
            "user": "Custom user prompt with {variable}"
        }
```

### 3. Custom Matching Logic

Customize the matching pipeline by:

1. Subclassing `TrialMatcher` 
2. Overriding methods like `_apply_exclusion_filter`, `_apply_inclusion_filter`, or `_apply_scoring`
3. Implementing your own matching logic while maintaining the same interface

```python
class CustomMatcher(TrialMatcher):
    def _apply_scoring(self, patient_summary, trials):
        # Your custom scoring logic
        # ...
        return scored_trials
```

### 4. Pipeline Customization

Create custom pipelines by combining TrialMesh components in new ways:

1. Use the programmatic API instead of CLI tools
2. Select components based on your specific needs
3. Integrate with external systems by following the provided patterns

Note that all core TrialMesh objects follow consistent initialization patterns and return standardized data structures for interoperability.