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
    max_model_len=2048,
    max_batch_size=8,
    use_cache=True,
    temperature=0.0
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
original_user_prompt = response.user_prompt
original_system_prompt = response.system_prompt

# Generate responses for multiple prompts in batch
prompts = [
    "Summarize the patient's condition.",
    "List key medical history.",
    "Identify potential contraindications."
]
batch_responses = llm.generate_batch(
    prompts=prompts,
    system_prompt="You are a clinical trial coordinator.",
    max_tokens=800,
    temperature=0.1
)
```

### PromptRunner

A higher-level interface that connects the LLM with prompt templates.

```python
from trialmesh.llm.prompt_runner import PromptRunner
from trialmesh.utils.prompt_registry import PromptRegistry

# Initialize with LlamaRunner and PromptRegistry
prompt_registry = PromptRegistry(prompt_dir="./prompts")
prompt_runner = PromptRunner(
    llama_runner=llm,
    prompt_registry=prompt_registry
)

# Run a prompt from the registry
response = prompt_runner.run_prompt(
    prompt_name="patient_summary_sigir2016",
    variables={"patient_text": "Patient presents with..."},
    max_tokens=1024,
    temperature=0.0
)

# Process a batch of documents
responses = prompt_runner.run_prompt_batch(
    prompt_name="trial_condensed_sigir2016",
    variables_list=[
        {"trial_text": "Trial 1 details..."},
        {"trial_text": "Trial 2 details..."}
    ],
    max_tokens=512,
    temperature=0.0
)
```

### Summarizer

Handles the generation of structured clinical summaries.

```python
from trialmesh.llm.summarizers import Summarizer
from trialmesh.utils.prompt_config import PromptConfig

# Initialize with model configuration
summarizer = Summarizer(
    model_path="/path/to/llama-model",
    prompt_dir="./prompts",
    cache_dir="./cache/llm_responses",
    tensor_parallel_size=4,
    max_model_len=16384,
    batch_size=8
)

# Define prompt configurations
trial_prompts = [
    PromptConfig(
        name="trial_summary_sigir2016",
        max_tokens=8000,
        temperature=0.0
    ),
    PromptConfig(
        name="trial_condensed_sigir2016",
        max_tokens=512,
        temperature=0.0
    )
]

patient_prompts = [
    PromptConfig(
        name="patient_summary_sigir2016",
        max_tokens=2048,
        temperature=0.0
    ),
    PromptConfig(
        name="patient_condensed_sigir2016",
        max_tokens=512,
        temperature=0.0
    )
]

# Generate summaries for trials
summarizer.summarize_trials(
    trials_path="./data/sigir2016/processed/corpus.jsonl",
    output_dir="./run/summaries",
    prompt_configs=trial_prompts,
    batch_size=8
)

# Generate summaries for patients
summarizer.summarize_patients(
    patients_path="./data/sigir2016/processed/queries.jsonl",
    output_dir="./run/summaries",
    prompt_configs=patient_prompts,
    batch_size=8
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
    device=None,  # Will use CUDA if available
    use_multi_gpu=False,
    normalize_embeddings=True,
    local_rank=-1  # Set to specific rank for distributed training
)

# Prepare the model (loads weights, moves to device)
model.prepare_model()
```

### BaseEmbeddingModel

Base interface for all embedding models.

```python
# Generate embeddings for a list of texts
embeddings_dict = model.encode(
    texts=["Patient with stage IV lung cancer", "History of breast cancer"],
    ids=["patient1", "patient2"],  # Optional IDs, will use indices if not provided
    show_progress=True
)

# Process a corpus from a JSONL file
model.encode_corpus(
    jsonl_path="./run/summaries/trial_condensed.jsonl",
    output_path="./run/summaries_embeddings/bge-large-en-v1.5/trial_embeddings.npy",
    text_field="summary",
    id_field="_id",
    batch_size=None  # Optional override for batch size
)
```

### FaissIndexBuilder

Creates FAISS indices for efficient similarity search.

```python
from trialmesh.embeddings.index_builder import FaissIndexBuilder
import numpy as np

# Create an index builder for HNSW index type
builder = FaissIndexBuilder(
    index_type="hnsw",  # Options: "flat", "ivf", "hnsw"
    dimension=None,     # Will be inferred from data
    metric="cosine",    # Options: "cosine", "l2", "ip"
    nlist=100,          # For IVF indices: number of centroids
    m=64,               # For HNSW indices: number of connections per layer
    ef_construction=200 # For HNSW indices: size of dynamic candidate list
)

# Build from embeddings dictionary
embeddings = np.load("./run/summaries_embeddings/bge-large-en-v1.5/trial_embeddings.npy", allow_pickle=True).item()
builder.build_from_dict(embeddings, normalize=True)

# Build from file directly
builder.build_from_file(
    embeddings_file="./run/summaries_embeddings/bge-large-en-v1.5/trial_embeddings.npy",
    normalize=True
)

# Save the index
builder.save_index("./run/indices/bge-large-en-v1.5_trials_hnsw.index")

# Load an existing index
loaded_builder = FaissIndexBuilder.load_index("./run/indices/bge-large-en-v1.5_trials_hnsw.index")
```

### FaissSearcher

Performs similarity searches using FAISS indices.

```python
from trialmesh.embeddings.query import FaissSearcher
import numpy as np

# Create a searcher from an index file
searcher = FaissSearcher(index_path="./run/indices/bge-large-en-v1.5_trials_hnsw.index")

# Search by vector
query_vector = np.random.rand(1024).astype(np.float32)  # Example vector
results = searcher.search(query_vector, query_id="test_query", k=10, normalize=True)

# Search by ID (using a dictionary of embeddings)
embeddings = np.load("./run/summaries_embeddings/bge-large-en-v1.5/patient_embeddings.npy", allow_pickle=True).item()
results = searcher.search_by_id("patient_123", embeddings, k=10, normalize=True)

# Batch search for multiple queries
batch_results = searcher.batch_search_by_id(
    query_ids=["patient_1", "patient_2", "patient_3"],
    embeddings=embeddings,
    k=100,
    normalize=True
)

# Convert results to JSON-serializable format
result_dicts = [result.to_dict() for result in batch_results]

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
    tensor_parallel_size=4,
    max_model_len=16384
)

# Create the matcher
matcher = TrialMatcher(
    llm=llm,
    patient_summaries_path="./run/summaries/patient_summary.jsonl",
    trials_path="./data/sigir2016/processed/corpus.jsonl",
    batch_size=8,
    prompt_dir="./prompts"
)

# Load search results
with open("./run/results/bge-large-en-v1.5_hnsw_search_results.json", "r") as f:
    search_results = json.load(f)

# Run the matching pipeline
match_results = matcher.match(
    search_results=search_results,
    exclusion_prompt="exclusion_filter_sigir2016",
    inclusion_prompt="inclusion_filter_sigir2016",
    scoring_prompt="final_match_scoring_sigir2016",
    exclusion_max_tokens=2048,
    inclusion_max_tokens=2048,
    scoring_max_tokens=4096,
    exclusion_temperature=0.0,
    inclusion_temperature=0.0,
    scoring_temperature=0.1,
    top_k=50,                    # Limit to top 50 trials per patient
    skip_exclusion=False,        # Enable exclusion filtering
    skip_inclusion=False,        # Enable inclusion filtering
    skip_scoring=False,          # Enable detailed scoring
    include_all_trials=True      # Include trials that were filtered out
)

# Save the results
with open("./run/matched/trial_matches.json", "w") as f:
    json.dump(match_results, f, indent=2)
```

## Utils Module

The utils module provides supporting functionality.

### PromptRegistry

Registry of prompt templates for LLM interactions.

```python
from trialmesh.utils.prompt_registry import PromptRegistry

# Create a registry with file-based prompts
registry = PromptRegistry(prompt_dir="./prompts")

# List available prompts
available_prompts = registry.list_available_prompts()
print(f"Available prompts: {available_prompts}")

# Get a prompt pair by name
prompt_pair = registry.get("patient_summary_sigir2016")
system_prompt = prompt_pair["system"]
user_prompt = prompt_pair["user"]

# Format a prompt with variables
formatted_prompt = user_prompt.format(patient_text="This patient has a history of...")
```

### PromptConfig

Configuration for prompt usage in summarization and matching.

```python
from trialmesh.utils.prompt_config import PromptConfig

# Create a configuration for a prompt
config = PromptConfig(
    name="trial_condensed_sigir2016",
    max_tokens=512,
    output_suffix="condensed",  # Optional, will be derived from name if not provided
    temperature=0.0
)

# The output_suffix will be used for naming output files
print(f"Output will be saved as: trial_{config.output_suffix}.jsonl")
```

### CodeMD Utility

Utility for generating code documentation.

```python
from trialmesh.utils.codemd import generate_codemd

# Generate comprehensive Markdown documentation of the codebase
generate_codemd()
# This will create a codecomplete.md file in the project root
```

## Integration Examples

### Complete Trial Matching Pipeline

```python
from trialmesh.llm.llama_runner import LlamaRunner
from trialmesh.llm.summarizers import Summarizer
from trialmesh.utils.prompt_config import PromptConfig
from trialmesh.embeddings.factory import EmbeddingModelFactory
from trialmesh.embeddings.index_builder import FaissIndexBuilder
from trialmesh.embeddings.query import FaissSearcher
from trialmesh.match.matcher import TrialMatcher
import json
import os
import numpy as np

# Define paths
data_dir = "./data/sigir2016"
run_dir = "./run"
cache_dir = "./cache"
prompt_dir = "./prompts"
os.makedirs(run_dir, exist_ok=True)
os.makedirs(os.path.join(run_dir, "summaries"), exist_ok=True)
os.makedirs(os.path.join(run_dir, "indices"), exist_ok=True)
os.makedirs(os.path.join(run_dir, "results"), exist_ok=True)
os.makedirs(os.path.join(run_dir, "matched"), exist_ok=True)

# 1. Initialize LLM
llm = LlamaRunner(
    model_path="/path/to/Llama-3.3-70B-Instruct-FP8-dynamic",
    cache_dir=os.path.join(cache_dir, "llm_responses"),
    tensor_parallel_size=4,
    max_model_len=16384,
    max_batch_size=8
)

# 2. Generate summaries
summarizer = Summarizer(
    model_path=llm.model_path,
    prompt_dir=prompt_dir,
    cache_dir=os.path.join(cache_dir, "llm_responses"),
    tensor_parallel_size=4,
    max_model_len=16384
)

# Define prompt configurations
trial_prompts = [
    PromptConfig(name="trial_condensed_sigir2016", max_tokens=512)
]

patient_prompts = [
    PromptConfig(name="patient_summary_sigir2016", max_tokens=2048),
    PromptConfig(name="patient_condensed_sigir2016", max_tokens=512)
]

# Generate summaries
summarizer.summarize_trials(
    trials_path=os.path.join(data_dir, "processed/corpus.jsonl"),
    output_dir=os.path.join(run_dir, "summaries"),
    prompt_configs=trial_prompts
)

summarizer.summarize_patients(
    patients_path=os.path.join(data_dir, "processed/queries.jsonl"),
    output_dir=os.path.join(run_dir, "summaries"),
    prompt_configs=patient_prompts
)

# 3. Generate embeddings
model = EmbeddingModelFactory.create_model(
    model_path="/path/to/bge-large-en-v1.5",
    batch_size=128,
    normalize_embeddings=True
)
model.prepare_model()

# Create embeddings directory
embeddings_dir = os.path.join(run_dir, "summaries_embeddings/bge-large-en-v1.5")
os.makedirs(embeddings_dir, exist_ok=True)

# Generate embeddings
model.encode_corpus(
    jsonl_path=os.path.join(run_dir, "summaries/trial_condensed.jsonl"),
    output_path=os.path.join(embeddings_dir, "trial_embeddings.npy")
)

model.encode_corpus(
    jsonl_path=os.path.join(run_dir, "summaries/patient_condensed.jsonl"),
    output_path=os.path.join(embeddings_dir, "patient_embeddings.npy")
)

# 4. Build FAISS index
builder = FaissIndexBuilder(index_type="hnsw", m=64, ef_construction=200)
builder.build_from_file(
    embeddings_file=os.path.join(embeddings_dir, "trial_embeddings.npy"),
    normalize=True
)
index_path = os.path.join(run_dir, "indices/bge-large-en-v1.5_trials_hnsw.index")
builder.save_index(index_path)

# 5. Perform search
searcher = FaissSearcher(index_path=index_path)
patient_embeddings = np.load(os.path.join(embeddings_dir, "patient_embeddings.npy"), allow_pickle=True).item()
results_path = os.path.join(run_dir, "results/bge-large-en-v1.5_hnsw_search_results.json")

batch_results = searcher.batch_search_by_id(
    query_ids=list(patient_embeddings.keys()),
    embeddings=patient_embeddings,
    k=100,
    normalize=True
)

# Convert results to JSON format
search_results = [result.to_dict() for result in batch_results]
with open(results_path, "w") as f:
    json.dump(search_results, f, indent=2)

# 6. Run trial matcher
matcher = TrialMatcher(
    llm=llm,
    patient_summaries_path=os.path.join(run_dir, "summaries/patient_summary.jsonl"),
    trials_path=os.path.join(data_dir, "processed/corpus.jsonl"),
    batch_size=8,
    prompt_dir=prompt_dir
)

match_results = matcher.match(
    search_results=search_results,
    exclusion_prompt="exclusion_filter_sigir2016",
    inclusion_prompt="inclusion_filter_sigir2016",
    scoring_prompt="final_match_scoring_sigir2016",
    exclusion_max_tokens=2048,
    inclusion_max_tokens=2048,
    scoring_max_tokens=2048,
    top_k=50
)

# Save match results
match_output_path = os.path.join(run_dir, "matched/trial_matches.json")
with open(match_output_path, "w") as f:
    json.dump(match_results, f, indent=2)
```

### Custom Embedding Model Implementation

Example of extending TrialMesh with a custom embedding model:

```python
from trialmesh.embeddings.base import BaseEmbeddingModel
import torch
from transformers import AutoModel, AutoTokenizer
from typing import List

class CustomEmbeddingModel(BaseEmbeddingModel):
    """Custom embedding model implementation."""
    
    def _load_model(self):
        """Load model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            logging.info(f"Loaded custom model from {self.model_path}")
        except Exception as e:
            logging.error(f"Error loading custom model: {str(e)}")
            raise
        
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
from trialmesh.embeddings.factory import EmbeddingModelFactory
model = EmbeddingModelFactory.create_model(
    model_type="custom-model",
    model_path="/path/to/custom/model",
    max_length=512,
    batch_size=32,
    normalize_embeddings=True
)
model.prepare_model()
```

## Extension Points

TrialMesh is designed to be extended in several ways:

### 1. Custom Embedding Models

Add new embedding models by:

1. Creating a class that inherits from `BaseEmbeddingModel`
2. Implementing the `_load_model` and `_batch_encode` methods
3. Registering the model in `MODEL_REGISTRY`

### 2. Custom Prompts

Add new prompt templates by creating text files in the prompts directory with the format:

```
==== SYSTEM PROMPT ====
Your system prompt content here.

==== USER PROMPT ====
Your user prompt content here with {variables} for replacement.
```

Or programmatically:

```python
from trialmesh.utils.prompt_registry import PromptRegistry

# Create a custom registry
class CustomPromptRegistry(PromptRegistry):
    def __init__(self, prompt_dir: str = "./prompts"):
        super().__init__(prompt_dir)
        
        # Add additional prompts programmatically
        self.prompts.update({
            "custom_prompt": {
                "system": "Custom system prompt",
                "user": "Custom user prompt with {variable}"
            }
        })
```

### 3. Custom Matching Logic

Customize the matching pipeline by:

1. Subclassing `TrialMatcher` 
2. Overriding methods like `_apply_exclusion_filter`, `_apply_inclusion_filter`, or `_apply_scoring`
3. Implementing your own matching logic while maintaining the same interface

```python
from trialmesh.match.matcher import TrialMatcher
from typing import List, Dict, Tuple, Any, Optional

class CustomMatcher(TrialMatcher):
    def _apply_scoring(self, patient_summary: str, trials: List[Dict[str, Any]],
                       max_tokens: Optional[int] = None,
                       temperature: Optional[float] = None,
                       prompt_name: str = "final_match_scoring_sigir2016") -> List[Dict[str, Any]]:
        """Override with custom scoring logic."""
        # Your custom scoring implementation
        logging.info("Using custom scoring logic")
        
        # Example: simplified scoring that retains the interface
        scored_trials = []
        for trial_result in trials:
            trial_id = trial_result["trial_id"]
            trial_data = trial_result["trial_data"]
            
            # Add a simplified scoring result
            trial_result["scoring_result"] = {
                "score": "8",
                "verdict": "LIKELY MATCH",
                "reasoning": "Custom matching logic applied"
            }
            
            scored_trials.append(trial_result)
            
        return scored_trials
```

### 4. Pipeline Customization

Create custom pipelines by combining TrialMesh components in new ways:

1. Use the programmatic API instead of CLI tools
2. Select components based on your specific needs
3. Integrate with external systems by following the provided patterns

Note that all core TrialMesh objects follow consistent initialization patterns and return standardized data structures for interoperability.