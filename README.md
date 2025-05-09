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
- **Distributed Processing:** Multi-GPU support for accelerated embedding generation and LLM inference.
- **Advanced Similarity Search:** FAISS indices for efficient retrieval with flat, IVF, and HNSW index types.
- **Comprehensive Evaluation:** Tools for evaluating retrieval performance against gold standard data.
- **Configurable Prompts:** File-based prompt system for easy customization without code changes.

## Documentation

- [Pipeline Overview](docs/pipeline_overview.md): Complete end-to-end pipeline explanation
- [Prompt Design](docs/prompt_design.md): Details on LLM prompt engineering for clinical tasks
- [API Documentation](docs/api.md): API reference for developers

## Installation

```bash
# Clone the repository
git clone https://github.com/mikeS141618/trialmesh.git
cd trialmesh

# Install the package in development mode
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

## Prompt System

TrialMesh uses a file-based prompt system that allows for easy customization and experimentation. Prompts are stored in individual text files in the `./prompts` directory with the following format:

```
==== SYSTEM PROMPT ====
[System prompt content goes here]

==== USER PROMPT ====
[User prompt content goes here]
```

To list available prompts:

```bash
trialmesh-summarize --prompt-dir ./prompts --list-prompts
# or
trialmesh-match --prompt-dir ./prompts --list-prompts
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
# Generate LLM summaries for patients and trials with specific prompt configurations
trialmesh-summarize \
  --model-path /path/to/llama-model \
  --data-dir ./data \
  --dataset processed \
  --output-dir ./run/summaries \
  --cache-dir ./cache/llm_responses \
  --prompt-dir ./prompts \
  --patient-prompts \
      patient_summary_sigir2016:8000 \
      patient_condensed_sigir2016:512 \
  --trial-prompts \
      trial_summary_sigir2016:8000 \
      trial_condensed_sigir2016:512

# You can specify multiple prompts with different token limits and temperatures
# Format: prompt_name:max_tokens[:temperature]
trialmesh-summarize \
  --model-path /path/to/llama-model \
  --prompt-dir ./prompts \
  --patient-prompts \
      patient_summary_sigir2016:8000 \
      patient_condensed_sigir2016:512:0.2 \
      patient_pathway_sigir2016:6000
```

### Embedding Generation

```bash
# Generate embeddings using a single GPU
trialmesh-embed \
  --model-path /path/to/SapBERT \
  --batch-size 32 \
  --normalize \
  --data-dir ./run \
  --dataset summaries

# Generate embeddings using multiple GPUs (distributed processing)
torchrun --nproc_per_node=4 \
  $(which trialmesh-embed) --multi-gpu \
  --model-path /path/to/bge-large-en-v1.5 \
  --batch-size 512 \
  --normalize \
  --data-dir ./run \
  --dataset summaries
```

### FAISS Index Building and Search

```bash
# Build a FAISS HNSW index (fast search)
trialmesh-index build \
  --embeddings ./run/summaries_embeddings/SapBERT/trial_embeddings.npy \
  --output ./run/indices/SapBERT_trials_hnsw.index \
  --index-type hnsw \
  --m 64

# Search for trials matching patients
trialmesh-index search \
  --index ./run/indices/SapBERT_trials_hnsw.index \
  --queries ./run/summaries_embeddings/SapBERT/patient_embeddings.npy \
  --output ./run/results/SapBERT_search_results.json \
  --k 100
```

### Retrieval Pipeline

```bash
# Run the complete retrieval pipeline (embedding, indexing, search) for all models
trialmesh-retrieval --run-dir ./run

# Run with specific models
trialmesh-retrieval --models SapBERT bge-large-v1.5 --run-dir ./run

# Run with HNSW index type
trialmesh-retrieval --index-type hnsw --m-value 96 --ef-construction 300 --run-dir ./run

# Skip certain stages
trialmesh-retrieval --skip-embeddings --skip-indexing --run-dir ./run
```

### Evaluation

```bash
# Evaluate all model search results
trialmesh-evaluate --data-dir ./data --run-dir ./run

# Evaluate specific models with visualization
trialmesh-evaluate --models SapBERT bge-large-v1.5 --visualize --run-dir ./run

# Evaluate only models using HNSW indices and save results
trialmesh-evaluate --index-type hnsw --output-file evaluation_results.csv --run-dir ./run
```

### Matching

```bash
# Run detailed matching analysis with LLM reasoning
trialmesh-match \
  --model-path /path/to/llama-model \
  --data-dir ./data \
  --run-dir ./run \
  --search-results results/bge-large-en-v1.5_hnsw_search_results.json \
  --tensor-parallel-size 4

# Use custom prompts with specific token limits
trialmesh-match \
  --model-path /path/to/llama-model \
  --run-dir ./run \
  --search-results results/bge-large-en-v1.5_flat_search_results.json \
  --prompt-dir ./prompts \
  --exclusion-prompt exclusion_filter_sigir2016:2048 \
  --inclusion-prompt inclusion_filter_sigir2016:2048 \
  --scoring-prompt final_match_scoring_sigir2016:4096:0.1 \
  --include-all-trials
```

## Complete End-to-End Pipeline

TrialMesh provides a complete pipeline for clinical trial matching. The following command runs the entire process from data preparation to final matching:

```bash
{
  clear && rm -rv run/ && rm -rv cache/ &&
  trialmesh-summarize \
    --model-path ../../models/Llama-3.3-70B-Instruct-FP8-dynamic \
    --data-dir ./data/sigir2016 \
    --dataset processed_cut \
    --cache-dir ./cache/llm_responses \
    --tensor-parallel-size=4 \
    --max-model-len=16384 \
    --batch-size=32 \
    --output-dir ./run/summaries \
    --prompt-dir ./prompts \
    --patient-prompts patient_summary_sigir2016:2048 patient_condensed_sigir2016:512 \
    --trial-prompts trial_condensed_sigir2016:512 &&
  torchrun --nproc_per_node=4 \
    $(which trialmesh-embed) --multi-gpu \
    --model-path ../../models/bge-large-en-v1.5 \
    --batch-size 512 \
    --normalize \
    --data-dir ./run \
    --dataset summaries &&
  trialmesh-index build \
    --embeddings ./run/summaries_embeddings/bge-large-en-v1.5/trial_embeddings.npy \
    --output ./run/indices/bge-large-en-v1.5_trials_flat.index \
    --index-type flat \
    --m 128 \
    --ef-construction 512 &&
  trialmesh-index search \
    --index ./run/indices/bge-large-en-v1.5_trials_flat.index \
    --queries ./run/summaries_embeddings/bge-large-en-v1.5/patient_embeddings.npy \
    --output ./run/results/bge-large-en-v1.5_flat_search_results.json \
    --k 128 &&
  trialmesh-evaluate \
    --data-dir ./data/sigir2016 \
    --dataset processed_cut &&
  trialmesh-match \
    --model-path ../../models/Llama-3.3-70B-Instruct-FP8-dynamic \
    --data-dir ./data/sigir2016 \
    --run-dir ./run \
    --search-results results/bge-large-en-v1.5_flat_search_results.json \
    --tensor-parallel-size 4 \
    --max-model-len=16384 \
    --prompt-dir ./prompts \
    --exclusion-prompt exclusion_filter_sigir2016:2048 \
    --inclusion-prompt inclusion_filter_sigir2016:2048 \
    --scoring-prompt final_match_scoring_sigir2016:2048 \
    --include-all-trials
} |& tee trialmesh_run_$(date +%Y%m%d_%H%M%S).log
```

For more details on the pipeline architecture and implementation, see the [Pipeline Overview](docs/pipeline_overview.md).

## Project Structure

The project follows a modern `src` layout with a clear separation between input data and runtime outputs:

```
trialmesh/
├── src/
│   └── trialmesh/         # Main package code
│       ├── cli/           # Command-line interfaces
│       ├── config/        # Configuration settings
│       ├── data/          # Data acquisition and processing
│       ├── embeddings/    # Vector embeddings and FAISS
│       ├── evaluation/    # Evaluation tools
│       ├── llm/           # LLM runners and processors
│       ├── match/         # Matching logic and pipeline
│       └── utils/         # Utilities and prompt extraction tools
├── data/                  # Source data storage
├── prompts/               # Individual prompt text files
├── run/                   # Runtime outputs
│   ├── summaries/         # LLM-generated summaries
│   ├── summaries_embeddings/ # Vector embeddings
│   ├── indices/           # FAISS indices
│   ├── results/           # Search results
│   ├── matched/           # Matching results
│   └── evaluation/        # Evaluation metrics and visualizations
├── cache/                 # Cache for LLM responses
├── docs/                  # Documentation
│   ├── pipeline_overview.md  # Complete pipeline explanation
│   └── prompt_design.md      # LLM prompt engineering details
├── tests/                 # Test suite
└── notebooks/             # Analysis notebooks
    ├── sigir_cut_to_test.ipynb    # Dataset preprocessing
    ├── sigir_eval_capture.ipynb   # Retrieval analysis
    └── sigir_eval_results.ipynb   # Matching evaluation
```

## Data Processing Pipeline

TrialMesh processes clinical trial data through several stages:

1. **Data Acquisition:** Download and extract trial documents and patient queries
2. **Structured Extraction:** Parse XML into structured JSON with clinical fields
3. **LLM Summarization:** Create summaries using configurable prompts and token limits
4. **Vector Embedding:** Generate semantic embeddings for efficient retrieval
   - Supports multi-GPU distributed processing for faster embedding generation
5. **Index Building:** Create optimized FAISS indices for fast similarity search
6. **Matching:** Apply multi-stage filtering from vector retrieval to detailed LLM scoring
7. **Evaluation:** Assess retrieval performance against gold standard relevance judgments

## Analysis Notebooks

TrialMesh includes several Jupyter notebooks for data analysis and evaluation:

- **[sigir_cut_to_test.ipynb](notebooks/sigir_cut_to_test.ipynb)**: Preprocesses and subsets the SIGIR2016 dataset to create a smaller, well-structured development/test set with complete inclusion and exclusion criteria.

- **[sigir_eval_capture.ipynb](notebooks/sigir_eval_capture.ipynb)**: Analyzes retrieval results from various embedding models by comparing their top-retrieved trials against gold-standard relevance annotations, focusing on identifying missing relevant trials.

- **[sigir_eval_results.ipynb](notebooks/sigir_eval_results.ipynb)**: Evaluates the clinical trial-patient matching system by comparing system-assigned trial suitability verdicts against gold-standard labels, with detailed confusion matrices and error analysis.

## Supported Embedding Models

TrialMesh supports multiple embedding models optimized for different aspects of clinical text:

- **E5-large-v2**: General-purpose embedding model with 1024 dimensions
- **BGE-large-en-v1.5**: Powerful similarity embedding model with 1024 dimensions
- **SapBERT**: Specialized biomedical model trained on UMLS medical concepts
- **BioClinicalBERT**: Clinical BERT model trained on medical records
- **BlueBERT**: Model trained on PubMed abstracts and MIMIC clinical notes

All models can be loaded from local disk with full multi-GPU support for distributed processing.

## Distributed Processing

TrialMesh offers comprehensive multi-GPU support:

- **Embedding Generation**: Distributes document processing across GPUs for faster embedding generation
- **LLM Inference**: Uses tensor parallelism via vLLM for efficient large model inference
- **Flexible Configuration**: Adjustable tensor-parallel size and batch parameters to optimize for different hardware
- **Performance Scaling**: Scales efficiently with the number of available GPUs for large datasets

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
- File-based prompt system for easy customization
- Multi-model embedding generation with GPU acceleration
- Distributed multi-GPU processing for faster embedding generation
- FAISS indexing and similarity search
- Automated retrieval pipeline for multiple models
- Performance evaluation against gold standard data
- Multi-stage matching pipeline with clinical reasoning
- Comprehensive documentation and analysis tools

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
