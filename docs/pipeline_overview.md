# TrialMesh Pipeline Overview

## Introduction

The TrialMesh pipeline transforms unstructured clinical trial documents and patient records into a structured, semantically comparable format, and then performs multi-stage matching to identify the most appropriate trials for each patient. The pipeline combines large language models (LLMs) for clinical understanding with vector embeddings for efficient retrieval, followed by detailed clinical reasoning.

## Pipeline Stages

### 1. Data Acquisition and Preparation

The pipeline begins with acquiring clinical trial data and patient records:

- **Download Trial Data**: Source clinical trial documents from sources like the SIGIR2016 dataset
- **Process XML**: Convert raw XML trial documents into structured JSONL format in `./data/processed`
- **Prepare Patient Queries**: Ensure patient records are in the required format

### 2. Clinical Summarization (LLM-Based)

Next, we use large language models with customizable prompts to generate structured summaries:

- **Trial Summarization**: Create configurable summaries of trial documents optimized for embedding
- **Patient Summarization**: Extract clinically relevant information from patient records in a standardized format
- **Prompt Selection**: Use specific prompt files with tailored token limits for different summarization tasks

This step transforms unstructured text into a more uniform representation, focusing on key clinical aspects. Summaries are stored in `./run/summaries/`.

### 3. Semantic Embedding Generation

The summarized documents are then converted into vector representations:

- **Trial Embedding**: Generate vector embeddings for all clinical trials
- **Patient Embedding**: Generate similar embeddings for patient records
- **Embedding Models**: Use domain-specific models like BGE, SapBERT, or BioClinicalBERT

These embeddings capture the semantic meaning of documents in vector space, allowing for similarity-based retrieval. Embeddings are stored in `./run/summaries_embeddings/`.

### 4. Vector Index Building

To enable efficient similarity search, we build optimized indices:

- **FAISS Index Creation**: Build fast vector indices (HNSW, IVF, or Flat)
- **Index Configuration**: Optimize parameters for the specific embedding model and dataset

The indices allow for rapid retrieval of relevant trials at scale and are saved to `./run/indices/`.

### 5. Initial Retrieval

Using the vector indices, we perform initial candidate retrieval:

- **Similarity Search**: Find trials with embeddings similar to each patient
- **K-Nearest Neighbors**: Retrieve top-k most similar trials as candidates

This stage efficiently narrows down the search space to a manageable set of potentially relevant trials. Results are stored in `./run/results/`.

### 6. Clinical Reasoning and Filtering

The final stages apply medical reasoning through LLMs to refine the matches:

- **Exclusion Filtering**: Verify patients don't meet trial exclusion criteria
- **Inclusion Analysis**: Check if patients satisfy core inclusion requirements
- **Final Scoring**: Perform detailed clinical assessment of trial-patient compatibility
- **Prompt Customization**: Each filter stage can use specialized prompts with appropriate token budgets

This stage simulates the medical judgment typically performed by trial coordinators. Match results are saved to `./run/matched/`.

### 7. Evaluation and Analysis

The pipeline concludes with performance evaluation:

- **Retrieval Evaluation**: Assess search quality against gold standard relevance judgments
- **Match Analysis**: Review match justifications and clinical reasoning

Evaluation metrics and visualizations are stored in `./run/evaluation/`.

## Data Flow Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Raw XML Trials │     │  Patient Queries│     │  Gold Standard  │
│   (data/)       │     │   (data/)       │     │  Relevance Data │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       │
┌─────────────────┐     ┌─────────────────┐              │
│  Processed      │     │  Processed      │              │
│  Trial JSONL    │     │  Patient JSONL  │              │
│   (data/)       │     │   (data/)       │              │
└────────┬────────┘     └────────┬────────┘              │
         │                       │                       │
         ▼                       ▼                       │
┌─────────────────┐     ┌─────────────────┐              │
│  LLM-Generated  │     │  LLM-Generated  │              │
│  Trial Summaries│     │  Patient Summary│              │
│  (run/summaries)│     │  (run/summaries)│              │
└────────┬────────┘     └────────┬────────┘              │
         │                       │                       │
         ▼                       ▼                       │
┌─────────────────┐     ┌─────────────────┐              │
│  Trial Vector   │     │  Patient Vector │              │
│  Embeddings     │     │  Embeddings     │              │
│  (run/embeddings)│    │ (run/embeddings)│              │
└────────┬────────┘     └────────┬────────┘              │
         │                       │                       │
         ▼                       │                       │
┌─────────────────┐              │                       │
│  FAISS Vector   │◄─────────────┘                       │
│  Index          │                                      │
│  (run/indices)  │                                      │
└────────┬────────┘                                      │
         │                                               │
         ▼                                               ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Vector Search  │────►│  LLM-Based      │────►│  Performance    │
│  Results        │     │  Clinical Match │     │  Evaluation     │
│  (run/results)  │     │  (run/matched)  │     │ (run/evaluation)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Configuration Options

### Prompt System Options
- **Prompt Directory**: Location of text files containing prompt templates
- **File Format**: `==== SYSTEM PROMPT ====` and `==== USER PROMPT ====` sections
- **Token Budgets**: Configure max tokens per prompt using `prompt_name:max_tokens[:temp]` syntax
- **Temperature**: Optionally specify temperature for generation after token count

### Summarization Options
- **Prompt Selection**: Choose specific prompt files for each summary type
- **Model Selection**: Choose appropriate LLM for medical summarization (Llama-3.3 recommended)
- **Context Length**: Adjust `max-model-len` based on document length and model capabilities
- **Output Tokens**: Set `max-tokens` to control summary length
- **Batch Size**: Optimize based on available GPU memory

### Embedding Options
- **Model Selection**: Choose domain-specific models (BGE, SapBERT, BioClinicalBERT)
- **Batch Size**: Adjust based on available GPU memory and model size
- **Normalization**: Always enable for cosine similarity search

### Indexing Options
- **Index Type**: Choose between:
  - `flat`: Exact search (slowest, most accurate)
  - `hnsw`: Hierarchical navigable small world graphs (fast, high accuracy)
  - `ivf`: Inverted file index (balanced speed/accuracy)
- **HNSW Parameters**:
  - `m`: Number of connections per node (higher = better recall, more memory)
  - `ef-construction`: Search depth during building (higher = better quality index)

### Retrieval Options
- **K Value**: Number of trial candidates to retrieve per patient
- **Metric**: Distance metric (cosine, L2, inner product)

### Matching Options
- **Model Selection**: Choose LLM with strong reasoning capabilities
- **Skip Stages**: Optionally bypass specific filtering stages for testing
- **Inclusion/Exclusion**: Configure filtering stringency
- **Prompt Selection**: Specify custom prompts for each matching stage
- **Token Limits**: Set appropriate token budgets for each filtering stage

## Example Commands

### Complete End-to-End Pipeline

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
    --batch-size=32 \
    --prompt-dir ./prompts \
    --exclusion-prompt exclusion_filter_sigir2016:2048 \
    --inclusion-prompt inclusion_filter_sigir2016:2048 \
    --scoring-prompt final_match_scoring_sigir2016:2048 \
    --include-all-trials
} |& tee trialmesh_run_$(date +%Y%m%d_%H%M%S).log
```

This command:
1. Removes any existing run and cache directories to start fresh
2. Generates summaries using specific prompt files with configured token limits
3. Creates vector embeddings with `trialmesh-embed` using multi-GPU acceleration
4. Builds a flat index with `trialmesh-index build`
5. Performs similarity search with `trialmesh-index search`
6. Evaluates search results against gold standard with `trialmesh-evaluate`
7. Performs detailed clinical matching with customized prompts and token budgets
8. Logs all output to a timestamped file

### Individual Stage Commands

#### 1. Data Preparation
```bash
# Download SIGIR2016 dataset
trialmesh-download-sigir2016 --data-dir ./data

# Process XML trial documents
trialmesh-process-sigir2016 --data-dir ./data --log-level INFO
```

#### 2. Prompt Listing and Summarization
```bash
# List available prompts
trialmesh-summarize --prompt-dir ./prompts --list-prompts

# Generate summaries with multiple configurable prompts
trialmesh-summarize \
  --model-path /path/to/llama-model \
  --data-dir ./data \
  --dataset processed \
  --output-dir ./run/summaries \
  --prompt-dir ./prompts \
  --patient-prompts \
      patient_summary_sigir2016:8000 \
      patient_condensed_sigir2016:512 \
      patient_pathway_sigir2016:6000:0.2 \
  --trial-prompts \
      trial_summary_sigir2016:8000 \
      trial_condensed_sigir2016:512
```

#### 3. Embedding Generation
```bash
# Generate embeddings using BGE
trialmesh-embed \
  --model-path /path/to/bge-large-en-v1.5 \
  --batch-size 128 \
  --normalize \
  --data-dir ./run \
  --dataset summaries
```

#### 4. Index Building
```bash
# Build a HNSW index
trialmesh-index build \
  --embeddings ./run/summaries_embeddings/bge-large-en-v1.5/trial_embeddings.npy \
  --output ./run/indices/bge-large-en-v1.5_trials_hnsw.index \
  --index-type hnsw \
  --m 64 \
  --ef-construction 200
```

#### 5. Similarity Search
```bash
# Search for trials matching patients
trialmesh-index search \
  --index ./run/indices/bge-large-en-v1.5_trials_hnsw.index \
  --queries ./run/summaries_embeddings/bge-large-en-v1.5/patient_embeddings.npy \
  --output ./run/results/bge-large-en-v1.5_hnsw_search_results.json \
  --k 100
```

#### 6. Evaluation
```bash
# Evaluate search results
trialmesh-evaluate \
  --data-dir ./data \
  --run-dir ./run \
  --models bge-large-en-v1.5 \
  --visualize \
  --output-file evaluation/eval_results.csv
```

#### 7. Clinical Matching with Custom Prompts
```bash
# List available matching prompts
trialmesh-match --prompt-dir ./prompts --list-prompts

# Run clinical trial matching with specific prompts and token budgets
trialmesh-match \
  --model-path /path/to/llama-model \
  --data-dir ./data \
  --run-dir ./run \
  --search-results results/bge-large-en-v1.5_hnsw_search_results.json \
  --prompt-dir ./prompts \
  --exclusion-prompt my_exclusion_filter:2048 \
  --inclusion-prompt my_inclusion_filter:2048:0.1 \
  --scoring-prompt my_final_scoring:4096 \
  --include-all-trials
```

## Optimizations

### Performance Optimizations
- **GPU Parallelism**: Use tensor parallelism for large LLMs
- **Batch Processing**: Process multiple documents in batches
- **HNSW Indices**: Use for fast approximate nearest neighbor search
- **Caching**: All LLM responses are cached to avoid redundant computation
- **Prompt Customization**: Tailor token budgets to each task's requirements

### Memory Optimizations
- **Quantization**: Use LLM quantization for reduced memory footprint (FP8)
- **Streaming**: Process large datasets incrementally
- **Condensed Summaries**: Use concise summaries optimized for embedding
- **Token Efficiency**: Set appropriate token limits for different pipeline stages

### Quality Optimizations
- **Domain-Specific Models**: Use biomedical embedding models
- **Structured Prompting**: Carefully engineered prompts for clinical reasoning
- **Multi-Stage Filtering**: Progressive refinement of matches
- **Prompt Experimentation**: Easily test different prompt variants without code changes

## Troubleshooting

### Common Issues
- **GPU Out of Memory**: Reduce batch size or model context length
- **Index Building Failures**: Increase available memory or reduce HNSW parameters
- **Missing Dependencies**: Ensure all requirements are installed
- **Trial Processing Errors**: Check for malformed XML or missing fields
- **Prompt Not Found**: Verify prompt files exist in the prompt directory

### Monitoring and Debugging
- **Verbose Logging**: Use `--log-level DEBUG` for detailed information
- **Output Inspection**: Check intermediate outputs in run directory
- **Cache Inspection**: Examine cached responses in cache directory
- **Prompt Listing**: Use `--list-prompts` to verify available prompts

## Best Practices

- **Directory Structure**: 
  - Keep source data in `./data`
  - Store all generated files in `./run`
  - Cache LLM responses in `./cache`
  - Keep prompt templates in `./prompts`
- **Start Small**: Test with a subset of data before full runs
- **Model Selection**: Llama-3.3 or higher recommended for clinical reasoning
- **Parameter Tuning**: Adjust HNSW parameters based on dataset size
- **Evaluation First**: Run evaluation before detailed matching to assess retrieval quality
- **Logging**: Always capture full logs with the tee command
- **Prompt Organization**: Keep prompts well-organized with descriptive filenames

## Next Steps

After running the pipeline, consider these follow-up actions:

- Analyze match results in detail with Python scripts
- Compare performance across different embedding models
- Experiment with different LLM prompts for improved clinical reasoning
- Evaluate the impact of different summarization approaches
- Extend the system to additional datasets by updating source data in `./data`
- Create specialized prompt variants for different medical specialties or trial types

## Directory Structure

The pipeline maintains a clean separation between source data and runtime outputs:

```
./data/              # Source data inputs
  └─ processed/      # Processed trial and patient documents
./prompts/           # Prompt template files
  ├─ patient_*.txt   # Patient summarization prompts
  ├─ trial_*.txt     # Trial summarization prompts
  ├─ exclusion_*.txt # Exclusion filtering prompts
  ├─ inclusion_*.txt # Inclusion filtering prompts
  └─ final_*.txt     # Final scoring prompts
./run/               # All generated files
  ├─ summaries/      # LLM-generated summaries
  ├─ summaries_embeddings/ # Generated embeddings
  ├─ indices/        # FAISS indices
  ├─ results/        # Search results
  ├─ matched/        # Match results
  └─ evaluation/     # Evaluation metrics and visualizations
./cache/             # Cached LLM responses
  └─ llm_responses/  # Hash-based response cache
```
