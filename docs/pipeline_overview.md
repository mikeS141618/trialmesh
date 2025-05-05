# TrialMesh Pipeline Overview

## Introduction

The TrialMesh pipeline transforms unstructured clinical trial documents and patient records into a structured, semantically comparable format, and then performs multi-stage matching to identify the most appropriate trials for each patient. The pipeline combines large language models (LLMs) for clinical understanding with vector embeddings for efficient retrieval, followed by detailed clinical reasoning.

## Pipeline Stages

### 1. Data Acquisition and Preparation

The pipeline begins with acquiring clinical trial data and patient records:

- **Download Trial Data**: Source clinical trial documents from the SIGIR2016 dataset
- **Process XML**: Convert raw XML trial documents into structured JSONL format
- **Prepare Patient Queries**: Ensure patient records are in the required format

### 2. Clinical Summarization (LLM-Based)

Next, we use large language models to generate structured summaries:

- **Trial Summarization**: Create condensed summaries of trial documents optimized for embedding
- **Patient Summarization**: Extract clinically relevant information from patient records in a standardized format

This step transforms unstructured text into a more uniform representation, focusing on key clinical aspects.

### 3. Semantic Embedding Generation

The summarized documents are then converted into vector representations:

- **Trial Embedding**: Generate vector embeddings for all clinical trials
- **Patient Embedding**: Generate similar embeddings for patient records
- **Embedding Models**: Use domain-specific models like BGE, SapBERT, or BioClinicalBERT

These embeddings capture the semantic meaning of documents in vector space, allowing for similarity-based retrieval.

### 4. Vector Index Building

To enable efficient similarity search, we build optimized indices:

- **FAISS Index Creation**: Build fast vector indices (HNSW, IVF, or Flat)
- **Index Configuration**: Optimize parameters for the specific embedding model and dataset

The indices allow for rapid retrieval of relevant trials at scale.

### 5. Initial Retrieval

Using the vector indices, we perform initial candidate retrieval:

- **Similarity Search**: Find trials with embeddings similar to each patient
- **K-Nearest Neighbors**: Retrieve top-k most similar trials as candidates

This stage efficiently narrows down the search space to a manageable set of potentially relevant trials.

### 6. Clinical Reasoning and Filtering

The final stages apply medical reasoning through LLMs to refine the matches:

- **Exclusion Filtering**: Verify patients don't meet trial exclusion criteria
- **Inclusion Analysis**: Check if patients satisfy core inclusion requirements
- **Final Scoring**: Perform detailed clinical assessment of trial-patient compatibility

This stage simulates the medical judgment typically performed by trial coordinators.

### 7. Evaluation and Analysis

The pipeline concludes with performance evaluation:

- **Retrieval Evaluation**: Assess search quality against gold standard relevance judgments
- **Match Analysis**: Review match justifications and clinical reasoning

## Data Flow Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Raw XML Trials │     │  Patient Queries│     │  Gold Standard  │
│                 │     │                 │     │  Relevance Data │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       │
┌─────────────────┐     ┌─────────────────┐             │
│                 │     │                 │             │
│  Processed      │     │  Processed      │             │
│  Trial JSONL    │     │  Patient JSONL  │             │
│                 │     │                 │             │
└────────┬────────┘     └────────┬────────┘             │
         │                       │                       │
         ▼                       ▼                       │
┌─────────────────┐     ┌─────────────────┐             │
│                 │     │                 │             │
│  LLM-Generated  │     │  LLM-Generated  │             │
│  Trial Summaries│     │  Patient Summary│             │
│                 │     │                 │             │
└────────┬────────┘     └────────┬────────┘             │
         │                       │                       │
         ▼                       ▼                       │
┌─────────────────┐     ┌─────────────────┐             │
│                 │     │                 │             │
│  Trial Vector   │     │  Patient Vector │             │
│  Embeddings     │     │  Embeddings     │             │
│                 │     │                 │             │
└────────┬────────┘     └────────┬────────┘             │
         │                       │                       │
         ▼                       │                       │
┌─────────────────┐             │                       │
│                 │             │                       │
│  FAISS Vector   │◄────────────┘                       │
│  Index          │                                     │
│                 │                                     │
└────────┬────────┘                                     │
         │                                              │
         ▼                                              ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Vector Search  │────►│  LLM-Based      │────►│  Performance    │
│  Results        │     │  Clinical Match │     │  Evaluation     │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Configuration Options

### Summarization Options
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

## Example Commands

### Complete End-to-End Pipeline

```bash
{
  clear &&
  trialmesh-clean --clean-all --force &&
  trialmesh-summarize --model-path ../../models/Llama-3.3-70B-Instruct-FP8-dynamic --data-dir ./data --dataset sigir2016/processed_cut --output-dir ./data/sigir2016/summaries --cache-dir ./cache/llm_responses --tensor-parallel-size=4 --max-model-len=16384 --max-tokens=2048 --batch-size=33 --condensed-trial-only &&
  trialmesh-embed --model-path /home/mikenet/deepNets/models/bge-large-en-v1.5 --batch-size 256 --normalize --data-dir ./data --dataset sigir2016/summaries &&
  trialmesh-index build --embeddings ./data/sigir2016/summaries_embeddings/bge-large-en-v1.5/trial_embeddings.npy --output ./data/sigir2016/indices/bge-large-en-v1.5_trials_hnsw.index --index-type hnsw --m 96 --ef-construction 256 &&
  trialmesh-index search --index ./data/sigir2016/indices/bge-large-en-v1.5_trials_hnsw.index --queries ./data/sigir2016/summaries_embeddings/bge-large-en-v1.5/patient_embeddings.npy --output ./data/sigir2016/results/bge-large-en-v1.5_hnsw_search_results.json --k 33 &&
  trialmesh-evaluate &&
  trialmesh-match --model-path ../../models/Llama-3.3-70B-Instruct-FP8-dynamic --data-dir ./data --tensor-parallel-size=4 --max-model-len=16384 --max-tokens=2048 --batch-size=33 --include-all-trials
} |& tee trialmesh_run_$(date +%Y%m%d_%H%M%S).log
```

This command:
1. Cleans previous outputs with `trialmesh-clean`
2. Generates trial and patient summaries with `trialmesh-summarize`
3. Creates vector embeddings with `trialmesh-embed`
4. Builds an HNSW index with `trialmesh-index build`
5. Performs similarity search with `trialmesh-index search`
6. Evaluates search results against gold standard with `trialmesh-evaluate`
7. Performs detailed clinical matching with `trialmesh-match`
8. Logs all output to a timestamped file

### Individual Stage Commands

#### 1. Data Preparation
```bash
# Download SIGIR2016 dataset
trialmesh-download-sigir2016 --data-dir ./data

# Process XML trial documents
trialmesh-process-sigir2016 --data-dir ./data --log-level INFO
```

#### 2. Summarization
```bash
# Generate summaries for trials and patients
trialmesh-summarize --model-path /path/to/llama-model \
  --data-dir ./data \
  --dataset sigir2016/processed_cut \
  --tensor-parallel-size=4 \
  --max-model-len=8192 \
  --max-tokens=1024 \
  --batch-size=16 \
  --condensed-trial-only
```

#### 3. Embedding Generation
```bash
# Generate embeddings using BGE
trialmesh-embed \
  --model-path /path/to/bge-large-en-v1.5 \
  --batch-size 128 \
  --normalize \
  --data-dir ./data \
  --dataset sigir2016/summaries
```

#### 4. Index Building
```bash
# Build a HNSW index
trialmesh-index build \
  --embeddings ./data/sigir2016/summaries_embeddings/bge-large-en-v1.5/trial_embeddings.npy \
  --output ./data/sigir2016/indices/bge-large-en-v1.5_trials_hnsw.index \
  --index-type hnsw \
  --m 64 \
  --ef-construction 200
```

#### 5. Similarity Search
```bash
# Search for trials matching patients
trialmesh-index search \
  --index ./data/sigir2016/indices/bge-large-en-v1.5_trials_hnsw.index \
  --queries ./data/sigir2016/summaries_embeddings/bge-large-en-v1.5/patient_embeddings.npy \
  --output ./data/sigir2016/results/bge-large-en-v1.5_hnsw_search_results.json \
  --k 100
```

#### 6. Evaluation
```bash
# Evaluate search results
trialmesh-evaluate \
  --models bge-large-en-v1.5 \
  --visualize \
  --output-file ./data/sigir2016/evaluation/eval_results.csv
```

#### 7. Clinical Matching
```bash
# Run clinical trial matching
trialmesh-match \
  --model-path /path/to/llama-model \
  --data-dir ./data \
  --search-results sigir2016/results/bge-large-en-v1.5_hnsw_search_results.json \
  --tensor-parallel-size=4 \
  --max-model-len=8192 \
  --batch-size=16 
```

## Optimizations

### Performance Optimizations
- **GPU Parallelism**: Use tensor parallelism for large LLMs
- **Batch Processing**: Process multiple documents in batches
- **HNSW Indices**: Use for fast approximate nearest neighbor search
- **Caching**: All LLM responses are cached to avoid redundant computation

### Memory Optimizations
- **Quantization**: Use LLM quantization for reduced memory footprint (FP8)
- **Streaming**: Process large datasets incrementally
- **Condensed Summaries**: Use concise summaries optimized for embedding

### Quality Optimizations
- **Domain-Specific Models**: Use biomedical embedding models
- **Structured Prompting**: Carefully engineered prompts for clinical reasoning
- **Multi-Stage Filtering**: Progressive refinement of matches

## Troubleshooting

### Common Issues
- **GPU Out of Memory**: Reduce batch size or model context length
- **Index Building Failures**: Increase available memory or reduce HNSW parameters
- **Missing Dependencies**: Ensure all requirements are installed
- **Trial Processing Errors**: Check for malformed XML or missing fields
- **Cache Conflicts**: Use `trialmesh-clean` to clear problematic caches

### Monitoring and Debugging
- **Verbose Logging**: Use `--log-level DEBUG` for detailed information
- **Output Inspection**: Check intermediate outputs in data directory
- **Cache Inspection**: Examine cached responses in cache directory

## Best Practices

- **Start Small**: Test with a subset of data before full runs
- **Regular Cleaning**: Use `trialmesh-clean` between major pipeline changes
- **Model Selection**: Llama-3.3 or higher recommended for clinical reasoning
- **Parameter Tuning**: Adjust HNSW parameters based on dataset size
- **Evaluation First**: Run evaluation before detailed matching to assess retrieval quality
- **Logging**: Always capture full logs with the tee command

## Next Steps

After running the pipeline, consider these follow-up actions:

- Analyze match results in detail with Python scripts
- Compare performance across different embedding models
- Experiment with different LLM prompts for improved clinical reasoning
- Evaluate the impact of different summarization approaches