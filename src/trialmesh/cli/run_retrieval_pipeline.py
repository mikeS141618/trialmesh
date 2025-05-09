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
    """Configure logging.

    Args:
        log_level: Desired logging level (DEBUG, INFO, etc.)

    Raises:
        ValueError: If invalid log level provided
    """
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
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Base directory containing source datasets (default: ./data)")
    parser.add_argument("--run-dir", type=str, default="./run",
                        help="Directory for storing pipeline outputs (default: ./run)")
    parser.add_argument("--models-dir", type=str, default=None,
                        help="Directory containing embedding model weights; will be auto-detected if not specified")
    parser.add_argument("--dataset", type=str, default="summaries",
                        help="Dataset subdirectory under data-dir containing documents to process (default: summaries)")

    # Model selection
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Embedding models to process; defaults to SapBERT, BGE, BioClinicalBERT, BlueBERT, E5")
    parser.add_argument("--single-model", type=str, default=None,
                        help="Run only one specific model (overrides --models)")

    # Index configuration
    parser.add_argument("--index-type", type=str, choices=["flat", "hnsw"], default="hnsw",
                        help="FAISS index type: flat (exact, slower) or hnsw (approximate, faster) (default: hnsw)")

    # HNSW parameters
    parser.add_argument("--m-value", type=int, default=DEFAULT_M_VALUE,
                        help=f"M parameter for HNSW index; more connections improve accuracy but increase memory usage (default: {DEFAULT_M_VALUE})")
    parser.add_argument("--ef-construction", type=int, default=DEFAULT_EF_CONSTRUCTION,
                        help=f"EF construction parameter for HNSW; higher values improve index quality (default: {DEFAULT_EF_CONSTRUCTION})")
    parser.add_argument("--k-value", type=int, default=DEFAULT_K_VALUE,
                        help=f"Number of nearest neighbors to retrieve per query (default: {DEFAULT_K_VALUE})")

    # Processing options
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding generation phase (use existing embeddings)")
    parser.add_argument("--skip-indexing", action="store_true",
                        help="Skip index building phase (use existing indices)")
    parser.add_argument("--skip-search", action="store_true",
                        help="Skip search execution phase")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set logging verbosity level (default: INFO)")

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
    os.makedirs(os.path.join(args.run_dir, "indices"), exist_ok=True)
    os.makedirs(os.path.join(args.run_dir, "results"), exist_ok=True)

    return args


def run_embedding(model_name: str, model_path: str, data_dir: str, run_dir: str,
                  dataset: str, batch_size: int) -> bool:
    """Run embedding generation for a single model.

    This function executes the embedding generation process for a specific model,
    creating vectors for both trials and patients.

    Args:
        model_name: Name of the embedding model
        model_path: Path to the model
        data_dir: Base data directory
        run_dir: experiment directory
        dataset: Dataset to process
        batch_size: Batch size for processing

    Returns:
        True if embedding generation succeeded, False otherwise
    """
    logging.info(f"Generating embeddings for {model_name}...")

    cmd = [
        "trialmesh-embed",
        "--model-path", model_path,
        "--batch-size", str(batch_size),
        "--normalize",
        "--data-dir", data_dir,
        "--dataset", dataset,
        "--output-dir", os.path.join(run_dir, f"{dataset}_embeddings", model_name)
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


def build_index(model_name: str, run_dir: str, dataset: str,
                index_type: str, m_value: int = DEFAULT_M_VALUE,
                ef_construction: int = DEFAULT_EF_CONSTRUCTION) -> bool:
    """Build FAISS index for a model.

    This function creates a FAISS index from pre-generated embeddings,
    configuring it with the specified parameters.

    Args:
        model_name: Name of the embedding model
        run_dir: experiment directory
        dataset: Dataset to process
        index_type: Type of FAISS index to build (flat, hnsw)
        m_value: Number of connections per layer for HNSW index
        ef_construction: Size of dynamic candidate list for HNSW

    Returns:
        True if index building succeeded, False otherwise
    """
    logging.info(f"Building {index_type} index for {model_name}...")

    # Set paths
    embeddings_dir = os.path.join(run_dir, f"{dataset}_embeddings", model_name)
    trial_embeddings = os.path.join(embeddings_dir, "trial_embeddings.npy")
    index_file = os.path.join(run_dir, "indices", f"{model_name}_trials_{index_type}.index")

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


def run_search(model_name: str, run_dir: str, dataset: str,
               index_type: str, k_value: int = DEFAULT_K_VALUE) -> bool:
    """Run vector search for a model.

    This function performs similarity search using a pre-built FAISS index,
    finding the most relevant trials for each patient.

    Args:
        model_name: Name of the embedding model
        run_dir: experiment directory
        dataset: Dataset to process
        index_type: Type of FAISS index to use (flat, hnsw)
        k_value: Number of results to return per query

    Returns:
        True if search succeeded, False otherwise
    """
    logging.info(f"Running search for {model_name} using {index_type} index...")

    # Set paths to use run_dir
    embeddings_dir = os.path.join(run_dir, f"{dataset}_embeddings", model_name)
    patient_embeddings = os.path.join(embeddings_dir, "patient_embeddings.npy")
    index_file = os.path.join(run_dir, "indices", f"{model_name}_trials_{index_type}.index")
    results_file = os.path.join(run_dir, "results", f"{model_name}_{index_type}_search_results.json")

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
    """Run the complete pipeline for all specified models.

    This function orchestrates the full retrieval pipeline for multiple models:
    1. Generate embeddings
    2. Build indices
    3. Run searches

    It tracks successes and failures and provides a summary report.

    Args:
        args: Command-line arguments namespace
    """
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
            if not run_embedding(model_name, model_path, args.data_dir, args.run_dir, args.dataset, batch_size):
                logging.error(f"Skipping {model_name} due to embedding generation failure")
                failed_models.append(model_name)
                continue

        # 2. Build index (if not skipped)
        if not args.skip_indexing:
            if not build_index(
                    model_name,
                    args.run_dir,
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
                    args.run_dir,
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
    logging.info(f"{args.run_dir}/results/")


def main():
    """Run the complete TrialMesh retrieval pipeline.

    This command orchestrates the full pipeline for clinical trial retrieval:
    1. Generate embeddings for trials and patients using specified models
    2. Build FAISS indices (flat or HNSW) for efficient similarity search
    3. Search for relevant trials for each patient query

    The pipeline can be configured to skip certain stages if needed, and
    supports multiple embedding models and index types.
    """
    args = parse_args()
    setup_logging(args.log_level)
    run_pipeline(args)


def cli_main():
    """Entry point for console script."""
    main()


if __name__ == "__main__":
    main()