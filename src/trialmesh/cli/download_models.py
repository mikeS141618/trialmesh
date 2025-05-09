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
    """Parse command line arguments.

    Returns:
        Namespace containing the parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Download models for TrialMesh from HuggingFace"
    )

    # Output directory
    parser.add_argument("--output-dir", type=str, default="./models",
                        help="Directory where downloaded models will be stored (default: ./models)")

    # Model selection groups
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true",
                       help="Download all available models (LLMs and embedding models)")
    group.add_argument("--llms", action="store_true",
                       help="Download only LLM models (Llama variants)")
    group.add_argument("--embeddings", action="store_true",
                       help="Download only embedding models (BERT variants, E5, BGE)")
    group.add_argument("--models", type=str, nargs="+",
                       help="Download specific models by name (space-separated list)")

    # List available models
    parser.add_argument("--list", action="store_true",
                        help="List available models with descriptions without downloading anything")

    # Concurrent downloads
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of concurrent downloads to run in parallel (default: 1)")

    # Force download even if exists
    parser.add_argument("--force", action="store_true",
                        help="Force download even if model directory already exists")

    # Resume incomplete downloads
    parser.add_argument("--resume", action="store_true",
                        help="Resume incomplete downloads instead of starting over")

    # General options
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set logging verbosity level (default: INFO)")

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


def download_model(model_name: str, repo_id: str, output_dir: str,
                   local_dir: str = None, force: bool = False,
                   resume: bool = False) -> Tuple[str, bool]:
    """Download a model from HuggingFace.

    Args:
        model_name: Short name of the model
        repo_id: HuggingFace repository ID
        output_dir: Directory to store downloaded models
        local_dir: Local directory name (defaults to model_name)
        force: Whether to force download even if directory exists
        resume: Whether to resume an incomplete download

    Returns:
        Tuple of (model_name, success_status)
    """
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
    """Download selected models based on arguments.

    This function handles the downloading of multiple models, with options for
    parallel downloads and progress reporting. It manages the entire download
    process including:
    1. Finding models to download based on user selection
    2. Creating appropriate directories
    3. Downloading models (potentially in parallel)
    4. Summarizing successes and failures

    Args:
        args: Command-line arguments namespace containing model selection options,
              output directory, workers count, and other download parameters
    """
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
    """Check if huggingface-cli is available.

    This function verifies that the huggingface-cli tool is installed and
    functioning, which is required for model downloads.

    Returns:
        True if huggingface-cli is installed and working, False otherwise
    """
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