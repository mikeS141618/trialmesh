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
                        help="Path to the embedding model directory containing model weights")
    parser.add_argument("--model-type", type=str, default=None,
                        help="Type of embedding model to use; if not specified, will be auto-detected from path")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum sequence length for tokenization; longer texts will be truncated (default: 512)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for embedding generation; adjust based on GPU memory (default: 32)")
    parser.add_argument("--normalize", action="store_true",
                        help="L2-normalize output embeddings; recommended for cosine similarity search")
    parser.add_argument("--local-rank", type=int, default=-1,
                        help="Local rank for distributed training (passed by PyTorch launchers)")

    # GPU configuration
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g., cuda:0, cuda:1, cpu); defaults to CUDA if available")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use distributed processing across multiple GPUs using torch.distributed")

    # Input/output options
    parser.add_argument("--data-dir", type=str, default="./run",
                        help="Base directory containing  trialmesh-summarize (default: ./run)")
    parser.add_argument("--dataset", type=str, default="processed_summaries",
                        help="Dataset subdirectory under data-dir containing documents to embed (default: processed_summaries)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for saving embeddings; defaults to ./run/{dataset}_embeddings/{model-name}")

    # Processing options
    parser.add_argument("--skip-trials", action="store_true",
                        help="Skip embedding generation for trial documents")
    parser.add_argument("--skip-patients", action="store_true",
                        help="Skip embedding generation for patient queries")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set logging verbosity level (default: INFO)")

    args = parser.parse_args()

    # Set default output directory if not specified
    if args.output_dir is None:
        model_name = os.path.basename(args.model_path.rstrip("/"))
        args.output_dir = os.path.join("./run", f"{args.dataset}_embeddings", model_name)

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
        local_rank=args.local_rank,
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