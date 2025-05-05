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
        description="Run trial-patient matching with LLM evaluation"
    )

    # Model configuration
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the LLaMA model directory containing model weights")
    parser.add_argument("--tensor-parallel-size", type=int, default=4,
                        help="Number of GPUs to use for tensor-parallel inference; adjust based on available hardware (default: 4)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Maximum number of tokens to generate in LLM responses (default: 1024)")
    parser.add_argument("--max-model-len", type=int, default=2048,
                        help="Maximum model context length for input+output tokens combined (default: 2048)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of trial-patient pairs to evaluate in each batch; adjust based on GPU memory (default: 8)")

    # Data paths
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Base directory containing datasets (default: ./data)")
    parser.add_argument("--search-results", type=str,
                        default="sigir2016/results/bge-large-en-v1.5_hnsw_search_results.json",
                        help="Path to search results JSON file relative to data-dir (default: sigir2016/results/bge-large-en-v1.5_hnsw_search_results.json)")
    parser.add_argument("--patient-summaries", type=str, default="sigir2016/summaries/patient_summaries.jsonl",
                        help="Path to patient summaries JSONL file relative to data-dir (default: sigir2016/summaries/patient_summaries.jsonl)")
    parser.add_argument("--trials-path", type=str, default="sigir2016/processed_cut/corpus.jsonl",
                        help="Path to trials JSONL file relative to data-dir (default: sigir2016/processed_cut/corpus.jsonl)")
    parser.add_argument("--output-file", type=str, default="sigir2016/matched/trial_matches.json",
                        help="Path for saving match results relative to data-dir (default: sigir2016/matched/trial_matches.json)")
    parser.add_argument("--cache-dir", type=str, default="./cache/matcher",
                        help="Directory for caching LLM responses to avoid redundant computation (default: ./cache/matcher)")

    # Processing options
    parser.add_argument("--include-all-trials", action="store_true",
                        help="Include all trials in output with their filtering status, instead of only final matches")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Maximum number of trials to evaluate per patient; None processes all retrieved trials")
    parser.add_argument("--skip-exclusion", action="store_true",
                        help="Skip exclusion criteria filtering step to evaluate all trials")
    parser.add_argument("--skip-inclusion", action="store_true",
                        help="Skip inclusion criteria filtering step")
    parser.add_argument("--skip-scoring", action="store_true",
                        help="Skip final detailed scoring step to save computation time")
    parser.add_argument("--patient-ids", type=str, nargs="+",
                        help="Process only specific patient IDs (space-separated list)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set logging verbosity level (default: INFO)")

    args = parser.parse_args()
    return args


def main():
    """Main entry point for the script.

    This function runs the trial-patient matching pipeline:
    1. Initializes the LLM for evaluation
    2. Sets up the matcher with appropriate data paths
    3. Loads search results as input
    4. Runs the multi-stage matching process
    5. Saves the detailed match results
    """
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