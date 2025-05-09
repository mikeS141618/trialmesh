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
from trialmesh.utils.prompt_registry import PromptRegistry


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
    parser.add_argument("--model-path", type=str,
                        help="Path to the LLaMA model directory containing model weights")
    parser.add_argument("--tensor-parallel-size", type=int, default=4,
                        help="Number of GPUs to use for tensor-parallel inference (default: 4)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Maximum number of tokens to generate in LLM responses (default: 1024)")
    parser.add_argument("--max-model-len", type=int, default=2048,
                        help="Maximum model context length for input+output tokens combined (default: 2048)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of trial-patient pairs to evaluate in each batch (default: 8)")

    # Data paths
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Base directory containing source datasets (default: ./data)")
    parser.add_argument("--run-dir", type=str, default="./run",
                        help="Directory for storing pipeline outputs (default: ./run)")
    parser.add_argument("--search-results", type=str,
                        default="results/bge-large-en-v1.5_hnsw_search_results.json",
                        help="Path to search results JSON file relative to run-dir")
    parser.add_argument("--patient-summaries", type=str, default="summaries/patient_summary.jsonl",
                        help="Path to patient summaries JSONL file relative to run-dir")
    parser.add_argument("--trials-path", type=str, default="processed/corpus.jsonl",
                        help="Path to trials JSONL file relative to data-dir")
    parser.add_argument("--output-file", type=str, default="matched/trial_matches.json",
                        help="Path for saving match results relative to run-dir")
    parser.add_argument("--cache-dir", type=str, default="./cache/matcher",
                        help="Directory for caching LLM responses")
    parser.add_argument("--prompt-dir", type=str, default="./prompts",
                        help="Directory containing prompt text files")

    # Prompt configuration
    parser.add_argument("--exclusion-prompt", type=str, default="exclusion_filter_sigir2016",
                        help="Name of the prompt for exclusion filtering (default: exclusion_filter_sigir2016)")
    parser.add_argument("--inclusion-prompt", type=str, default="inclusion_filter_sigir2016",
                        help="Name of the prompt for inclusion filtering (default: inclusion_filter_sigir2016)")
    parser.add_argument("--scoring-prompt", type=str, default="final_match_scoring_sigir2016",
                        help="Name of the prompt for final scoring (default: final_match_scoring_sigir2016)")
    parser.add_argument("--list-prompts", action="store_true",
                        help="List all available prompts in the prompt directory and exit")

    # Processing options
    parser.add_argument("--include-all-trials", action="store_true",
                        help="Include all trials in output with their filtering status")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Maximum number of trials to evaluate per patient")
    parser.add_argument("--skip-exclusion", action="store_true",
                        help="Skip exclusion criteria filtering step")
    parser.add_argument("--skip-inclusion", action="store_true",
                        help="Skip inclusion criteria filtering step")
    parser.add_argument("--skip-scoring", action="store_true",
                        help="Skip final detailed scoring step")
    parser.add_argument("--patient-ids", type=str, nargs="+",
                        help="Process only specific patient IDs (space-separated list)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set logging verbosity level (default: INFO)")

    args = parser.parse_args()
    return args


def parse_prompt_spec(prompt_spec: str) -> tuple:
    """Parse a prompt specification with optional token count and temperature.

    Format: prompt_name[:max_tokens[:temperature]]

    Args:
        prompt_spec: String in format "name:tokens:temp" or "name:tokens" or just "name"

    Returns:
        Tuple of (prompt_name, max_tokens, temperature) with None for unspecified values
    """

    parts = prompt_spec.split(':')
    prompt_name = parts[0]
    max_tokens = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
    temperature = float(parts[2]) if len(parts) > 2 and parts[2] else None

    return prompt_name, max_tokens, temperature


def main():
    """Main entry point for the trial matching script.

    This function orchestrates the complete trial matching process:
    1. Setting up logging and environment
    2. Initializing the LLM with the specified model
    3. Loading patient summaries and trial data
    4. Loading search results from vector retrieval
    5. Running the multi-stage matching pipeline
    6. Saving detailed match results to disk

    The matching process applies clinical reasoning to determine which
    trials are appropriate for each patient, with explanations for each
    decision. The process can optionally skip certain stages (exclusion,
    inclusion, or scoring) based on command-line arguments.
    """

    args = parse_args()
    setup_logging(args.log_level)

    # If just listing prompts, show them and exit
    if args.list_prompts:
        prompt_registry = PromptRegistry(args.prompt_dir)
        available_prompts = prompt_registry.list_available_prompts()
        print("Available prompts:")
        for prompt in available_prompts:
            print(f"  - {prompt}")
        return

    # Ensure model path is provided if not just listing prompts
    if not args.model_path:
        raise ValueError("--model-path is required when not using --list-prompts")

    # Create output directories
    output_path = os.path.join(args.run_dir, args.output_file)
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

    # Construct full paths for matcher
    patient_summaries_path = os.path.join(args.run_dir, args.patient_summaries)
    trials_path = os.path.join(args.data_dir, args.trials_path)

    # Initialize the matcher with full paths and prompt directory
    matcher = TrialMatcher(
        llm=llm,
        patient_summaries_path=patient_summaries_path,
        trials_path=trials_path,
        batch_size=args.batch_size,
        prompt_dir=args.prompt_dir,
    )

    # Load search results from run_dir
    search_results_path = os.path.join(args.run_dir, args.search_results)
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

    # Parse prompt specs for each stage
    exclusion_name, exclusion_tokens, exclusion_temp = parse_prompt_spec(args.exclusion_prompt)
    inclusion_name, inclusion_tokens, inclusion_temp = parse_prompt_spec(args.inclusion_prompt)
    scoring_name, scoring_tokens, scoring_temp = parse_prompt_spec(args.scoring_prompt)

    # Log parsed prompt configs
    logging.info(f"Using exclusion prompt: {exclusion_name} (max_tokens: {exclusion_tokens}, temp: {exclusion_temp})")
    logging.info(f"Using inclusion prompt: {inclusion_name} (max_tokens: {inclusion_tokens}, temp: {inclusion_temp})")
    logging.info(f"Using scoring prompt: {scoring_name} (max_tokens: {scoring_tokens}, temp: {scoring_temp})")

    # Run the matching process with specified prompts
    results = matcher.match(
        search_results=search_results,
        exclusion_prompt=exclusion_name,
        inclusion_prompt=inclusion_name,
        scoring_prompt=scoring_name,
        exclusion_max_tokens=exclusion_tokens,
        inclusion_max_tokens=inclusion_tokens,
        scoring_max_tokens=scoring_tokens,
        exclusion_temperature=exclusion_temp,
        inclusion_temperature=inclusion_temp,
        scoring_temperature=scoring_temp,
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