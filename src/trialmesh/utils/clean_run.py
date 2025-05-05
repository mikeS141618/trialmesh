#!/usr/bin/env python
# src/trialmesh/utils/clean_run.py

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict


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


def clean_directory(directory: Path, dry_run: bool = False) -> bool:
    """Remove a directory and its contents.

    Args:
        directory: Path to the directory to remove
        dry_run: If True, only log what would be removed without deleting

    Returns:
        True if directory was successfully removed or would be in dry run,
        False otherwise
    """
    if not directory.exists():
        logging.info(f"Directory does not exist (skipping): {directory}")
        return False

    try:
        if dry_run:
            logging.info(f"Would remove directory: {directory}")
            return True
        else:
            shutil.rmtree(directory)
            logging.info(f"Removed directory: {directory}")
            return True
    except PermissionError:
        logging.error(f"Permission denied when removing: {directory}")
        return False
    except Exception as e:
        logging.error(f"Error removing directory {directory}: {e}")
        return False


def confirm_action(message: str) -> bool:
    """Ask for user confirmation.

    Args:
        message: Confirmation message to display

    Returns:
        True if user confirmed, False otherwise
    """
    response = input(f"{message} (y/N): ").strip().lower()
    return response in ('y', 'yes')


def resolve_directories(args) -> Dict[str, List[Path]]:
    """Resolve directories to clean based on arguments.

    This function determines which directories should be cleaned based on
    the command-line arguments provided.

    Args:
        args: Command-line arguments namespace

    Returns:
        Dictionary mapping category names to lists of directories to clean
    """
    dirs_by_category = {
        "cache": [],
        "summaries": [],
        "embeddings": [],
        "indices": [],
        "results": [],
        "matches": [],  # Add new category for matcher results
    }

    # Base data directory
    data_dir = Path(args.data_dir)

    # Cache directories
    if args.clean_cache or args.clean_all:
        # LLM response cache
        llm_cache_dir = Path(args.llm_cache_dir)

        # Matcher cache
        matcher_cache_dir = Path(args.matcher_cache_dir)

        if args.all_caches:
            # If all_caches is True, we'll clean the parent cache directory
            # Get the parent cache directory (typically ./cache)
            parent_cache_dir = llm_cache_dir.parent
            dirs_by_category["cache"].append(parent_cache_dir)
        else:
            # Add specific cache directories
            dirs_by_category["cache"].append(llm_cache_dir)
            dirs_by_category["cache"].append(matcher_cache_dir)

    # Summaries directories
    if args.clean_summaries or args.clean_all:
        if args.dataset:
            summaries_dir = data_dir / f"{args.dataset}_summaries"
        else:
            summaries_dir = data_dir / "sigir2016/summaries"
        dirs_by_category["summaries"].append(summaries_dir)

    # Embeddings directories
    if args.clean_embeddings or args.clean_all:
        if args.dataset:
            embeddings_dir = data_dir / f"{args.dataset}_embeddings"
            if args.model_name:
                embeddings_dir = embeddings_dir / args.model_name
        else:
            embeddings_dir = data_dir / "sigir2016/summaries_embeddings"
        dirs_by_category["embeddings"].append(embeddings_dir)

    # Indices directories
    if args.clean_indices or args.clean_all:
        indices_dir = data_dir / "sigir2016/indices"
        if args.model_name:
            # Look for specific model indices
            indices_pattern = f"{args.model_name}_*.index*"
            # We'll handle these files individually
            for index_file in data_dir.glob(f"**/indices/{indices_pattern}"):
                dirs_by_category["indices"].append(index_file)
        else:
            dirs_by_category["indices"].append(indices_dir)

    # Results directories
    if args.clean_results or args.clean_all:
        results_dir = data_dir / "sigir2016/results"
        if args.model_name:
            # Look for specific model results
            results_pattern = f"{args.model_name}_*.json"
            # We'll handle these files individually
            for result_file in data_dir.glob(f"**/results/{results_pattern}"):
                dirs_by_category["results"].append(result_file)
        else:
            dirs_by_category["results"].append(results_dir)

    # Matcher results directories
    if args.clean_matches or args.clean_all:
        matches_dir = data_dir / "sigir2016/matched"
        if args.patient_id:
            # Look for specific patient matches
            matches_pattern = f"*{args.patient_id}*.json"
            # We'll handle these files individually
            for match_file in data_dir.glob(f"**/matched/{matches_pattern}"):
                dirs_by_category["matches"].append(match_file)
        else:
            dirs_by_category["matches"].append(matches_dir)

    return dirs_by_category


def clean_run(args):
    """Clean directories based on arguments.

    This function orchestrates the cleaning process, confirming with the
    user if necessary and handling both files and directories.

    Args:
        args: Command-line arguments namespace
    """
    # Resolve directories to clean
    dirs_by_category = resolve_directories(args)

    # Flatten all directories to clean
    all_dirs = []
    for category, dirs in dirs_by_category.items():
        all_dirs.extend(dirs)

    if not all_dirs:
        logging.warning("No directories selected for cleaning.")
        return

    # Confirm action if not forced
    if not args.force and not args.dry_run:
        print("\nDirectories to clean:")
        for category, dirs in dirs_by_category.items():
            if dirs:
                print(f"\n{category.upper()}:")
                for d in dirs:
                    print(f"  - {d}")

        if not confirm_action("\nDo you want to continue?"):
            logging.info("Operation cancelled by user.")
            return

    # Clean directories
    for category, dirs in dirs_by_category.items():
        if dirs:
            logging.info(f"Cleaning {category} directories...")
            for directory in dirs:
                if directory.is_file():
                    # Handle individual files (like indices and results)
                    try:
                        if args.dry_run:
                            logging.info(f"Would remove file: {directory}")
                        else:
                            directory.unlink()
                            logging.info(f"Removed file: {directory}")
                    except Exception as e:
                        logging.error(f"Error removing file {directory}: {e}")
                else:
                    # Handle directories
                    clean_directory(directory, args.dry_run)


def main():
    """Command-line interface for cleaning runs."""
    parser = argparse.ArgumentParser(
        description="Clean TrialMesh directories for cache, summaries, embeddings, indices, results, and matches"
    )

    # Base directories
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Base directory containing datasets (default: ./data)")
    parser.add_argument("--llm-cache-dir", type=str, default="./cache/llm_responses",
                        help="Directory containing cached LLM responses (default: ./cache/llm_responses)")
    parser.add_argument("--matcher-cache-dir", type=str, default="./cache/matcher",
                        help="Directory containing cached matcher results (default: ./cache/matcher)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (e.g., sigir2016/processed_cut) for selective cleaning")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Embedding model name for selective cleaning (e.g., SapBERT)")
    parser.add_argument("--patient-id", type=str, default=None,
                        help="Patient ID for selectively cleaning specific match results")

    # Cleaning categories
    parser.add_argument("--clean-cache", action="store_true",
                        help="Clean LLM and matcher response caches to force regeneration")
    parser.add_argument("--clean-summaries", action="store_true",
                        help="Clean LLM-generated summaries to force regeneration")
    parser.add_argument("--clean-embeddings", action="store_true",
                        help="Clean vector embeddings to force regeneration")
    parser.add_argument("--clean-indices", action="store_true",
                        help="Clean FAISS indices to force rebuilding")
    parser.add_argument("--clean-results", action="store_true",
                        help="Clean search results to force re-searching")
    parser.add_argument("--clean-matches", action="store_true",
                        help="Clean trial matcher results to force re-matching")
    parser.add_argument("--clean-all", action="store_true",
                        help="Clean all categories (cache, summaries, embeddings, indices, results, matches)")
    parser.add_argument("--all-caches", action="store_true",
                        help="Remove the entire cache directory instead of specific subdirectories")

    # General options
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be removed without actually deleting any files")
    parser.add_argument("--force", action="store_true",
                        help="Skip confirmation prompt and proceed with deletion")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set logging verbosity level (default: INFO)")

    args = parser.parse_args()
    setup_logging(args.log_level)

    # If no cleaning option specified, default to basic clean
    if not (args.clean_cache or args.clean_summaries or args.clean_embeddings or
            args.clean_indices or args.clean_results or args.clean_matches or args.clean_all):
        args.clean_cache = True
        args.clean_summaries = True
        logging.info("No specific cleaning option selected, defaulting to clean cache and summaries.")

    clean_run(args)


def cli_main():
    """Entry point for the console script"""
    main()


if __name__ == "__main__":
    main()