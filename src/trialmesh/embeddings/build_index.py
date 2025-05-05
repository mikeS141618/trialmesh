# src/trialmesh/embeddings/build_index.py

import argparse
import logging
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any

from trialmesh.embeddings.index_builder import FaissIndexBuilder
from trialmesh.embeddings.query import FaissSearcher


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
    parser = argparse.ArgumentParser(description="Build and search FAISS indices")

    # Common arguments
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")

    # Create subparsers for build and search commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build a FAISS index")

    # Required arguments for build
    build_parser.add_argument("--embeddings", type=str, required=True,
                              help="Path to .npy file containing embeddings dictionary")
    build_parser.add_argument("--output", type=str, required=True,
                              help="Destination path for saving the FAISS index")
    build_parser.add_argument("--index-type", type=str, default="hnsw", choices=["flat", "ivf", "hnsw"],
                              help="FAISS index type: flat (exact, slowest), ivf (approximate, balanced), hnsw (fast, accurate) (default: flat)")
    build_parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2", "ip"],
                              help="Distance metric: cosine (angular), l2 (Euclidean), ip (inner product) (default: cosine)")
    build_parser.add_argument("--nlist", type=int, default=100,
                              help="Number of clusters for IVF index; higher values improve accuracy but slow indexing (default: 100)")
    build_parser.add_argument("--m", type=int, default=32,
                              help="Number of connections per layer for HNSW index; higher values increase accuracy and memory usage (default: 32)")
    build_parser.add_argument("--ef-construction", type=int, default=200,
                              help="Size of dynamic candidate list for HNSW; higher values improve index quality but slow construction (default: 200)")
    build_parser.add_argument("--normalize", action="store_true",
                              help="L2-normalize all vectors before indexing (automatically enabled for cosine metric)")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search a FAISS index")

    # Required arguments for search
    search_parser.add_argument("--index", type=str, required=True,
                               help="Path to the FAISS index file to be searched")
    search_parser.add_argument("--queries", type=str, required=True,
                               help="Path to .npy file containing query embeddings dictionary")
    search_parser.add_argument("--output", type=str, required=True,
                               help="Destination path for saving search results in JSON format")
    search_parser.add_argument("--k", type=int, default=100,
                               help="Number of nearest neighbors to retrieve per query (default: 100)")
    search_parser.add_argument("--query-ids", type=str, default=None, nargs="+",
                               help="Specific query IDs to search for; if omitted, all queries will be processed")
    search_parser.add_argument("--normalize", action="store_true",
                               help="L2-normalize query vectors before searching (automatically enabled for cosine indices)")

    args = parser.parse_args()

    # Validate command
    if args.command not in ["build", "search"]:
        parser.print_help()
        parser.error("Please specify a command: build or search")

    return args


def build_index(args):
    """Build a FAISS index from embeddings."""
    logging.info(f"Building {args.index_type} index with {args.metric} metric")

    # Create index builder
    builder = FaissIndexBuilder(
        index_type=args.index_type,
        metric=args.metric,
        nlist=args.nlist,
        m=args.m,
        ef_construction=args.ef_construction,
    )

    # Build index from embeddings file
    builder.build_from_file(args.embeddings, normalize=args.normalize)

    # Save the index
    builder.save_index(args.output)
    logging.info(f"Index saved to {args.output}")


def search_index(args):
    """Search a FAISS index with query embeddings."""
    logging.info(f"Searching index {args.index} with k={args.k}")

    # Load query embeddings
    logging.info(f"Loading query embeddings from {args.queries}")
    query_embeddings = np.load(args.queries, allow_pickle=True).item()

    if not isinstance(query_embeddings, dict):
        raise ValueError(f"Query embeddings file should contain a dictionary, got {type(query_embeddings)}")

    # Create searcher
    searcher = FaissSearcher(index_path=args.index)

    # Filter query IDs if specified
    if args.query_ids:
        query_ids = [qid for qid in args.query_ids if qid in query_embeddings]
        if not query_ids:
            raise ValueError("None of the specified query IDs were found in embeddings")
        logging.info(f"Searching for {len(query_ids)} specified queries")
    else:
        query_ids = list(query_embeddings.keys())
        logging.info(f"Searching for all {len(query_ids)} queries")

    # Perform search
    results = searcher.batch_search_by_id(
        query_ids=query_ids,
        embeddings=query_embeddings,
        k=args.k,
        normalize=args.normalize,
    )

    # Convert results to a list of dictionaries
    result_dicts = [result.to_dict() for result in results]

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result_dicts, f, indent=2)

    logging.info(f"Search results saved to {args.output}")


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    setup_logging(args.log_level)

    if args.command == "build":
        build_index(args)
    elif args.command == "search":
        search_index(args)


def cli_main():
    """Entry point for the console script"""
    main()


if __name__ == "__main__":
    main()