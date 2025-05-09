#!/usr/bin/env python3
# src/trialmesh/evaluation/evaluate_results.py

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
        description="Evaluate search results against gold standard"
    )

    # Data directories
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Base directory containing source datasets (default: ./data)")
    parser.add_argument("--run-dir", type=str, default="./run",
                        help="Directory for storing pipeline outputs (default: ./run)")
    parser.add_argument("--dataset", type=str, default="processed",
                        help="Dataset path containing gold standard data (default: processed)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory containing search results to evaluate; defaults to {run-dir}/results")

    # Models to evaluate
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Specific models to evaluate; if omitted, evaluates all models with results")

    # Index type (for filtering result files)
    parser.add_argument("--index-type", type=str, default=None, choices=["flat", "hnsw"],
                        help="Evaluate only results from this index type (flat or hnsw)")

    # Output options
    parser.add_argument("--output-file", type=str, default=None,
                        help="Save evaluation metrics to CSV file for further analysis; saved to {run-dir}/evaluation/ if path is relative")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate comparison plots of model performance")

    # General options
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set logging verbosity level (default: INFO)")

    args = parser.parse_args()

    # Set default results directory if not specified
    if args.results_dir is None:
        args.results_dir = os.path.join(args.run_dir, "results")

    # Add evaluation directory to run_dir
    if args.output_file:
        # If not an absolute path, make it relative to run_dir/evaluation
        if not os.path.isabs(args.output_file):
            args.output_file = os.path.join(args.run_dir, "evaluation", args.output_file)
            # Ensure the evaluation directory exists
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    return args


def load_gold_data(data_dir: str, dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load gold standard data files.

    This function loads the relevance judgments, queries, and corpus
    for evaluation purposes.

    Args:
        data_dir: Base data directory
        dataset: Dataset path relative to data_dir

    Returns:
        Tuple of DataFrames (relevance_judgments, queries, corpus)
    """
    dataset_path = os.path.join(data_dir, dataset)

    # Load relevance judgments
    tsv_file = os.path.join(dataset_path, "test.tsv")
    logging.info(f"Loading relevance judgments from {tsv_file}")
    df_tsv = pd.read_csv(tsv_file, sep='\t')

    # Load queries
    queries_file = os.path.join(dataset_path, "queries.jsonl")
    logging.info(f"Loading queries from {queries_file}")
    df_queries = pd.read_json(queries_file, lines=True)

    # Load corpus
    corpus_file = os.path.join(dataset_path, "corpus.jsonl")
    logging.info(f"Loading corpus from {corpus_file}")
    df_corpus = pd.read_json(corpus_file, lines=True)

    logging.info(f"df_tsv len:{len(df_tsv)} df_queries len:{len(df_queries)} df_corpus len:{len(df_corpus)}")

    return df_tsv, df_queries, df_corpus


def load_search_results(results_file: str) -> pd.DataFrame:
    """Load search results from a JSON file.

    Args:
        results_file: Path to the search results JSON file

    Returns:
        DataFrame containing the search results
    """
    logging.info(f"Loading search results from {results_file}")
    try:
        return pd.read_json(results_file)
    except Exception as e:
        logging.error(f"Error loading results file {results_file}: {e}")
        return pd.DataFrame()


def flatten_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the nested results into a long DataFrame.

    Args:
        results_df: DataFrame with nested search results

    Returns:
        Flattened DataFrame with one row per (query, document) pair
    """
    dfs = []
    for _, row in results_df.iterrows():
        for result in row['results']:
            dfs.append({
                'query-id': row['query_id'],
                'corpus-id': result['doc_id'],
                'score': result['score']
            })
    logging.info(f"results_df len:{len(dfs)}")
    return pd.DataFrame(dfs)


def per_query_stats(df_tsv: pd.DataFrame, df_results_long: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-query statistics.

    This function computes detailed statistics for each query, including
    the number of relevant documents found and missed at different
    relevance levels.

    Args:
        df_tsv: DataFrame with relevance judgments
        df_results_long: Flattened DataFrame of search results

    Returns:
        DataFrame with per-query statistics
    """
    df_tsv = df_tsv.copy()
    df_tsv['label_1'] = (df_tsv['score'] == 1).astype(int)
    df_tsv['label_2'] = (df_tsv['score'] == 2).astype(int)

    # Create a set of predicted pairs for fast lookup
    pred_pairs = set(zip(df_results_long['query-id'], df_results_long['corpus-id']))

    # Mark which relevant pairs were found in search results
    df_tsv['found'] = df_tsv.apply(
        lambda row: (row['query-id'], row['corpus-id']) in pred_pairs,
        axis=1
    )

    # Group by query and calculate statistics
    per_query = (
        df_tsv
        .groupby('query-id', as_index=True)
        .agg(
            total_score_1=('label_1', 'sum'),
            total_score_2=('label_2', 'sum'),
            found_score_1=('found', lambda x: int(((x) & (df_tsv.loc[x.index, 'label_1'] == 1)).sum())),
            found_score_2=('found', lambda x: int(((x) & (df_tsv.loc[x.index, 'label_2'] == 1)).sum()))
        )
    )

    # Calculate missing items and percentages
    per_query['missing_score_1'] = per_query['total_score_1'] - per_query['found_score_1']
    per_query['missing_score_2'] = per_query['total_score_2'] - per_query['found_score_2']

    # Calculate percentages, handling division by zero
    per_query['percent_missing_1'] = np.where(
        per_query['total_score_1'] == 0,
        0.0,
        100 * per_query['missing_score_1'] / per_query['total_score_1']
    ).round(1)

    per_query['percent_missing_2'] = np.where(
        per_query['total_score_2'] == 0,
        0.0,
        100 * per_query['missing_score_2'] / per_query['total_score_2']
    ).round(1)

    return per_query


def calculate_metrics(df_tsv: pd.DataFrame, df_results_long: pd.DataFrame) -> Dict:
    """Calculate various evaluation metrics.

    This function computes overall evaluation metrics, including:
    - Total relevant documents at each relevance level
    - Number and percentage of documents found/missed

    Args:
        df_tsv: DataFrame with relevance judgments
        df_results_long: Flattened DataFrame of search results

    Returns:
        Dictionary of evaluation metrics
    """
    # Get per-query statistics first
    per_query = per_query_stats(df_tsv, df_results_long)

    # Calculate overall statistics
    total_1 = int(per_query['total_score_1'].sum())
    total_2 = int(per_query['total_score_2'].sum())
    missing_1 = int(per_query['missing_score_1'].sum())
    missing_2 = int(per_query['missing_score_2'].sum())
    found_1 = total_1 - missing_1
    found_2 = total_2 - missing_2

    # Calculate percentages, handling division by zero
    percent_missing_1 = (missing_1 / total_1 * 100) if total_1 > 0 else 0
    percent_missing_2 = (missing_2 / total_2 * 100) if total_2 > 0 else 0
    percent_found_1 = (found_1 / total_1 * 100) if total_1 > 0 else 0
    percent_found_2 = (found_2 / total_2 * 100) if total_2 > 0 else 0

    # Create overall metrics
    metrics = {
        "Total Score 1": total_1,
        "Total Score 2": total_2,
        "Found Score 1": found_1,
        "Found Score 2": found_2,
        "Missing Score 1": missing_1,
        "Missing Score 2": missing_2,
        "Percent Found 1": round(percent_found_1, 1),
        "Percent Found 2": round(percent_found_2, 1),
        "Percent Missing 1": round(percent_missing_1, 1),
        "Percent Missing 2": round(percent_missing_2, 1),
    }

    return metrics


def analyze_model(model_name: str, results_file: str, df_tsv: pd.DataFrame,
                  df_queries: pd.DataFrame, df_corpus: pd.DataFrame) -> Dict:
    """Analyze a single model's search results.

    This function evaluates the performance of a specific model by:
    1. Loading its search results
    2. Computing evaluation metrics
    3. Logging key performance indicators

    Args:
        model_name: Name of the model to analyze
        results_file: Path to the model's search results
        df_tsv: DataFrame with relevance judgments
        df_queries: DataFrame with queries
        df_corpus: DataFrame with corpus

    Returns:
        Dictionary with model name and evaluation metrics
    """
    logging.info(f"Analyzing model: {model_name}")

    # Load and process results
    results_df = load_search_results(results_file)
    if results_df.empty:
        logging.warning(f"Skipping {model_name} due to empty results")
        return None

    df_results_long = flatten_results(results_df)
    metrics = calculate_metrics(df_tsv, df_results_long)

    # Log key metrics
    logging.info(f"  Total Score 1: {metrics['Total Score 1']}")
    logging.info(f"  Total Score 2: {metrics['Total Score 2']}")
    logging.info(f"  Found Score 1: {metrics['Found Score 1']} ({metrics['Percent Found 1']}%)")
    logging.info(f"  Found Score 2: {metrics['Found Score 2']} ({metrics['Percent Found 2']}%)")

    # Return complete results
    return {
        "Model": model_name,
        **metrics
    }


def find_result_files(results_dir: str, models: List[str] = None,
                      index_type: str = None) -> Dict[str, str]:
    """Find result files to evaluate.

    This function locates search result files that match the specified
    criteria for evaluation.

    Args:
        results_dir: Directory containing result files
        models: Specific models to evaluate (None for all)
        index_type: Index type to filter by (None for all)

    Returns:
        Dictionary mapping model names to result file paths
    """
    result_files = {}

    # List all JSON files in the results directory
    for file in os.listdir(results_dir):
        if not file.endswith(".json"):
            continue

        # Skip if index type doesn't match
        if index_type and index_type not in file:
            continue

        # Extract model name from filename
        # Initialize model_name to None first
        model_name = None

        # Format: modelname_indextype_search_results.json or modelname_search_results.json
        file_parts = file.split("_")

        # Try to extract model name based on common patterns
        if len(file_parts) >= 3:
            if "search" in file_parts and "results" in file_parts:
                # Handle case with index type in filename
                if len(file_parts) >= 4 and file_parts[1] in ["flat", "hnsw"]:
                    model_name = file_parts[0]
                else:
                    # No index type in filename
                    model_name = file_parts[0]

        # If we couldn't determine model name, use filename without extension as fallback
        if not model_name:
            model_name = file.rsplit(".", 1)[0]

        # Skip if not in requested models
        if models and model_name not in models:
            continue

        result_files[model_name] = os.path.join(results_dir, file)

    return result_files


def generate_visualizations(summary_df: pd.DataFrame, output_prefix: str = None):
    """Generate visualization plots.

    This function creates bar plots comparing model performance across
    different relevance levels.

    Args:
        summary_df: DataFrame with model performance summaries
        output_prefix: Prefix for saving output files (None for display only)
    """
    try:
        # Set style
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 8))

        # Sort by performance on Score 2 (more relevant)
        plot_df = summary_df.sort_values(by='Percent Found 2', ascending=False)

        # Create bar plot for percentage found
        ax = sns.barplot(
            x='Model',
            y='Percent Found 2',
            data=plot_df,
            palette='viridis',
            label='Score 2 (High Relevance)'
        )

        # Add bars for score 1
        sns.barplot(
            x='Model',
            y='Percent Found 1',
            data=plot_df,
            palette='muted',
            label='Score 1 (Relevant)',
            alpha=0.6
        )

        # Add labels and title
        plt.title('Model Performance: Percent of Relevant Trials Retrieved', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Percent Found (%)', fontsize=14)
        plt.ylim(0, 100)

        # Add legend
        plt.legend()

        # Rotate x labels for readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save or display
        if output_prefix:
            plt.savefig(f"{output_prefix}_performance.png", dpi=300, bbox_inches='tight')
            logging.info(f"Saved visualization to {output_prefix}_performance.png")
        else:
            plt.show()

    except Exception as e:
        logging.error(f"Error generating visualizations: {e}")


def evaluate_models(args):
    """Evaluate all models based on command line arguments.

    This function orchestrates the evaluation process:
    1. Loading gold standard data
    2. Finding result files to evaluate
    3. Analyzing each model's performance
    4. Creating summary tables and visualizations
    5. Saving results to files if requested

    Args:
        args: Command-line arguments namespace

    Returns:
        DataFrame with model performance summaries
    """
    # Load gold standard data
    df_tsv, df_queries, df_corpus = load_gold_data(args.data_dir, args.dataset)

    # Find result files to evaluate
    result_files = find_result_files(args.results_dir, args.models, args.index_type)

    if not result_files:
        logging.error(f"No result files found in {args.results_dir}")
        if args.models:
            logging.error(f"Requested models: {args.models}")
        if args.index_type:
            logging.error(f"Requested index type: {args.index_type}")
        return

    logging.info(f"Found {len(result_files)} result files to evaluate")

    # Analyze each model
    summary_list = []
    for model_name, results_file in result_files.items():
        result = analyze_model(model_name, results_file, df_tsv, df_queries, df_corpus)
        if result:
            summary_list.append(result)

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_list)

    # Sort by performance on Score 2 (more relevant)
    summary_df = summary_df.sort_values(by='Percent Found 2', ascending=False)

    # Display summary table
    print("\n=== Model Performance Summary ===")
    print(summary_df.to_string(index=False))

    # Save to CSV if requested
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        summary_df.to_csv(args.output_file, index=False)
        logging.info(f"Saved evaluation results to {args.output_file}")

    # Generate visualizations if requested
    if args.visualize:
        output_prefix = args.output_file.rsplit('.', 1)[0] if args.output_file else None
        generate_visualizations(summary_df, output_prefix)

    return summary_df


def main():
    """Main entry point for the script."""
    args = parse_args()
    setup_logging(args.log_level)
    evaluate_models(args)


def cli_main():
    """Entry point for console script."""
    main()


if __name__ == "__main__":
    main()