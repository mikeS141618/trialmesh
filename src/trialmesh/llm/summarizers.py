# src/trialmesh/llm/summarizers.py

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

from trialmesh.llm.llama_runner import LlamaRunner, LlamaResponse
from trialmesh.llm.prompt_runner import PromptRunner
from trialmesh.utils.prompt_registry import PromptRegistry


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of dictionaries, one for each line in the file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to a JSONL file.

    Args:
        data: List of dictionaries to save
        file_path: Path to save the JSONL file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


class Summarizer:
    """Generate summaries for trials and patients using LLM.

    This class handles the creation of detailed and condensed summaries
    for both clinical trials and patient records, using a large language
    model to extract and structure relevant information.

    Attributes:
        runner: LlamaRunner instance for generating text
        prompt_runner: PromptRunner for template-based generation
    """

    def __init__(
            self,
            model_path: str,
            cache_dir: str = None,
            tensor_parallel_size: int = 4,
            max_model_len: int = 2048,
            batch_size: int = 8,
    ):
        """Initialize the summarizer with model configuration."""
        self.runner = LlamaRunner(
            model_path=model_path,
            cache_dir=cache_dir,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            max_batch_size=batch_size,
        )
        self.prompt_runner = PromptRunner(self.runner)

    def summarize_trials(self, trials_path: str, output_dir: str,
                         batch_size: int = 8, max_tokens: int = 1024,
                         condensed_trial_only: bool = True) -> None:
        """Generate summaries for clinical trials.

        This method processes clinical trial documents to create:
        1. Full summaries with detailed structured information (optional)
        2. Condensed summaries optimized for embedding and retrieval

        Args:
            trials_path: Path to the trial documents JSONL file
            output_dir: Directory to save generated summaries
            batch_size: Number of trials to process in each batch
            max_tokens: Maximum tokens to generate per summary
            condensed_trial_only: Whether to generate only condensed summaries
        """
        logging.info(f"Loading trials from {trials_path}")
        logging.info(f"condensed_trial_only {condensed_trial_only} batch_size {batch_size}")
        trials = load_jsonl(trials_path)
        logging.info(f"Loaded {len(trials)} trials")

        # Prepare output directory
        os.makedirs(output_dir, exist_ok=True)
        full_summary_path = os.path.join(output_dir, "trial_summaries.jsonl")
        condensed_summary_path = os.path.join(output_dir, "trial_condensed.jsonl")

        # Process in batches
        all_full_summaries = []
        all_condensed_summaries = []

        for i in tqdm(range(0, len(trials), batch_size), desc="Processing trials"):
            batch = trials[i:i + batch_size]

            # Prepare variables for prompts
            variables_list = [{"trial_text": self._format_trial(trial)} for trial in batch]

            # Run full summaries only if needed
            full_summary_responses = []
            if not condensed_trial_only:
                full_summary_responses = self.prompt_runner.run_prompt_batch(
                    prompt_name="trial_summary_sigir2016",
                    variables_list=variables_list,
                    max_tokens=max_tokens,
                )

            # Run condensed summaries for embedding (always needed)
            condensed_summary_responses = self.prompt_runner.run_prompt_batch(
                prompt_name="trial_condensed_sigir2016",
                variables_list=variables_list,
                max_tokens=min(max_tokens, 512),
            )

            # Process and save results
            for j, trial in enumerate(batch):
                trial_id = trial["_id"]

                # Full summary (if not skipped)
                if not condensed_trial_only and full_summary_responses and full_summary_responses[j]:
                    full_summary = {
                        "_id": trial_id,
                        "summary": full_summary_responses[j].text,
                        "input_tokens": full_summary_responses[j].input_tokens,
                        "output_tokens": full_summary_responses[j].output_tokens
                    }
                    all_full_summaries.append(full_summary)

                # Condensed summary for embedding
                if condensed_summary_responses[j]:
                    condensed_summary = {
                        "_id": trial_id,
                        "summary": condensed_summary_responses[j].text,
                        "input_tokens": condensed_summary_responses[j].input_tokens,
                        "output_tokens": condensed_summary_responses[j].output_tokens
                    }
                    all_condensed_summaries.append(condensed_summary)

        # Save results
        if not condensed_trial_only and all_full_summaries:
            save_jsonl(all_full_summaries, full_summary_path)
            logging.info(f"Saved {len(all_full_summaries)} trial summaries to {full_summary_path}")

        save_jsonl(all_condensed_summaries, condensed_summary_path)
        logging.info(f"Saved {len(all_condensed_summaries)} condensed trial summaries to {condensed_summary_path}")

    def summarize_patients(self, patients_path: str, output_dir: str,
                           batch_size: int = 8, max_tokens: int = 1024) -> None:
        """Generate summaries for patient queries.

        This method processes patient records to create:
        1. Full summaries with detailed structured information
        2. Condensed summaries optimized for embedding and retrieval

        Args:
            patients_path: Path to the patient records JSONL file
            output_dir: Directory to save generated summaries
            batch_size: Number of patients to process in each batch
            max_tokens: Maximum tokens to generate per summary
        """
        logging.info(f"Loading patients from {patients_path}")
        patients = load_jsonl(patients_path)
        logging.info(f"Loaded {len(patients)} patients")

        # Prepare output directory
        os.makedirs(output_dir, exist_ok=True)
        full_summary_path = os.path.join(output_dir, "patient_summaries.jsonl")
        condensed_summary_path = os.path.join(output_dir, "patient_condensed.jsonl")

        # Process in batches
        all_full_summaries = []
        all_condensed_summaries = []

        for i in tqdm(range(0, len(patients), batch_size), desc="Processing patients"):
            batch = patients[i:i + batch_size]

            # Prepare variables for prompts
            variables_list = [{"patient_text": patient["text"]} for patient in batch]

            # Run full summaries
            full_summary_responses = self.prompt_runner.run_prompt_batch(
                prompt_name="patient_summary_sigir2016",
                variables_list=variables_list,
                max_tokens=max_tokens,
            )

            # Run condensed summaries for embedding
            condensed_summary_responses = self.prompt_runner.run_prompt_batch(
                prompt_name="patient_condensed_sigir2016",
                variables_list=variables_list,
                max_tokens=min(max_tokens, 512),
            )

            # Process and save results
            for j, patient in enumerate(batch):
                patient_id = patient["_id"]

                # Full summary
                if full_summary_responses[j]:
                    full_summary = {
                        "_id": patient_id,
                        "summary": full_summary_responses[j].text,
                        "input_tokens": full_summary_responses[j].input_tokens,
                        "output_tokens": full_summary_responses[j].output_tokens,
                    }
                    all_full_summaries.append(full_summary)

                # Condensed summary for embedding
                if condensed_summary_responses[j]:
                    condensed_summary = {
                        "_id": patient_id,
                        "summary": condensed_summary_responses[j].text,
                        "input_tokens": condensed_summary_responses[j].input_tokens,
                        "output_tokens": condensed_summary_responses[j].output_tokens,
                    }
                    all_condensed_summaries.append(condensed_summary)

        # Save all results
        save_jsonl(all_full_summaries, full_summary_path)
        save_jsonl(all_condensed_summaries, condensed_summary_path)
        logging.info(f"Saved {len(all_full_summaries)} patient summaries to {full_summary_path}")
        logging.info(f"Saved {len(all_condensed_summaries)} condensed patient summaries to {condensed_summary_path}")

    def _format_trial(self, trial: Dict[str, Any]) -> str:
        """Format trial data for LLM input."""
        metadata = trial.get("metadata", {})

        formatted_text = f"Title: {trial.get('title', '')}\n\n"

        if metadata.get("brief_summary"):
            formatted_text += f"Summary: {metadata.get('brief_summary', '')}\n\n"

        if metadata.get("detailed_description"):
            formatted_text += f"Description: {metadata.get('detailed_description', '')}\n\n"

        if metadata.get("inclusion_criteria"):
            formatted_text += f"Inclusion Criteria: {metadata.get('inclusion_criteria', '')}\n\n"

        if metadata.get("exclusion_criteria"):
            formatted_text += f"Exclusion Criteria: {metadata.get('exclusion_criteria', '')}\n\n"

        if metadata.get("diseases_list"):
            diseases = ", ".join(metadata.get("diseases_list", []))
            formatted_text += f"Conditions: {diseases}\n\n"

        if metadata.get("drugs_list"):
            drugs = ", ".join(metadata.get("drugs_list", []))
            formatted_text += f"Interventions: {drugs}\n\n"

        if metadata.get("phase"):
            formatted_text += f"Phase: {metadata.get('phase', '')}\n\n"

        return formatted_text.strip()


def main():
    """Command-line interface for running summarizations."""
    parser = argparse.ArgumentParser(description="Generate summaries for trials and patients")

    # Model configuration
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the LLaMA model directory containing model weights")
    parser.add_argument("--tensor-parallel-size", type=int, default=4,
                        help="Number of GPUs to use for tensor-parallel inference; adjust based on available hardware (default: 4)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Maximum number of tokens to generate per summary (default: 1024)")
    parser.add_argument("--max-model-len", type=int, default=2048,
                        help="Maximum model context length for input+output tokens combined (default: 2048)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of documents to process in each batch; adjust based on GPU memory (default: 8)")

    # Data paths
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Base directory containing datasets (default: ./data)")
    parser.add_argument("--dataset", type=str, default="sigir2016/processed_cut",
                        help="Dataset subdirectory under data-dir containing documents to summarize (default: sigir2016/processed_cut)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for saving summaries; defaults to {data-dir}/{dataset}_summaries")
    parser.add_argument("--cache-dir", type=str, default="./cache",
                        help="Directory for caching LLM responses to avoid redundant computation (default: ./cache)")

    # Processing options
    parser.add_argument("--skip-trials", action="store_true",
                        help="Skip trial document summarization")
    parser.add_argument("--skip-patients", action="store_true",
                        help="Skip patient query summarization")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set logging verbosity level (default: INFO)")
    parser.add_argument("--condensed-trial-only", action="store_true",
                        help="Generate only condensed trial summaries for embedding and skip full summaries to save time")

    args = parser.parse_args()

    # Set up logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, f"{args.dataset}_summaries")

    # Create summarizer
    summarizer = Summarizer(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
    )

    # Run summarizations
    data_path = os.path.join(args.data_dir, args.dataset)

    if not args.skip_trials:
        trials_path = os.path.join(data_path, "corpus.jsonl")
        summarizer.summarize_trials(
            trials_path=trials_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
        )

    if not args.skip_patients:
        patients_path = os.path.join(data_path, "queries.jsonl")
        summarizer.summarize_patients(
            patients_path=patients_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
        )

    logging.info("Summarization complete!")


def cli_main():
    """Entry point for the console script"""
    main()


if __name__ == "__main__":
    main()