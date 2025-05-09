# src/trialmesh/llm/summarizers.py
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm

from trialmesh.llm.llama_runner import LlamaRunner, LlamaResponse
from trialmesh.llm.prompt_runner import PromptRunner
from trialmesh.utils.prompt_registry import PromptRegistry
from trialmesh.utils.prompt_config import PromptConfig


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to a JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


class Summarizer:
    """Generate summaries for trials and patients using LLM.

    This class handles the creation of summaries for both clinical trials
    and patient records using multiple prompt configurations.
    """

    def __init__(
            self,
            model_path: str,
            prompt_dir: str = "./prompts",
            cache_dir: str = None,
            tensor_parallel_size: int = 4,
            max_model_len: int = 2048,
            batch_size: int = 8,
    ):
        """Initialize the summarizer with model and prompt configuration."""
        self.runner = LlamaRunner(
            model_path=model_path,
            cache_dir=cache_dir,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            max_batch_size=batch_size,
        )

        # Use the file-based prompt registry
        self.prompt_registry = PromptRegistry(prompt_dir)
        self.prompt_runner = PromptRunner(self.runner, self.prompt_registry)

    def summarize_trials(self, trials_path: str, output_dir: str,
                         prompt_configs: List[PromptConfig],
                         batch_size: int = 8) -> None:
        """Generate summaries for clinical trials using multiple prompts.

        Args:
            trials_path: Path to the trial documents JSONL file
            output_dir: Directory to save generated summaries
            prompt_configs: List of prompt configurations to run
            batch_size: Number of trials to process in each batch
        """
        logging.info(f"Loading trials from {trials_path}")
        trials = load_jsonl(trials_path)
        logging.info(f"Loaded {len(trials)} trials")

        # Prepare output directory
        os.makedirs(output_dir, exist_ok=True)

        # Process each prompt config
        for config in prompt_configs:
            logging.info(f"Processing trials with prompt: {config.name}, "
                         f"max_tokens: {config.max_tokens}, "
                         f"output_suffix: {config.output_suffix}")

            output_file = os.path.join(output_dir, f"trial_{config.output_suffix}.jsonl")
            all_summaries = []

            # Process in batches
            for i in tqdm(range(0, len(trials), batch_size),
                          desc=f"Processing trials with {config.name}"):
                batch = trials[i:i + batch_size]

                # Prepare variables for prompts
                variables_list = [{"trial_text": self._format_trial(trial)} for trial in batch]

                # Run the prompt for this batch
                responses = self.prompt_runner.run_prompt_batch(
                    prompt_name=config.name,
                    variables_list=variables_list,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                )

                # Process and save results
                for j, trial in enumerate(batch):
                    trial_id = trial["_id"]

                    if responses[j]:
                        summary = {
                            "_id": trial_id,
                            "summary": responses[j].text,
                            "input_tokens": responses[j].input_tokens,
                            "output_tokens": responses[j].output_tokens
                        }
                        all_summaries.append(summary)

            # Save results for this prompt
            save_jsonl(all_summaries, output_file)
            logging.info(f"Saved {len(all_summaries)} trial summaries to {output_file}")

    def summarize_patients(self, patients_path: str, output_dir: str,
                           prompt_configs: List[PromptConfig],
                           batch_size: int = 8) -> None:
        """Generate summaries for patient records using multiple prompts.

        Args:
            patients_path: Path to the patient records JSONL file
            output_dir: Directory to save generated summaries
            prompt_configs: List of prompt configurations to run
            batch_size: Number of patients to process in each batch
        """
        logging.info(f"Loading patients from {patients_path}")
        patients = load_jsonl(patients_path)
        logging.info(f"Loaded {len(patients)} patients")

        # Prepare output directory
        os.makedirs(output_dir, exist_ok=True)

        # Process each prompt config
        for config in prompt_configs:
            logging.info(f"Processing patients with prompt: {config.name}, "
                         f"max_tokens: {config.max_tokens}, "
                         f"output_suffix: {config.output_suffix}")

            output_file = os.path.join(output_dir, f"patient_{config.output_suffix}.jsonl")
            all_summaries = []

            # Process in batches
            for i in tqdm(range(0, len(patients), batch_size),
                          desc=f"Processing patients with {config.name}"):
                batch = patients[i:i + batch_size]

                # Prepare variables for prompts
                variables_list = [{"patient_text": patient["text"]} for patient in batch]

                # Run the prompt for this batch
                responses = self.prompt_runner.run_prompt_batch(
                    prompt_name=config.name,
                    variables_list=variables_list,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                )

                # Process and save results
                for j, patient in enumerate(batch):
                    patient_id = patient["_id"]

                    if responses[j]:
                        summary = {
                            "_id": patient_id,
                            "summary": responses[j].text,
                            "input_tokens": responses[j].input_tokens,
                            "output_tokens": responses[j].output_tokens
                        }
                        all_summaries.append(summary)

            # Save results for this prompt
            save_jsonl(all_summaries, output_file)
            logging.info(f"Saved {len(all_summaries)} patient summaries to {output_file}")

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
    parser.add_argument("--model-path", type=str,
                        help="Path to the LLaMA model directory containing model weights")
    parser.add_argument("--tensor-parallel-size", type=int, default=4,
                        help="Number of GPUs to use for tensor-parallel inference (default: 4)")
    parser.add_argument("--max-model-len", type=int, default=2048,
                        help="Maximum model context length for input+output tokens combined (default: 2048)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of documents to process in each batch (default: 8)")

    # Data paths
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Base directory containing source datasets (default: ./data)")
    parser.add_argument("--dataset", type=str, default="processed",
                        help="Dataset subdirectory under data-dir (default: processed)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for saving summaries (default: ./run/{dataset}_summaries)")
    parser.add_argument("--cache-dir", type=str, default="./cache/llm_responses",
                        help="Directory for caching LLM responses (default: ./cache/llm_responses)")
    parser.add_argument("--prompt-dir", type=str, default="./prompts",
                        help="Directory containing prompt text files (default: ./prompts)")

    # Processing options
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set logging verbosity level (default: INFO)")

    # New argument for patient prompts
    parser.add_argument("--patient-prompts", type=str, nargs="+", default=[],
                        help="List of patient prompt configs in format 'prompt_name:max_tokens[:temp]'")

    # New argument for trial prompts
    parser.add_argument("--trial-prompts", type=str, nargs="+", default=[],
                        help="List of trial prompt configs in format 'prompt_name:max_tokens[:temp]'")

    # Add capability to list available prompts
    parser.add_argument("--list-prompts", action="store_true",
                        help="List all available prompts in the prompt directory and exit")

    args = parser.parse_args()

    # Set up logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Create prompt registry to check available prompts
    prompt_registry = PromptRegistry(args.prompt_dir)

    # If requested, list available prompts and exit
    if args.list_prompts:
        available_prompts = prompt_registry.list_available_prompts()
        print("Available prompts:")
        for prompt in available_prompts:
            print(f"  - {prompt}")
        return

    # Ensure model path is provided if not just listing prompts
    if not args.model_path:
        parser.error("--model-path is required when not using --list-prompts")

    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join("./run", f"{args.dataset}_summaries")

    # Parse prompt configs
    patient_configs = []
    trial_configs = []

    for prompt_spec in args.patient_prompts:
        parts = prompt_spec.split(':')
        if len(parts) < 2:
            logging.error(f"Invalid prompt spec: {prompt_spec}, format should be 'name:tokens[:temp]'")
            continue

        prompt_name = parts[0]
        try:
            max_tokens = int(parts[1])
            temp = float(parts[2]) if len(parts) > 2 else 0.0
            patient_configs.append(PromptConfig(
                name=prompt_name,
                max_tokens=max_tokens,
                temperature=temp
            ))
        except ValueError as e:
            logging.error(f"Error parsing prompt spec {prompt_spec}: {e}")

    for prompt_spec in args.trial_prompts:
        parts = prompt_spec.split(':')
        if len(parts) < 2:
            logging.error(f"Invalid prompt spec: {prompt_spec}, format should be 'name:tokens[:temp]'")
            continue

        prompt_name = parts[0]
        try:
            max_tokens = int(parts[1])
            temp = float(parts[2]) if len(parts) > 2 else 0.0
            trial_configs.append(PromptConfig(
                name=prompt_name,
                max_tokens=max_tokens,
                temperature=temp
            ))
        except ValueError as e:
            logging.error(f"Error parsing prompt spec {prompt_spec}: {e}")

    # Create summarizer
    summarizer = Summarizer(
        model_path=args.model_path,
        prompt_dir=args.prompt_dir,
        cache_dir=args.cache_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
    )

    # Run summarizations
    data_path = os.path.join(args.data_dir, args.dataset)

    # Process trials if any trial prompts specified
    if trial_configs:
        trials_path = os.path.join(data_path, "corpus.jsonl")
        summarizer.summarize_trials(
            trials_path=trials_path,
            output_dir=args.output_dir,
            prompt_configs=trial_configs,
            batch_size=args.batch_size,
        )
    else:
        logging.info("No trial prompts specified, skipping trial summarization")

    # Process patients if any patient prompts specified
    if patient_configs:
        patients_path = os.path.join(data_path, "queries.jsonl")
        summarizer.summarize_patients(
            patients_path=patients_path,
            output_dir=args.output_dir,
            prompt_configs=patient_configs,
            batch_size=args.batch_size,
        )
    else:
        logging.info("No patient prompts specified, skipping patient summarization")

    logging.info("Summarization complete!")


def cli_main():
    """Entry point for the console script"""
    main()


if __name__ == "__main__":
    main()