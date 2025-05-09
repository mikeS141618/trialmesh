# src/trialmesh/match/matcher.py

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from tqdm import tqdm

from trialmesh.llm.llama_runner import LlamaRunner
from trialmesh.llm.prompt_runner import PromptRunner
from trialmesh.utils.prompt_registry import PromptRegistry


class TrialMatcher:
    """Pipeline for matching patients to trials using LLM evaluation.

    This class implements a multi-stage filtering pipeline to match patients
    to appropriate clinical trials. The matching process includes:

    1. Vector-based retrieval of candidate trials
    2. Exclusion criteria filtering to eliminate obvious mismatches
    3. Inclusion criteria analysis to verify potential matches
    4. Detailed clinical reasoning and scoring for final ranking

    Each stage leverages LLMs to apply clinical judgment similar to how
    trial coordinators would evaluate potential candidates.

    Attributes:
        patient_summaries_path (str): Path to patient summaries file
        trials_path (str): Path to trial corpus file
        batch_size (int): Batch size for processing
        llm (LlamaRunner): LlamaRunner instance for generating text
        prompt_registry (PromptRegistry): Registry with available prompts
        prompt_runner (PromptRunner): Interface for running prompts
        patients (dict): Dictionary of loaded patient data
        trials (dict): Dictionary of loaded trial data
    """

    def __init__(
            self,
            llm: LlamaRunner,
            patient_summaries_path: str,
            trials_path: str,
            batch_size: int = 8,
            prompt_dir: str = "./prompts",
    ):
        """Initialize the trial matcher.

        Args:
            llm: LlamaRunner instance
            patient_summaries_path: Full path to patient summaries file
            trials_path: Full path to trial corpus file
            batch_size: Batch size for processing
            prompt_dir: Directory containing prompt text files
        """
        self.patient_summaries_path = patient_summaries_path
        self.trials_path = trials_path
        self.batch_size = batch_size

        # Initialize LLM components with file-based prompts
        self.llm = llm
        self.prompt_registry = PromptRegistry(prompt_dir)
        self.prompt_runner = PromptRunner(llm, self.prompt_registry)

        # Load data
        self.patients = self._load_patients()
        self.trials = self._load_trials()

        logging.info(f"Loaded {len(self.patients)} patients and {len(self.trials)} trials")

    def _load_patients(self) -> Dict[str, Dict[str, Any]]:
        """Load patient summaries."""
        patients = {}
        with open(self.patient_summaries_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    patient = json.loads(line)
                    patient_id = patient.get("_id")
                    if patient_id:
                        patients[patient_id] = patient
                except json.JSONDecodeError:
                    logging.warning(f"Error parsing patient line: {line[:100]}...")

        return patients

    def _load_trials(self) -> Dict[str, Dict[str, Any]]:
        """Load trial data."""
        trials = {}
        with open(self.trials_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    trial = json.loads(line)
                    trial_id = trial.get("_id")
                    if trial_id:
                        trials[trial_id] = trial
                except json.JSONDecodeError:
                    logging.warning(f"Error parsing trial line: {line[:100]}...")

        return trials

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

    def match(self, search_results: List[Dict[str, Any]],
              exclusion_prompt: str = "exclusion_filter_sigir2016",
              inclusion_prompt: str = "inclusion_filter_sigir2016",
              scoring_prompt: str = "final_match_scoring_sigir2016",
              exclusion_max_tokens: Optional[int] = None,
              inclusion_max_tokens: Optional[int] = None,
              scoring_max_tokens: Optional[int] = None,
              exclusion_temperature: Optional[float] = None,
              inclusion_temperature: Optional[float] = None,
              scoring_temperature: Optional[float] = None,
              top_k: Optional[int] = None,
              skip_exclusion: bool = False,
              skip_inclusion: bool = False,
              skip_scoring: bool = False,
              include_all_trials: bool = False) -> List[Dict[str, Any]]:
        """Run the complete matching pipeline.

        This method processes the search results through the multi-stage
        filtering pipeline to generate detailed trial matching assessments
        for each patient.

        Args:
            search_results: List of search result dictionaries from vector retrieval
            exclusion_prompt: Name of the prompt for exclusion filtering
            inclusion_prompt: Name of the prompt for inclusion filtering
            scoring_prompt: Name of the prompt for final scoring
            exclusion_max_tokens: Maximum tokens for exclusion prompt responses
            inclusion_max_tokens: Maximum tokens for inclusion prompt responses
            scoring_max_tokens: Maximum tokens for scoring prompt responses
            exclusion_temperature: Temperature for exclusion LLM generation
            inclusion_temperature: Temperature for inclusion LLM generation
            scoring_temperature: Temperature for scoring LLM generation
            top_k: Maximum number of trials to evaluate per patient
            skip_exclusion: Whether to skip the exclusion filtering stage
            skip_inclusion: Whether to skip the inclusion filtering stage
            skip_scoring: Whether to skip the detailed scoring stage
            include_all_trials: Whether to include all trials in output regardless of filtering status

        Returns:
            List of patient match results with detailed trial evaluations
        """
        logging.info("Starting trial matching process")
        logging.info(f"Using exclusion prompt: {exclusion_prompt}")
        logging.info(f"Using inclusion prompt: {inclusion_prompt}")
        logging.info(f"Using scoring prompt: {scoring_prompt}")

        # Process patients in order
        all_patient_results = []

        for patient_result in tqdm(search_results, desc="Processing patients"):
            patient_id = patient_result["query_id"]
            patient_data = self.patients.get(patient_id)

            if not patient_data:
                logging.warning(f"Patient {patient_id} not found in summaries, skipping")
                continue

            patient_summary = patient_data.get("summary", "")

            # Get trial results for this patient
            if top_k is not None:
                trial_results = patient_result.get("results", [])[:top_k]
                logging.info(f"Using top {top_k} trials for patient {patient_id}")
            else:
                trial_results = patient_result.get("results", [])
                logging.info(f"Using all {len(trial_results)} trials for patient {patient_id}")

            trial_ids = [tr["doc_id"] for tr in trial_results]

            # Prepare patient result structure
            patient_match_result = {
                "patient_id": patient_id,
                "patient_summary": patient_summary,
                "trial_evaluations": []
            }

            # Get trial data
            valid_trials = []
            for trial_id in trial_ids:
                trial_data = self.trials.get(trial_id)
                if trial_data:
                    valid_trials.append((trial_id, trial_data))
                else:
                    logging.warning(f"Trial {trial_id} not found in corpus, skipping")

            # Dictionary to track all trials for complete results option
            all_trial_results = {}

            # 1. Exclusion Filter
            if not skip_exclusion:
                filtered_trials, excluded_trials = self._apply_exclusion_filter(
                    patient_summary,
                    valid_trials,
                    prompt_name=exclusion_prompt,
                    max_tokens=exclusion_max_tokens,
                    temperature=exclusion_temperature,
                    return_excluded=include_all_trials
                )

                # Track all trials if needed
                if include_all_trials:
                    for trial_result in filtered_trials + excluded_trials:
                        all_trial_results[trial_result["trial_id"]] = trial_result
            else:
                # Skip exclusion filter - pass all trials through with PASS verdict
                filtered_trials = []
                for trial_id, trial_data in valid_trials:
                    trial_result = {
                        "trial_id": trial_id,
                        "trial_data": trial_data,
                        "exclusion_result": {"verdict": "PASS", "reason": "Exclusion filter skipped"}
                    }
                    filtered_trials.append(trial_result)
                    if include_all_trials:
                        all_trial_results[trial_id] = trial_result

            # 2. Inclusion Filter (for trials that passed exclusion)
            if not skip_inclusion:
                inclusion_results, failed_inclusion = self._apply_inclusion_filter(
                    patient_summary,
                    filtered_trials,
                    prompt_name=inclusion_prompt,
                    max_tokens=inclusion_max_tokens,
                    temperature=inclusion_temperature,
                    return_failed=include_all_trials
                )

                # Track inclusion failures if needed
                if include_all_trials:
                    for trial_result in failed_inclusion:
                        all_trial_results[trial_result["trial_id"]] = trial_result
            else:
                # Skip inclusion filter - pass all trials through with UNDETERMINED verdict
                inclusion_results = []
                for trial_result in filtered_trials:
                    trial_result["inclusion_result"] = {
                        "verdict": "UNDETERMINED",
                        "missing_information": "None",
                        "unmet_criteria": "None",
                        "reasoning": "Inclusion filter skipped"
                    }
                    inclusion_results.append(trial_result)
                    if include_all_trials:
                        all_trial_results[trial_result["trial_id"]] = trial_result

            # 3. Final Scoring (for trials that didn't fail inclusion)
            if not skip_scoring:
                final_results = self._apply_scoring(
                    patient_summary,
                    inclusion_results,
                    prompt_name=scoring_prompt,
                    max_tokens=scoring_max_tokens,
                    temperature=scoring_temperature
                )

                # Update tracked results with scores
                if include_all_trials:
                    for trial_result in final_results:
                        all_trial_results[trial_result["trial_id"]] = trial_result
            else:
                # Skip scoring - keep all trials with default score
                final_results = []
                for trial_result in inclusion_results:
                    trial_result["scoring_result"] = {
                        "score": "5",
                        "verdict": "POSSIBLE MATCH",
                        "reasoning": "Scoring skipped"
                    }
                    final_results.append(trial_result)
                    if include_all_trials:
                        all_trial_results[trial_result["trial_id"]] = trial_result

            # Add trial evaluations to patient result
            if include_all_trials:
                # Include all trials we've tracked
                for trial_id, trial_result in all_trial_results.items():
                    trial_data = trial_result["trial_data"]

                    evaluation = {
                        "trial_id": trial_id,
                        "trial_title": trial_data.get("title", ""),
                        "exclusion_result": trial_result.get("exclusion_result", {}),
                        "inclusion_result": trial_result.get("inclusion_result", {}),
                        "scoring_result": trial_result.get("scoring_result", {})
                    }

                    patient_match_result["trial_evaluations"].append(evaluation)
            else:
                # Include only trials that made it through all filters
                for trial_result in final_results:
                    trial_id = trial_result["trial_id"]
                    trial_data = trial_result["trial_data"]

                    evaluation = {
                        "trial_id": trial_id,
                        "trial_title": trial_data.get("title", ""),
                        "exclusion_result": trial_result.get("exclusion_result", {}),
                        "inclusion_result": trial_result.get("inclusion_result", {}),
                        "scoring_result": trial_result.get("scoring_result", {})
                    }

                    patient_match_result["trial_evaluations"].append(evaluation)

            # Add patient result to all results
            all_patient_results.append(patient_match_result)

        logging.info(f"Completed matching for {len(all_patient_results)} patients")
        return all_patient_results

    def _apply_exclusion_filter(self, patient_summary: str, trials: List[Tuple[str, Dict[str, Any]]],
                                prompt_name: str = "exclusion_filter_sigir2016",
                                max_tokens: Optional[int] = None,
                                temperature: Optional[float] = None,
                                return_excluded: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Apply exclusion filter to trials.

        This method evaluates whether a patient should be excluded from trials
        based on exclusion criteria. It uses an LLM to analyze the patient summary
        against each trial's exclusion criteria to identify clear mismatches.

        The method automatically passes trials without exclusion criteria and
        processes the remaining trials in batches for efficiency.

        Args:
            patient_summary: Patient summary text
            trials: List of (trial_id, trial_data) tuples
            prompt_name: Name of the prompt to use for exclusion filtering
            max_tokens: Maximum tokens to generate in response
            temperature: Temperature for generation
            return_excluded: Whether to return excluded trials

        Returns:
            Tuple of (passed_trials, excluded_trials), where excluded_trials
            is empty if return_excluded is False
        """
        logging.info(f"Running exclusion filter on {len(trials)} trials with prompt {prompt_name}")

        # Check if prompt exists
        if not self.prompt_registry.get(prompt_name).get("user"):
            logging.error(f"Exclusion prompt {prompt_name} not found")
            raise ValueError(f"Exclusion prompt {prompt_name} not found in registry")

        # Pre-filter trials with empty exclusion criteria
        trials_with_criteria = []
        auto_passed_trials = []

        for trial_id, trial_data in trials:
            exclusion_criteria = trial_data.get("metadata", {}).get("exclusion_criteria", "")
            if not exclusion_criteria.strip():
                # Auto-pass trials with no exclusion criteria
                trial_result = {
                    "trial_id": trial_id,
                    "trial_data": trial_data,
                    "exclusion_result": {"verdict": "PASS", "reason": "No exclusion criteria specified"}
                }
                auto_passed_trials.append(trial_result)
            else:
                trials_with_criteria.append((trial_id, trial_data))

        logging.info(f"Auto-passed {len(auto_passed_trials)} trials with no exclusion criteria")

        # If no trials have exclusion criteria, return early
        if not trials_with_criteria:
            # Always return a tuple, with empty list as second value if not return_excluded
            return auto_passed_trials, []

        # Prepare batches for trials that have exclusion criteria
        batches = [trials_with_criteria[i:i + self.batch_size]
                   for i in range(0, len(trials_with_criteria), self.batch_size)]

        # Process each batch
        passed_trials = auto_passed_trials  # Start with auto-passed trials
        excluded_trials = []

        for batch in tqdm(batches, desc="Exclusion filtering"):
            variables_list = []

            for trial_id, trial_data in batch:
                # Extract exclusion criteria
                exclusion_criteria = trial_data.get("metadata", {}).get("exclusion_criteria", "")

                variables = {
                    "patient_summary": patient_summary,
                    "exclusion_criteria": exclusion_criteria
                }

                variables_list.append(variables)

            # Run LLM for the batch
            responses = self.prompt_runner.run_prompt_batch(
                prompt_name=prompt_name,
                variables_list=variables_list,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Process responses
            for i, ((trial_id, trial_data), response) in enumerate(zip(batch, responses)):
                if response is None:
                    logging.warning(f"No response for trial {trial_id}, skipping")
                    continue

                # Extract verdict and reason using regex
                verdict_match = re.search(r"VERDICT:\s*(\w+)", response.text)
                reason_match = re.search(r"REASON:\s*(.*?)(?=\n\n|\Z)", response.text, re.DOTALL)

                verdict = verdict_match.group(1) if verdict_match else "UNPARSABLE_VERDICT"
                reason = reason_match.group(1).strip() if reason_match else "Unparsable reasoning from model output"

                # Log unparsable responses for review
                if verdict == "UNPARSABLE_VERDICT":
                    logging.warning(f"Could not parse verdict for trial {trial_id} in exclusion filter")
                    logging.debug(f"Response fragment: {response.text[:200]}...")

                # Create trial result with exclusion data
                trial_result = {
                    "trial_id": trial_id,
                    "trial_data": trial_data,
                    "exclusion_result": {"verdict": verdict, "reason": reason}
                }

                # If verdict is not EXCLUDE, add trial to passed list
                if verdict != "EXCLUDE":
                    passed_trials.append(trial_result)
                elif return_excluded:
                    excluded_trials.append(trial_result)

        logging.info(f"Exclusion filter passed {len(passed_trials)} of {len(trials)} trials")

        # Always return a tuple with both values
        return passed_trials, excluded_trials

    def _apply_inclusion_filter(self, patient_summary: str, trials: List[Dict[str, Any]],
                                prompt_name: str = "inclusion_filter_sigir2016",
                                max_tokens: Optional[int] = None,
                                temperature: Optional[float] = None,
                                return_failed: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Apply inclusion filter to trials that passed exclusion.

        This method evaluates whether a patient meets the inclusion criteria
        for each trial. It uses an LLM to analyze the patient summary against
        each trial's inclusion criteria to determine eligibility.

        The method identifies missing information and unmet criteria,
        providing detailed reasoning for each decision.

        Args:
            patient_summary: Patient summary text
            trials: List of dictionaries with trial data and exclusion results
            prompt_name: Name of the prompt to use for inclusion filtering
            max_tokens: Maximum tokens to generate in response
            temperature: Temperature for generation
            return_failed: Whether to return trials that failed inclusion

        Returns:
            Tuple of (included_trials, failed_trials), where failed_trials
            is empty if return_failed is False
        """
        logging.info(f"Running inclusion filter on {len(trials)} trials with prompt {prompt_name}")

        # Check if prompt exists
        if not self.prompt_registry.get(prompt_name).get("user"):
            logging.error(f"Inclusion prompt {prompt_name} not found")
            raise ValueError(f"Inclusion prompt {prompt_name} not found in registry")

        # Pre-filter trials with empty inclusion criteria
        trials_with_criteria = []
        auto_undetermined_trials = []

        for trial_result in trials:
            trial_data = trial_result["trial_data"]
            inclusion_criteria = trial_data.get("metadata", {}).get("inclusion_criteria", "")

            if not inclusion_criteria.strip():
                # Auto-mark trials with no inclusion criteria as UNDETERMINED
                trial_result["inclusion_result"] = {
                    "verdict": "UNDETERMINED",
                    "missing_information": "N/A",
                    "unmet_criteria": "N/A",
                    "reasoning": "No inclusion criteria specified"
                }
                auto_undetermined_trials.append(trial_result)
            else:
                trials_with_criteria.append(trial_result)

        logging.info(f"Auto-undetermined {len(auto_undetermined_trials)} trials with no inclusion criteria")

        # If no trials have inclusion criteria, return early
        if not trials_with_criteria:
            if return_failed:
                return auto_undetermined_trials, []
            return auto_undetermined_trials

        # Prepare batches for trials that have inclusion criteria
        batches = [trials_with_criteria[i:i + self.batch_size]
                   for i in range(0, len(trials_with_criteria), self.batch_size)]

        # Process each batch
        included_trials = auto_undetermined_trials  # Start with auto-undetermined trials
        failed_trials = []

        for batch in tqdm(batches, desc="Inclusion filtering"):
            variables_list = []

            for trial_result in batch:
                trial_data = trial_result["trial_data"]

                # Extract inclusion criteria
                inclusion_criteria = trial_data.get("metadata", {}).get("inclusion_criteria", "")

                variables = {
                    "patient_summary": patient_summary,
                    "inclusion_criteria": inclusion_criteria
                }

                variables_list.append(variables)

            # Run LLM for the batch
            responses = self.prompt_runner.run_prompt_batch(
                prompt_name=prompt_name,
                variables_list=variables_list,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Process responses
            for i, (trial_result, response) in enumerate(zip(batch, responses)):
                if response is None:
                    logging.warning(f"No response for trial {trial_result['trial_id']}, skipping")
                    continue

                # Extract information using improved regex
                verdict_match = re.search(r"VERDICT:\s*(\w+)", response.text)
                missing_match = re.search(r"MISSING INFORMATION:\s*(.*?)(?=\nUNMET CRITERIA:|\nREASONING:|\Z)",
                                          response.text, re.DOTALL)
                unmet_match = re.search(r"UNMET CRITERIA:\s*(.*?)(?=\nREASONING:|\Z)", response.text, re.DOTALL)
                reasoning_match = re.search(r"REASONING:\s*(.*?)(?=\n\n|\Z)", response.text, re.DOTALL)

                verdict = verdict_match.group(1) if verdict_match else "UNPARSABLE_VERDICT"
                missing = missing_match.group(1).strip() if missing_match else "Unparsable missing information"
                unmet = unmet_match.group(1).strip() if unmet_match else "Unparsable unmet criteria"
                reasoning = reasoning_match.group(
                    1).strip() if reasoning_match else "Unparsable reasoning from model output"

                # Handle unparsable verdicts
                if verdict == "UNPARSABLE_VERDICT":
                    logging.warning(f"Could not parse verdict for trial {trial_result['trial_id']} in inclusion filter")
                    logging.debug(f"Response fragment: {response.text[:200]}...")

                # Add inclusion result to trial data
                trial_result["inclusion_result"] = {
                    "verdict": verdict,
                    "missing_information": missing,
                    "unmet_criteria": unmet,
                    "reasoning": reasoning
                }

                # Add to appropriate list based on verdict
                if verdict != "FAIL":
                    included_trials.append(trial_result)
                elif return_failed:
                    failed_trials.append(trial_result)

        logging.info(f"Inclusion filter kept {len(included_trials)} of {len(trials)} trials")

        if return_failed:
            return included_trials, failed_trials
        return included_trials, []  # Return empty list as second value

    def _apply_scoring(self, patient_summary: str, trials: List[Dict[str, Any]],
                       max_tokens: Optional[int] = None,
                       temperature: Optional[float] = None,
                       prompt_name: str = "final_match_scoring_sigir2016") -> List[Dict[str, Any]]:
        """Apply final scoring to trials that passed inclusion filter.

        This method performs detailed clinical evaluation of each trial
        for the given patient, considering:
        1. Results from previous filtering stages
        2. Detailed trial information
        3. Patient characteristics

        It assigns a score (typically 1-10) and detailed verdict with
        clinical reasoning explaining the match quality.

        Args:
            patient_summary: Patient summary text
            trials: List of dictionaries with trial data and previous filter results
            max_tokens: Maximum tokens to generate in response
            temperature: Temperature for generation
            prompt_name: Name of the prompt to use for final scoring

        Returns:
            List of trial dictionaries with added scoring results
        """
        logging.info(f"Running final scoring on {len(trials)} trials with prompt {prompt_name}")

        # Check if prompt exists
        if not self.prompt_registry.get(prompt_name).get("user"):
            logging.error(f"Scoring prompt {prompt_name} not found")
            raise ValueError(f"Scoring prompt {prompt_name} not found in registry")

        # Prepare batches for processing
        batches = [trials[i:i + self.batch_size] for i in range(0, len(trials), self.batch_size)]

        # Process each batch
        scored_trials = []

        for batch in tqdm(batches, desc="Final scoring"):
            variables_list = []

            for trial_result in batch:
                trial_data = trial_result["trial_data"]

                # Format trial text
                trial_summary = self._format_trial(trial_data)

                # Extract previous filter results
                exclusion_result = trial_result.get("exclusion_result", {})
                inclusion_result = trial_result.get("inclusion_result", {})

                exclusion_verdict = exclusion_result.get("verdict", "UNKNOWN")
                exclusion_reason = exclusion_result.get("reason", "No reason provided")

                inclusion_verdict = inclusion_result.get("verdict", "UNKNOWN")
                inclusion_reasoning = inclusion_result.get("reasoning", "No reasoning provided")
                missing_information = inclusion_result.get("missing_information", "None")
                unmet_criteria = inclusion_result.get("unmet_criteria", "None")

                variables = {
                    "patient_summary": patient_summary,
                    "trial_summary": trial_summary,
                    "exclusion_verdict": exclusion_verdict,
                    "exclusion_reason": exclusion_reason,
                    "inclusion_verdict": inclusion_verdict,
                    "inclusion_reasoning": inclusion_reasoning,
                    "missing_information": missing_information,
                    "unmet_criteria": unmet_criteria
                }

                variables_list.append(variables)

            # Run LLM for the batch
            responses = self.prompt_runner.run_prompt_batch(
                prompt_name=prompt_name,
                variables_list=variables_list,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Process responses
            for i, (trial_result, response) in enumerate(zip(batch, responses)):
                if response is None:
                    logging.warning(f"No response for trial {trial_result['trial_id']}, skipping")
                    continue

                # Extract information using improved regex with better section boundaries
                score_match = re.search(r"SCORE:\s*(\d+)", response.text)
                verdict_match = re.search(r"VERDICT:\s*(.*?)(?=\nREASONING:|\n\n|\Z)", response.text)
                reasoning_match = re.search(r"REASONING:\s*(.*?)(?=\n\n|\Z)", response.text, re.DOTALL)

                score = score_match.group(1) if score_match else "UNPARSABLE_SCORE"
                verdict = verdict_match.group(1).strip() if verdict_match else "UNPARSABLE_VERDICT"
                reasoning = reasoning_match.group(
                    1).strip() if reasoning_match else "Unparsable reasoning from model output"

                # Handle unparsable scores/verdicts
                if score == "UNPARSABLE_SCORE":
                    logging.warning(f"Trial {trial_result['trial_id']} had unparsable score")

                if verdict == "UNPARSABLE_VERDICT":
                    reasoning += " (Original verdict was unparsable)"
                    logging.warning(f"Trial {trial_result['trial_id']} had unparsable verdict")

                # Add scoring result to trial data
                trial_result["scoring_result"] = {
                    "score": score,
                    "verdict": verdict,
                    "reasoning": reasoning
                }

                scored_trials.append(trial_result)

        logging.info(f"Completed scoring for {len(scored_trials)} trials")
        return scored_trials