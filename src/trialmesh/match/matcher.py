# src/trialmesh/match/matcher.py

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union, NamedTuple
from tqdm import tqdm
from dataclasses import dataclass

from trialmesh.llm.llama_runner import LlamaRunner
from trialmesh.llm.prompt_runner import PromptRunner
from trialmesh.utils.prompt_registry import PromptRegistry


@dataclass
class PatientTrialPair:
    """Container for a patient-trial pair with filtering results.

    This class tracks a single patient-trial combination through
    the entire filtering pipeline, maintaining all intermediate
    results and metadata.
    """
    patient_id: str
    trial_id: str
    patient_summary: str
    trial_data: Dict[str, Any]
    vector_score: float = 0.0
    exclusion_result: Optional[Dict[str, Any]] = None
    inclusion_result: Optional[Dict[str, Any]] = None
    scoring_result: Optional[Dict[str, Any]] = None

    def to_evaluation_dict(self) -> Dict[str, Any]:
        """Convert to evaluation dictionary for final output."""
        return {
            "trial_id": self.trial_id,
            "trial_title": self.trial_data.get("title", ""),
            "vector_score": self.vector_score,
            "exclusion_result": self.exclusion_result or {},
            "inclusion_result": self.inclusion_result or {},
            "scoring_result": self.scoring_result or {}
        }


class TrialMatcher:
    """Pipeline for matching patients to trials using LLM evaluation.

    This class implements a multi-stage filtering pipeline to match patients
    to appropriate clinical trials. The matching process includes:

    1. Vector-based retrieval of candidate trials
    2. Exclusion criteria filtering to eliminate obvious mismatches
    3. Inclusion criteria analysis to verify potential matches
    4. Detailed clinical reasoning and scoring for final ranking

    The pipeline is optimized for cross-patient batching to maximize
    vLLM efficiency by processing massive batches across all patients
    rather than small batches per patient.

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
            batch_size: Batch size for processing (used for massive cross-patient batches)
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

    def _collect_all_patient_trial_pairs(self, search_results: List[Dict[str, Any]],
                                         top_k: Optional[int] = None) -> List[PatientTrialPair]:
        """Collect all patient-trial pairs from search results.

        This method creates PatientTrialPair objects for every combination
        of patients and their candidate trials, setting up the data structure
        for cross-patient batch processing.

        Args:
            search_results: List of search result dictionaries from vector retrieval
            top_k: Maximum number of trials to evaluate per patient

        Returns:
            List of PatientTrialPair objects ready for batch processing
        """
        logging.info("Collecting all patient-trial pairs for batch processing")

        all_pairs = []
        total_pairs = 0

        for patient_result in search_results:
            patient_id = patient_result["query_id"]
            patient_data = self.patients.get(patient_id)

            if not patient_data:
                logging.warning(f"Patient {patient_id} not found in summaries, skipping")
                continue

            patient_summary = patient_data.get("summary", "")

            # Get trial results for this patient
            if top_k is not None:
                trial_results = patient_result.get("results", [])[:top_k]
            else:
                trial_results = patient_result.get("results", [])

            # Create pairs for valid trials
            for trial_result in trial_results:
                trial_id = trial_result["doc_id"]
                vector_score = trial_result.get("score", 0.0)
                trial_data = self.trials.get(trial_id)

                if trial_data:
                    pair = PatientTrialPair(
                        patient_id=patient_id,
                        trial_id=trial_id,
                        patient_summary=patient_summary,
                        trial_data=trial_data,
                        vector_score=vector_score
                    )
                    all_pairs.append(pair)
                    total_pairs += 1
                else:
                    logging.warning(f"Trial {trial_id} not found in corpus, skipping")

        logging.info(f"Collected {total_pairs} patient-trial pairs for processing")
        return all_pairs

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
        """Run the complete matching pipeline with cross-patient batching.

        This method processes all patient-trial pairs through massive
        cross-patient batches to maximize vLLM efficiency. Instead of
        processing patients sequentially, it processes all pairs globally
        in three massive batches.

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
        logging.info("Starting cross-patient batch trial matching process")
        logging.info(f"Using exclusion prompt: {exclusion_prompt}")
        logging.info(f"Using inclusion prompt: {inclusion_prompt}")
        logging.info(f"Using scoring prompt: {scoring_prompt}")

        # Step 1: Collect all patient-trial pairs
        all_pairs = self._collect_all_patient_trial_pairs(search_results, top_k)

        if not all_pairs:
            logging.warning("No valid patient-trial pairs found")
            return []

        # Step 2: Apply exclusion filter globally
        if not skip_exclusion:
            all_pairs = self._apply_exclusion_filter_global(
                all_pairs,
                prompt_name=exclusion_prompt,
                max_tokens=exclusion_max_tokens,
                temperature=exclusion_temperature
            )
        else:
            # Mark all as passed if skipping exclusion
            for pair in all_pairs:
                pair.exclusion_result = {"verdict": "PASS", "reason": "Exclusion filter skipped"}

        # Step 3: Apply inclusion filter globally (only to pairs that passed exclusion)
        if not skip_inclusion:
            all_pairs = self._apply_inclusion_filter_global(
                all_pairs,
                prompt_name=inclusion_prompt,
                max_tokens=inclusion_max_tokens,
                temperature=inclusion_temperature,
                include_all_trials=include_all_trials
            )
        else:
            # Mark all as undetermined if skipping inclusion
            for pair in all_pairs:
                if pair.exclusion_result and pair.exclusion_result.get("verdict") != "EXCLUDE":
                    pair.inclusion_result = {
                        "verdict": "UNDETERMINED",
                        "missing_information": "None",
                        "unmet_criteria": "None",
                        "reasoning": "Inclusion filter skipped"
                    }

        # Step 4: Apply scoring globally (only to pairs that didn't fail inclusion)
        if not skip_scoring:
            all_pairs = self._apply_scoring_global(
                all_pairs,
                prompt_name=scoring_prompt,
                max_tokens=scoring_max_tokens,
                temperature=scoring_temperature,
                include_all_trials=include_all_trials
            )
        else:
            # Mark all with default score if skipping scoring
            for pair in all_pairs:
                if (pair.exclusion_result and pair.exclusion_result.get("verdict") != "EXCLUDE" and
                        pair.inclusion_result and pair.inclusion_result.get("verdict") != "FAIL"):
                    pair.scoring_result = {
                        "score": "5",
                        "verdict": "POSSIBLE MATCH",
                        "reasoning": "Scoring skipped"
                    }

        # Step 5: Reconstruct patient-centric results
        patient_results = self._reconstruct_patient_results(all_pairs, include_all_trials)

        logging.info(f"Completed matching for {len(patient_results)} patients")
        return patient_results

    def _apply_exclusion_filter_global(self, all_pairs: List[PatientTrialPair],
                                       prompt_name: str = "exclusion_filter_sigir2016",
                                       max_tokens: Optional[int] = None,
                                       temperature: Optional[float] = None) -> List[PatientTrialPair]:
        """Apply exclusion filter globally across all patient-trial pairs.

        This method processes all patient-trial pairs in massive batches,
        maximizing vLLM efficiency by batching across patients rather than
        within patients.

        Args:
            all_pairs: List of all PatientTrialPair objects
            prompt_name: Name of the prompt to use for exclusion filtering
            max_tokens: Maximum tokens to generate in response
            temperature: Temperature for generation

        Returns:
            List of PatientTrialPair objects with exclusion results filled in
        """
        logging.info(f"Running global exclusion filter on {len(all_pairs)} pairs with prompt {prompt_name}")

        # Check if prompt exists
        if not self.prompt_registry.get(prompt_name).get("user"):
            logging.error(f"Exclusion prompt {prompt_name} not found")
            raise ValueError(f"Exclusion prompt {prompt_name} not found in registry")

        # Pre-filter pairs with empty exclusion criteria
        pairs_with_criteria = []
        auto_passed_pairs = []

        for pair in all_pairs:
            exclusion_criteria = pair.trial_data.get("metadata", {}).get("exclusion_criteria", "")
            if not exclusion_criteria.strip():
                # Auto-pass trials with no exclusion criteria
                pair.exclusion_result = {"verdict": "PASS", "reason": "No exclusion criteria specified"}
                auto_passed_pairs.append(pair)
            else:
                pairs_with_criteria.append(pair)

        logging.info(f"Auto-passed {len(auto_passed_pairs)} pairs with no exclusion criteria")

        # If no pairs have exclusion criteria, return early
        if not pairs_with_criteria:
            return all_pairs

        # Prepare massive batches for cross-patient processing
        # Use larger batch sizes since we're batching across patients
        batch_size = self.batch_size
        batches = [pairs_with_criteria[i:i + batch_size]
                   for i in range(0, len(pairs_with_criteria), batch_size)]

        logging.info(
            f"Processing {len(pairs_with_criteria)} pairs in {len(batches)} batches of size ~{batch_size}")

        # Process each batch
        for batch_idx, batch in enumerate(tqdm(batches, desc="Global exclusion filtering")):
            variables_list = []

            for pair in batch:
                # Extract exclusion criteria
                exclusion_criteria = pair.trial_data.get("metadata", {}).get("exclusion_criteria", "")

                variables = {
                    "patient_summary": pair.patient_summary,
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
            for pair, response in zip(batch, responses):
                if response is None:
                    logging.warning(f"No response for pair {pair.patient_id}-{pair.trial_id}, marking as PASS")
                    pair.exclusion_result = {"verdict": "PASS", "reason": "No LLM response received"}
                    continue

                # Extract verdict and reason using regex
                verdict_match = re.search(r"VERDICT:\s*(\w+)", response.text)
                reason_match = re.search(r"REASON:\s*(.*?)(?=\n\n|\Z)", response.text, re.DOTALL)

                verdict = verdict_match.group(1) if verdict_match else "UNPARSABLE_VERDICT"
                reason = reason_match.group(1).strip() if reason_match else "Unparsable reasoning from model output"

                # Log unparsable responses for review
                if verdict == "UNPARSABLE_VERDICT":
                    logging.warning(
                        f"Could not parse verdict for pair {pair.patient_id}-{pair.trial_id} in exclusion filter")
                    logging.debug(f"Response fragment: {response.text[:200]}...")
                    # Treat unparsable as PASS to be conservative
                    verdict = "PASS"
                    reason = f"Unparsable response (treated as PASS): {reason}"

                # Store exclusion result
                pair.exclusion_result = {"verdict": verdict, "reason": reason}

        # Count results
        excluded_count = sum(1 for pair in all_pairs
                             if pair.exclusion_result and pair.exclusion_result.get("verdict") == "EXCLUDE")
        passed_count = len(all_pairs) - excluded_count

        logging.info(
            f"Global exclusion filter: {passed_count} passed, {excluded_count} excluded of {len(all_pairs)} total pairs")

        return all_pairs

    def _apply_inclusion_filter_global(self, all_pairs: List[PatientTrialPair],
                                       prompt_name: str = "inclusion_filter_sigir2016",
                                       max_tokens: Optional[int] = None,
                                       temperature: Optional[float] = None,
                                       include_all_trials: bool = False) -> List[PatientTrialPair]:
        """Apply inclusion filter globally to pairs that passed exclusion.

        This method evaluates whether patients meet inclusion criteria
        for trials in massive cross-patient batches.

        Args:
            all_pairs: List of all PatientTrialPair objects
            prompt_name: Name of the prompt to use for inclusion filtering
            max_tokens: Maximum tokens to generate in response
            temperature: Temperature for generation
            include_all_trials: Whether to process all trials or only those that passed exclusion

        Returns:
            List of PatientTrialPair objects with inclusion results filled in
        """
        # Filter to only pairs that passed exclusion (unless including all)
        if include_all_trials:
            eligible_pairs = [pair for pair in all_pairs if pair.exclusion_result]
        else:
            eligible_pairs = [pair for pair in all_pairs
                              if pair.exclusion_result and pair.exclusion_result.get("verdict") != "EXCLUDE"]

        logging.info(
            f"Running global inclusion filter on {len(eligible_pairs)} eligible pairs with prompt {prompt_name}")

        # Check if prompt exists
        if not self.prompt_registry.get(prompt_name).get("user"):
            logging.error(f"Inclusion prompt {prompt_name} not found")
            raise ValueError(f"Inclusion prompt {prompt_name} not found in registry")

        # Pre-filter pairs with empty inclusion criteria
        pairs_with_criteria = []
        auto_undetermined_pairs = []

        for pair in eligible_pairs:
            inclusion_criteria = pair.trial_data.get("metadata", {}).get("inclusion_criteria", "")

            if not inclusion_criteria.strip():
                # Auto-mark trials with no inclusion criteria as UNDETERMINED
                pair.inclusion_result = {
                    "verdict": "UNDETERMINED",
                    "missing_information": "N/A",
                    "unmet_criteria": "N/A",
                    "reasoning": "No inclusion criteria specified"
                }
                auto_undetermined_pairs.append(pair)
            else:
                pairs_with_criteria.append(pair)

        logging.info(f"Auto-undetermined {len(auto_undetermined_pairs)} pairs with no inclusion criteria")

        # If no pairs have inclusion criteria, return early
        if not pairs_with_criteria:
            return all_pairs

        # Prepare massive batches for cross-patient processing
        batch_size = self.batch_size
        batches = [pairs_with_criteria[i:i + batch_size]
                   for i in range(0, len(pairs_with_criteria), batch_size)]

        logging.info(
            f"Processing {len(pairs_with_criteria)} pairs in {len(batches)} batches of size ~{batch_size}")

        # Process each batch
        for batch_idx, batch in enumerate(tqdm(batches, desc="Global inclusion filtering")):
            variables_list = []

            for pair in batch:
                # Extract inclusion criteria
                inclusion_criteria = pair.trial_data.get("metadata", {}).get("inclusion_criteria", "")

                variables = {
                    "patient_summary": pair.patient_summary,
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
            for pair, response in zip(batch, responses):
                if response is None:
                    logging.warning(f"No response for pair {pair.patient_id}-{pair.trial_id}, marking as UNDETERMINED")
                    pair.inclusion_result = {
                        "verdict": "UNDETERMINED",
                        "missing_information": "No LLM response received",
                        "unmet_criteria": "Unknown",
                        "reasoning": "No LLM response received"
                    }
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
                    logging.warning(
                        f"Could not parse verdict for pair {pair.patient_id}-{pair.trial_id} in inclusion filter")
                    logging.debug(f"Response fragment: {response.text[:200]}...")
                    # Treat unparsable as UNDETERMINED to be conservative
                    verdict = "UNDETERMINED"
                    reasoning = f"Unparsable response (treated as UNDETERMINED): {reasoning}"

                # Store inclusion result
                pair.inclusion_result = {
                    "verdict": verdict,
                    "missing_information": missing,
                    "unmet_criteria": unmet,
                    "reasoning": reasoning
                }

        # Count results
        failed_count = sum(1 for pair in eligible_pairs
                           if pair.inclusion_result and pair.inclusion_result.get("verdict") == "FAIL")
        passed_count = len(eligible_pairs) - failed_count

        logging.info(
            f"Global inclusion filter: {passed_count} passed/undetermined, {failed_count} failed of {len(eligible_pairs)} eligible pairs")

        return all_pairs

    def _apply_scoring_global(self, all_pairs: List[PatientTrialPair],
                              prompt_name: str = "final_match_scoring_sigir2016",
                              max_tokens: Optional[int] = None,
                              temperature: Optional[float] = None,
                              include_all_trials: bool = False) -> List[PatientTrialPair]:
        """Apply final scoring globally to pairs that passed both filters.

        This method performs detailed clinical evaluation in massive
        cross-patient batches for maximum efficiency.

        Args:
            all_pairs: List of all PatientTrialPair objects
            prompt_name: Name of the prompt to use for final scoring
            max_tokens: Maximum tokens to generate in response
            temperature: Temperature for generation
            include_all_trials: Whether to process all trials or only those that passed filters

        Returns:
            List of PatientTrialPair objects with scoring results filled in
        """
        # Filter to only pairs that should be scored
        if include_all_trials:
            eligible_pairs = [pair for pair in all_pairs
                              if (pair.exclusion_result and pair.inclusion_result)]
        else:
            eligible_pairs = [pair for pair in all_pairs
                              if (pair.exclusion_result and pair.exclusion_result.get("verdict") != "EXCLUDE" and
                                  pair.inclusion_result and pair.inclusion_result.get("verdict") != "FAIL")]

        logging.info(f"Running global scoring on {len(eligible_pairs)} eligible pairs with prompt {prompt_name}")

        # Check if prompt exists
        if not self.prompt_registry.get(prompt_name).get("user"):
            logging.error(f"Scoring prompt {prompt_name} not found")
            raise ValueError(f"Scoring prompt {prompt_name} not found in registry")

        if not eligible_pairs:
            logging.info("No eligible pairs for scoring")
            return all_pairs

        # Prepare massive batches for cross-patient processing
        batch_size = self.batch_size
        batches = [eligible_pairs[i:i + batch_size]
                   for i in range(0, len(eligible_pairs), batch_size)]

        logging.info(
            f"Processing {len(eligible_pairs)} pairs in {len(batches)} batches of size ~{batch_size}")

        # Process each batch
        for batch_idx, batch in enumerate(tqdm(batches, desc="Global scoring")):
            variables_list = []

            for pair in batch:
                # Format trial text
                trial_summary = self._format_trial(pair.trial_data)

                # Extract previous filter results
                exclusion_result = pair.exclusion_result or {}
                inclusion_result = pair.inclusion_result or {}

                exclusion_verdict = exclusion_result.get("verdict", "UNKNOWN")
                exclusion_reason = exclusion_result.get("reason", "No reason provided")

                inclusion_verdict = inclusion_result.get("verdict", "UNKNOWN")
                inclusion_reasoning = inclusion_result.get("reasoning", "No reasoning provided")
                missing_information = inclusion_result.get("missing_information", "None")
                unmet_criteria = inclusion_result.get("unmet_criteria", "None")

                variables = {
                    "patient_summary": pair.patient_summary,
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
            for pair, response in zip(batch, responses):
                if response is None:
                    logging.warning(f"No response for pair {pair.patient_id}-{pair.trial_id}, assigning default score")
                    pair.scoring_result = {
                        "score": "5",
                        "verdict": "POSSIBLE MATCH",
                        "reasoning": "No LLM response received, assigned default score"
                    }
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
                    logging.warning(f"Pair {pair.patient_id}-{pair.trial_id} had unparsable score, assigning default")
                    score = "5"  # Default middle score
                    reasoning = f"Unparsable score (assigned default 5): {reasoning}"

                if verdict == "UNPARSABLE_VERDICT":
                    logging.warning(f"Pair {pair.patient_id}-{pair.trial_id} had unparsable verdict")
                    verdict = "POSSIBLE MATCH"
                    reasoning = f"Unparsable verdict (treated as POSSIBLE MATCH): {reasoning}"

                # Store scoring result
                pair.scoring_result = {
                    "score": score,
                    "verdict": verdict,
                    "reasoning": reasoning
                }

        scored_count = sum(1 for pair in eligible_pairs if pair.scoring_result)
        logging.info(f"Global scoring completed for {scored_count} of {len(eligible_pairs)} eligible pairs")

        return all_pairs

    def _reconstruct_patient_results(self, all_pairs: List[PatientTrialPair],
                                     include_all_trials: bool = False) -> List[Dict[str, Any]]:
        """Reconstruct patient-centric results from processed pairs.

        This method groups the processed PatientTrialPair objects back
        into the expected patient-centric output format.

        Args:
            all_pairs: List of processed PatientTrialPair objects
            include_all_trials: Whether to include all trials regardless of filtering status

        Returns:
            List of patient match results in the expected format
        """
        logging.info("Reconstructing patient-centric results from processed pairs")

        # Group pairs by patient
        patient_pairs = {}
        for pair in all_pairs:
            if pair.patient_id not in patient_pairs:
                patient_pairs[pair.patient_id] = []
            patient_pairs[pair.patient_id].append(pair)

        # Build patient results
        patient_results = []

        for patient_id, pairs in patient_pairs.items():
            if not pairs:
                continue

            # Get patient summary from the first pair
            patient_summary = pairs[0].patient_summary

            # Prepare patient result structure
            patient_match_result = {
                "patient_id": patient_id,
                "patient_summary": patient_summary,
                "trial_evaluations": []
            }

            # Filter pairs based on include_all_trials setting
            if include_all_trials:
                # Include all trials that have been processed
                relevant_pairs = [pair for pair in pairs
                                  if pair.exclusion_result is not None]
            else:
                # Include only trials that made it through all filters
                relevant_pairs = [pair for pair in pairs
                                  if (pair.exclusion_result and pair.exclusion_result.get("verdict") != "EXCLUDE" and
                                      pair.inclusion_result and pair.inclusion_result.get("verdict") != "FAIL" and
                                      pair.scoring_result is not None)]

            # Convert pairs to evaluation dictionaries
            for pair in relevant_pairs:
                evaluation = pair.to_evaluation_dict()
                patient_match_result["trial_evaluations"].append(evaluation)

            # Sort by score if available, otherwise by vector score
            def sort_key(eval_dict):
                score_result = eval_dict.get("scoring_result", {})
                if score_result and score_result.get("score", "").isdigit():
                    return -int(score_result["score"])  # Negative for descending
                return -eval_dict.get("vector_score", 0.0)  # Fallback to vector score

            patient_match_result["trial_evaluations"].sort(key=sort_key)

            patient_results.append(patient_match_result)

        # Sort patient results by patient_id for consistency
        patient_results.sort(key=lambda x: x["patient_id"])

        # Log summary statistics
        total_evaluations = sum(len(pr["trial_evaluations"]) for pr in patient_results)
        avg_evaluations = total_evaluations / len(patient_results) if patient_results else 0

        logging.info(f"Reconstructed results for {len(patient_results)} patients")
        logging.info(f"Total trial evaluations: {total_evaluations} (avg: {avg_evaluations:.1f} per patient)")

        return patient_results