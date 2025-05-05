# src/trialmesh/llm/prompt_runner.py

import logging
from typing import Dict, List, Optional, Union, Tuple

from trialmesh.utils.prompt_registry import PromptRegistry
from trialmesh.llm.llama_runner import LlamaRunner, LlamaResponse


class PromptRunner:
    """High-level interface for running prompts from the registry.

    This class connects the LlamaRunner with the PromptRegistry, providing
    a simplified interface for formatting and running prompts with variables.
    It handles the details of template substitution and LLM execution.

    Attributes:
        llm: LlamaRunner instance for generating text
        prompts: PromptRegistry containing available prompt templates
    """

    def __init__(
            self,
            llama_runner: LlamaRunner,
            prompt_registry: Optional[PromptRegistry] = None,
    ):
        self.llm = llama_runner
        self.prompts = prompt_registry or PromptRegistry()

    def run_prompt(self, prompt_name: str, variables: Dict[str, str],
                   override_system_prompt: Optional[str] = None,
                   max_tokens: Optional[int] = None,
                   temperature: Optional[float] = None) -> LlamaResponse:
        """Run a prompt from the registry with the given variables.

        This method:
        1. Retrieves the prompt template from the registry
        2. Formats the template with provided variables
        3. Runs the LLM with the formatted prompt

        Args:
            prompt_name: Name of the prompt template in the registry
            variables: Dictionary of variables to format into the template
            override_system_prompt: Optional system prompt to use instead of the template's
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            LlamaResponse containing the generated text and metadata

        Raises:
            ValueError: If prompt_name is not found or variables are missing
        """
        prompt_pair = self.prompts.get(prompt_name)
        if not prompt_pair or not prompt_pair.get("user"):
            raise ValueError(f"Prompt '{prompt_name}' not found in registry")

        # Get system and user prompts
        system_prompt = override_system_prompt or prompt_pair.get("system", "")
        user_prompt = prompt_pair.get("user", "")

        # Format the user prompt template with variables
        try:
            formatted_user_prompt = user_prompt.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing variable {e} for prompt '{prompt_name}'")

        # Run the formatted prompt with the system prompt
        return self.llm.generate(
            prompt=formatted_user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def run_prompt_batch(self, prompt_name: str, variables_list: List[Dict[str, str]],
                         override_system_prompt: Optional[str] = None,
                         max_tokens: Optional[int] = None,
                         temperature: Optional[float] = None) -> List[Optional[LlamaResponse]]:
        """Run a prompt from the registry with multiple sets of variables.

        This method efficiently processes multiple prompts with the same template
        by batching the requests to the LLM.

        Args:
            prompt_name: Name of the prompt template in the registry
            variables_list: List of variable dictionaries to format into the template
            override_system_prompt: Optional system prompt to use instead of the template's
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            List of LlamaResponse objects, one for each set of variables

        Raises:
            ValueError: If prompt_name is not found
        """
        prompt_pair = self.prompts.get(prompt_name)
        if not prompt_pair or not prompt_pair.get("user"):
            raise ValueError(f"Prompt '{prompt_name}' not found in registry")

        # Get system and user prompts
        system_prompt = override_system_prompt or prompt_pair.get("system", "")
        user_prompt = prompt_pair.get("user", "")

        # Format each prompt with its variables
        formatted_prompts = []
        for variables in variables_list:
            try:
                formatted_prompt = user_prompt.format(**variables)
                formatted_prompts.append(formatted_prompt)
            except KeyError as e:
                logging.warning(f"Skipping prompt with missing variable {e}")
                formatted_prompts.append(None)

        # Filter out None prompts
        valid_prompts = [p for p in formatted_prompts if p is not None]
        valid_indices = [i for i, p in enumerate(formatted_prompts) if p is not None]

        # Run valid prompts
        if not valid_prompts:
            return []

        results = self.llm.generate_batch(
            prompts=valid_prompts,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Reconstruct full results list with None for invalid prompts
        full_results = [None] * len(variables_list)
        for i, result in zip(valid_indices, results):
            full_results[i] = result

        return full_results