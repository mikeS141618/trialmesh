# src/trialmesh/utils/prompt_registry.py
import os
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple, List


class PromptRegistry:
    """Registry that loads prompts from individual text files.

    This class dynamically loads prompts from text files in a specified directory,
    supporting the "==== SYSTEM PROMPT ====" and "==== USER PROMPT ====" format.

    Attributes:
        prompt_dir (Path): Path to directory containing prompt text files
        prompts (dict): Dictionary of loaded prompts
    """

    def __init__(self, prompt_dir: str = "./prompts"):
        """Initialize the registry with prompts from the specified directory.

        Args:
            prompt_dir: Path to directory containing prompt text files
        """
        self.prompt_dir = Path(prompt_dir)
        self.prompts = {}
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load all prompt files from the prompt directory."""
        if not self.prompt_dir.exists():
            logging.warning(f"Prompt directory '{self.prompt_dir}' does not exist")
            return

        for file_path in self.prompt_dir.glob("*.txt"):
            prompt_name = file_path.stem
            system_prompt, user_prompt = self._parse_prompt_file(file_path)

            if user_prompt:  # Only add if we have at least a user prompt
                self.prompts[prompt_name] = {
                    "system": system_prompt,
                    "user": user_prompt
                }
                logging.debug(f"Loaded prompt '{prompt_name}' from {file_path}")
            else:
                logging.warning(f"Failed to load prompt from {file_path}")

    def _parse_prompt_file(self, file_path: Path) -> Tuple[str, str]:
        """Parse a prompt file into system and user prompts.

        This method reads a prompt file and separates it into system and
        user prompt components based on section markers or fallback rules.

        The file format supports multiple layouts:
        1. Files with both system and user sections marked
        2. Files with only user section marked
        3. Files with no section markers (treated as user prompt)

        Args:
            file_path: Path to the prompt file

        Returns:
            Tuple of (system_prompt, user_prompt) with empty strings for missing parts
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split by section markers
            system_marker = "==== SYSTEM PROMPT ===="
            user_marker = "==== USER PROMPT ===="

            system_prompt = ""
            user_prompt = ""

            if system_marker in content and user_marker in content:
                # Standard format with both system and user prompts
                parts = content.split(system_marker, 1)
                if len(parts) > 1:
                    system_user_parts = parts[1].split(user_marker, 1)
                    if len(system_user_parts) > 1:
                        system_prompt = system_user_parts[0].strip()
                        user_prompt = system_user_parts[1].strip()
            elif user_marker in content:
                # Only user prompt
                parts = content.split(user_marker, 1)
                if len(parts) > 1:
                    user_prompt = parts[1].strip()
            else:
                # Assume the whole file is a user prompt
                user_prompt = content.strip()

            return system_prompt, user_prompt

        except Exception as e:
            logging.error(f"Error parsing prompt file {file_path}: {e}")
            return "", ""

    def get(self, name: str) -> dict:
        """Get prompt pair (system and user) by name.

        Args:
            name: Name of the prompt template to retrieve

        Returns:
            Dictionary containing 'system' and 'user' prompt templates
            Empty strings for missing components
        """
        prompt = self.prompts.get(name)
        if not prompt:
            # Try loading the file directly if not already loaded
            file_path = self.prompt_dir / f"{name}.txt"
            if file_path.exists():
                system_prompt, user_prompt = self._parse_prompt_file(file_path)
                prompt = {"system": system_prompt, "user": user_prompt}
                self.prompts[name] = prompt
                return prompt
            else:
                logging.warning(f"Prompt '{name}' not found in registry or prompt directory")
                return {"system": "", "user": ""}
        return prompt

    def list_available_prompts(self) -> List[str]:
        """List all available prompt names.

        This method returns a comprehensive list of all prompt names available
        in the registry, combining both:
        1. Previously loaded prompts
        2. Prompt files that exist in the directory but haven't been loaded yet

        Returns:
            Sorted list of prompt names available in the registry
        """
        # Combine loaded prompts and files in the directory
        available_prompts = set(self.prompts.keys())

        # Add any files in the directory that haven't been loaded yet
        if self.prompt_dir.exists():
            file_prompts = {file_path.stem for file_path in self.prompt_dir.glob("*.txt")}
            available_prompts.update(file_prompts)

        return sorted(list(available_prompts))