# src/trialmesh/utils/prompt_config.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptConfig:
    """Configuration for a single prompt in the summarization pipeline.

    Attributes:
        name: Name of the prompt file (without .txt extension)
        max_tokens: Maximum number of tokens to generate for this prompt
        output_suffix: Suffix to append to output filename
        temperature: Optional temperature parameter for generation
    """
    name: str
    max_tokens: int
    output_suffix: Optional[str] = None
    temperature: float = 0.0

    def __post_init__(self):
        """Set default output suffix if not provided."""
        if self.output_suffix is None:
            # Extract suffix from name (e.g., patient_summary_sigir2016 -> summary)
            parts = self.name.split('_')
            if len(parts) >= 2:
                # Take the word after the entity type (patient/trial)
                if parts[0] in ('patient', 'trial'):
                    self.output_suffix = parts[1]
                else:
                    self.output_suffix = parts[0]
            else:
                self.output_suffix = self.name