# src/trialmesh/utils/prompt_config.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptConfig:
    """Configuration for a single prompt in the summarization pipeline.

    This class defines the parameters for running a specific prompt,
    including its name, token limits, and temperature settings.

    Attributes:
        name (str): Name of the prompt file (without .txt extension)
        max_tokens (int): Maximum number of tokens to generate for this prompt
        output_suffix (Optional[str]): Suffix to append to output filename
        temperature (float): Temperature parameter for generation (default: 0.0)
    """
    name: str
    max_tokens: int
    output_suffix: Optional[str] = None
    temperature: float = 0.0

    def __post_init__(self):
        """Set default output suffix if not provided.

        This method automatically derives a default output suffix from the
        prompt name if one isn't explicitly provided. It extracts a meaningful
        component from the prompt name to use in output filenames.
        """
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