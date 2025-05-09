# src/trialmesh/llm/llama_runner.py

import hashlib
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, NamedTuple

from vllm import LLM, EngineArgs, SamplingParams

class LlamaResponse(NamedTuple):
    """Container for LLaMA response with token counts and prompt info.

    This named tuple stores the model's response text along with metadata
    about token usage and the original prompts that generated the response.

    Attributes:
        text (str): The generated text response from the model
        input_tokens (int): Number of tokens in the input prompt
        output_tokens (int): Number of tokens in the generated response
        user_prompt (str): The user prompt that generated this response
        system_prompt (Optional[str]): The system prompt used (if any)
    """
    text: str
    input_tokens: int
    output_tokens: int
    user_prompt: str
    system_prompt: Optional[str] = None

class LlamaRunner:
    """Runner for LLaMA models using vLLM with caching support.

    This class provides an interface to the LLaMA language model through vLLM,
    with efficient tensor parallelism across multiple GPUs. It includes a robust
    caching system that stores previous responses based on input hashes to avoid
    redundant computation, improve reproducibility, and speed up repeated runs.

    Attributes:
        model_path (str): Path to the LLaMA model
        use_cache (bool): Whether to use caching for responses
        cache_dir (Path): Directory to store cached responses
        llm (vllm.LLM): The vLLM model instance
        default_sampling_params (vllm.SamplingParams): Default generation parameters
    """

    def __init__(self, model_path: str, cache_dir: Optional[str] = None,
                 tensor_parallel_size: int = 4, max_tokens: int = 1024,
                 max_model_len: int = 2048, max_batch_size: int = 8,
                 use_cache: bool = True, temperature: float = 0.0,
                 top_p: float = 1.0, top_k: int = -1):
        """Initialize the LlamaRunner with model and caching configuration.

        Args:
            model_path: Path to the LLaMA model
            cache_dir: Directory to store cached responses (default: cache/llm_responses)
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            max_tokens: Maximum sequence length for output generation
            max_model_len: Maximum model context length (input + output)
            max_batch_size: Maximum batch size for efficient inference
            use_cache: Whether to use caching for responses
            temperature: Sampling temperature (0.0 for deterministic output)
            top_p: Top-p sampling parameter (nucleus sampling)
            top_k: Top-k sampling parameter (-1 to disable)
        """

        # Store config
        self.model_path = model_path
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/llm_responses")

        # Ensure cache dir exists
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize engine args
        engine_args = EngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_seq_len_to_capture=max_tokens,
            disable_custom_all_reduce=True,
            max_num_seqs=max_batch_size,
            max_model_len=max_model_len,
            enforce_eager=True,
            seed=42
        )

        # Initialize default sampling params
        self.default_sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            min_tokens=10,
            seed=42
        )

        # Initialize LLM engine
        self.llm = LLM(**vars(engine_args))

        logging.info(f"LlamaRunner initialized with model: {model_path}")
        logging.info(f"LlamaRunner temperature: {temperature}")

    def _get_cache_key(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a deterministic hash key for caching based on prompt input.

        This method creates a unique identifier for caching LLM responses
        based on the content of the prompt and system prompt.

        Args:
            prompt: User prompt text
            system_prompt: Optional system prompt text

        Returns:
            A SHA-256 hash string that uniquely identifies this prompt combination
        """
        content_to_hash = prompt
        if system_prompt:
            content_to_hash = f"{system_prompt}|||{prompt}"

        # Create a deterministic hash
        key = hashlib.sha256(content_to_hash.encode('utf-8')).hexdigest()
        return key

    def _get_cached_response(self, cache_key: str) -> Optional[LlamaResponse]:
        """Retrieve cached response if it exists.

        Args:
            cache_key: Hash key for the prompt

        Returns:
            The cached LlamaResponse if found, None otherwise
        """
        if not self.use_cache:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logging.debug(f"Cache hit for key: {cache_key}")
                return LlamaResponse(
                    text=data['text'],
                    input_tokens=data['input_tokens'],
                    output_tokens=data['output_tokens'],
                    user_prompt=data.get('user_prompt', ""),
                    system_prompt=data.get('system_prompt', "")
                )
        return None

    def _cache_response(self, cache_key: str, response: LlamaResponse) -> None:
        """Save response with token counts and prompts to cache.

        Args:
            cache_key: Hash key for the prompt
            response: The LlamaResponse to cache
        """
        if not self.use_cache:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        data = {
            'text': response.text,
            'input_tokens': response.input_tokens,
            'output_tokens': response.output_tokens,
            'user_prompt': response.user_prompt,
            'system_prompt': response.system_prompt if response.system_prompt else ""
        }

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)  # Pretty print for readability
        logging.debug(f"Cached response for key: {cache_key}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> LlamaResponse:
        """Generate text from a prompt with optional system prompt.

        This method handles the core generation functionality, including:
        1. Checking the cache for existing responses
        2. Configuring generation parameters
        3. Running inference via vLLM
        4. Tracking token usage
        5. Caching the response

        Args:
            prompt: User prompt text to generate from
            system_prompt: Optional system prompt for chat models
            max_tokens: Maximum number of tokens to generate (overrides default)
            temperature: Sampling temperature (overrides default)

        Returns:
            A LlamaResponse containing the generated text and metadata
        """
        # Try to get from cache first
        cache_key = self._get_cache_key(prompt, system_prompt)
        cached_response = self._get_cached_response(cache_key)
        if cached_response is not None:
            return cached_response

        # Create sampling params with any overrides
        sampling_params = SamplingParams(
            temperature=temperature if temperature is not None else self.default_sampling_params.temperature,
            top_p=self.default_sampling_params.top_p,
            top_k=self.default_sampling_params.top_k,
            max_tokens=max_tokens if max_tokens is not None else self.default_sampling_params.max_tokens,
            min_tokens=self.default_sampling_params.min_tokens,
        )
        logging.info(f"LlamaRunner SamplingParams temperature: {temperature}")

        # Run generation
        if system_prompt:
            # Use chat interface
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            outputs = self.llm.chat([messages], sampling_params)
            response_text = outputs[0].outputs[0].text
            input_tokens = len(outputs[0].prompt_token_ids) if outputs[0].prompt_token_ids is not None else 0
            output_tokens = len(outputs[0].outputs[0].token_ids) if outputs[0].outputs[0].token_ids is not None else 0
        else:
            # Use completion interface
            outputs = self.llm.generate([prompt], sampling_params)
            response_text = outputs[0].outputs[0].text
            input_tokens = len(outputs[0].prompt_token_ids) if outputs[0].prompt_token_ids is not None else 0
            output_tokens = len(outputs[0].outputs[0].token_ids) if outputs[0].outputs[0].token_ids is not None else 0

        # Create response object with prompt info
        response = LlamaResponse(
            text=response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            user_prompt=prompt,
            system_prompt=system_prompt
        )

        # Cache the response
        self._cache_response(cache_key, response)

        return response

    def generate_batch(self, prompts: List[str], system_prompt: Optional[str] = None,
                       max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> List[LlamaResponse]:
        """Generate responses for a batch of prompts.

        This method efficiently handles multiple prompts by:
        1. Checking which prompts are already cached
        2. Only running inference for uncached prompts
        3. Using vLLM's batching capabilities for efficient processing
        4. Reconstructing the full results set with cached and new responses

        Args:
            prompts: List of user prompt texts
            system_prompt: Optional system prompt to use for all prompts
            max_tokens: Maximum number of tokens to generate (overrides default)
            temperature: Sampling temperature (overrides default)

        Returns:
            List of LlamaResponse objects, one for each input prompt
        """
        # Find which prompts are cached and which need generation
        uncached_prompts = []
        uncached_indices = []
        results = [None] * len(prompts)

        # Check cache for each prompt
        for i, prompt in enumerate(prompts):
            cache_key = self._get_cache_key(prompt, system_prompt)
            cached_response = self._get_cached_response(cache_key)

            if cached_response is not None:
                results[i] = cached_response
            else:
                uncached_prompts.append(prompt)
                uncached_indices.append(i)

        # If all prompts were cached, return early
        if not uncached_prompts:
            return results

        # Create sampling params with any overrides
        sampling_params = SamplingParams(
            temperature=temperature if temperature is not None else self.default_sampling_params.temperature,
            top_p=self.default_sampling_params.top_p,
            top_k=self.default_sampling_params.top_k,
            max_tokens=max_tokens if max_tokens is not None else self.default_sampling_params.max_tokens,
            min_tokens=self.default_sampling_params.min_tokens,
        )

        # Generate for uncached prompts
        if system_prompt:
            # Use chat interface
            conversations = []
            for prompt in uncached_prompts:
                conversations.append([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ])
            outputs = self.llm.chat(conversations, sampling_params)

            # Extract responses with token counts and prompts
            new_responses = []
            for i, output in enumerate(outputs):
                prompt = uncached_prompts[i]
                response_text = output.outputs[0].text
                input_tokens = len(output.prompt_token_ids) if output.prompt_token_ids is not None else 0
                output_tokens = len(output.outputs[0].token_ids) if output.outputs[0].token_ids is not None else 0
                new_responses.append(LlamaResponse(
                    text=response_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    user_prompt=prompt,
                    system_prompt=system_prompt
                ))
        else:
            # Use completion interface
            outputs = self.llm.generate(uncached_prompts, sampling_params)

            # Extract responses with token counts
            new_responses = []
            for output in outputs:
                response_text = output.outputs[0].text
                input_tokens = len(output.prompt_token_ids) if output.prompt_token_ids is not None else 0
                output_tokens = len(output.outputs[0].token_ids) if output.outputs[0].token_ids is not None else 0
                new_responses.append(LlamaResponse(
                    text=response_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                ))

        # Cache the new responses and fill in results
        for i, (prompt, response) in enumerate(zip(uncached_prompts, new_responses)):
            cache_key = self._get_cache_key(prompt, system_prompt)
            self._cache_response(cache_key, response)
            results[uncached_indices[i]] = response

        return results