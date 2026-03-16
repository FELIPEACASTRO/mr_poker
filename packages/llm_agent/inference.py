"""LLM inference wrapper for poker action generation.

Provides a thin abstraction over HuggingFace transformers for generating
poker actions from PokerBench-format prompts. Supports lazy model loading,
optional quantization, and thread-safe usage.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a GTO poker solver for 6-max No Limit Texas Hold'em. "
    "Given a game state description, respond with ONLY the optimal action. "
    "Valid actions: check, fold, call, bet X, raise X, all-in. "
    "Do not explain your reasoning."
)


class LLMInference:
    """Lazy-loading LLM inference engine for poker decisions.

    The model is loaded on first call to :meth:`generate` and cached for
    subsequent calls.  Thread-safe via a lock around model loading.
    """

    def __init__(self, model_id: str, *, quantize: bool = False, max_new_tokens: int = 20) -> None:
        self.model_id = model_id
        self.quantize = quantize
        self.max_new_tokens = max_new_tokens
        self._pipeline: Any | None = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        """Load the model and tokenizer (called once on first generate)."""
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            raise RuntimeError(
                "The 'transformers' library is required for LLM inference. "
                "Install it with: pip install transformers torch"
            ) from exc

        kwargs: dict[str, Any] = {
            "task": "text-generation",
            "model": self.model_id,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "return_full_text": False,
        }

        if self.quantize:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["model_kwargs"] = {
                    "quantization_config": BitsAndBytesConfig(load_in_4bit=True)
                }
            except ImportError:
                logger.warning("bitsandbytes not available, loading without quantization")

        logger.info("Loading LLM model: %s (quantize=%s)", self.model_id, self.quantize)
        self._pipeline = hf_pipeline(**kwargs)
        logger.info("LLM model loaded successfully")

    def generate(self, prompt: str) -> str:
        """Generate a poker action from a PokerBench-format prompt.

        Returns the raw text output from the model (e.g. 'check', 'bet 10').
        """
        with self._lock:
            if self._pipeline is None:
                self._load()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        result = self._pipeline(messages)  # type: ignore[misc]
        if result and len(result) > 0:
            text = result[0].get("generated_text", "")
            return text.strip()
        return "check"

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded into memory."""
        return self._pipeline is not None
