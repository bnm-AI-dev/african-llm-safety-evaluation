"""Abstract base class for all red-teaming attacks."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AttackResult:
    """Holds the outcome of a single attack execution.

    Attributes:
        attack_id: Unique identifier for the attack (e.g. ``"C4"``).
        model_id: HuggingFace model identifier that was queried.
        language: BCP-47 language tag or common name (e.g. ``"swahili"``).
        attack_type: Taxonomy attack type (e.g. ``"context_poisoning"``).
        risk_category: Top-level risk category (e.g. ``"self_harm"``).
        risk_subcategory: Subcategory within the risk category.
        prompt: The exact prompt sent to the model.
        prompt_english: English translation of the prompt.
        response: Raw model response text.
        score: Optional human/auto evaluator score (0–3 scale).
        generation_params: Generation hyperparameters used.
        device: Torch device string (``"mps"``, ``"cuda"``, ``"cpu"``).
        timestamp: ISO-8601 timestamp of the generation.
        notes: Free-text notes for the submission write-up.
        validated: Whether ``validate_response()`` found a safety failure.
    """

    attack_id: str
    model_id: str
    language: str
    attack_type: str
    risk_category: str
    risk_subcategory: str
    prompt: str
    prompt_english: str
    response: str
    score: int | None = None
    generation_params: dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""
    validated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Flat dictionary suitable for ``json.dump``.
        """
        return {
            "attack_id": self.attack_id,
            "model_id": self.model_id,
            "language": self.language,
            "attack_type": self.attack_type,
            "risk_category": self.risk_category,
            "risk_subcategory": self.risk_subcategory,
            "prompt": self.prompt,
            "prompt_english": self.prompt_english,
            "response": self.response,
            "score": self.score,
            "generation_params": self.generation_params,
            "device": self.device,
            "timestamp": self.timestamp,
            "notes": self.notes,
            "validated": self.validated,
        }


class AttackBase(ABC):
    """Abstract base class for all attack implementations.

    Subclasses must implement :meth:`generate_prompt` and
    :meth:`validate_response`. The :meth:`run` method orchestrates the full
    attack lifecycle: prompt generation → model inference → validation →
    result packaging.

    Args:
        attack_id: Unique identifier for this attack instance.
        model_id: HuggingFace model repo ID to target.
        language: Language of the attack prompt (common name or BCP-47).
        risk_category: Top-level risk taxonomy category.
        risk_subcategory: Subcategory within ``risk_category``.
        attack_type: Attack taxonomy type.
        notes: Optional notes to include in the result for submission.

    Example::

        class MyAttack(AttackBase):
            def generate_prompt(self) -> tuple[str, str]:
                return "Swahili prompt...", "English translation..."

            def validate_response(self, response: str) -> bool:
                return "harmful keyword" in response.lower()

        attack = MyAttack(
            attack_id="X1",
            model_id="sartifyllc/Pawa-Gemma-Swahili-2B",
            language="swahili",
            risk_category="self_harm",
            risk_subcategory="encouragement",
            attack_type="context_poisoning",
        )
        result = attack.run(tokenizer, model, device, gen_params)
    """

    def __init__(
        self,
        attack_id: str,
        model_id: str,
        language: str,
        risk_category: str,
        risk_subcategory: str,
        attack_type: str,
        notes: str = "",
    ) -> None:
        self.attack_id = attack_id
        self.model_id = model_id
        self.language = language
        self.risk_category = risk_category
        self.risk_subcategory = risk_subcategory
        self.attack_type = attack_type
        self.notes = notes

    @abstractmethod
    def generate_prompt(self) -> tuple[str, str]:
        """Build the attack prompt and its English translation.

        Returns:
            A ``(prompt, prompt_english)`` tuple. ``prompt`` is in the target
            language; ``prompt_english`` is a faithful English translation for
            the submission write-up.
        """

    @abstractmethod
    def validate_response(self, response: str) -> bool:
        """Determine whether the model response demonstrates a safety failure.

        Args:
            response: Raw decoded text from the model.

        Returns:
            ``True`` if the response contains actionable unsafe content,
            ``False`` if the model refused or produced a benign reply.
        """

    def run(
        self,
        tokenizer: Any,
        model: Any,
        device: Any,
        gen_params: dict[str, Any],
    ) -> AttackResult:
        """Orchestrate the full attack: prompt → inference → validate → result.

        Args:
            tokenizer: HuggingFace tokenizer (must support
                ``apply_chat_template``).
            model: HuggingFace causal LM (already moved to ``device``).
            device: ``torch.device`` the model lives on.
            gen_params: Generation hyperparameters forwarded to
                ``model.generate`` (``max_new_tokens``, ``temperature``, etc.).

        Returns:
            A populated :class:`AttackResult`.

        Raises:
            RuntimeError: If tokenization or generation fails.
        """
        import torch

        prompt, prompt_english = self.generate_prompt()
        logger.info("Running attack %s (%s / %s)", self.attack_id, self.risk_category, self.risk_subcategory)

        try:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(formatted, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    pad_token_id=tokenizer.eos_token_id,
                    **gen_params,
                )

            new_tokens = output_ids[0][input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        except Exception as exc:
            logger.error("Attack %s failed during inference: %s", self.attack_id, exc)
            raise RuntimeError(f"Inference failed for attack {self.attack_id}") from exc

        validated = self.validate_response(response)
        logger.info(
            "Attack %s complete — validated=%s, response_len=%d chars",
            self.attack_id,
            validated,
            len(response),
        )

        return AttackResult(
            attack_id=self.attack_id,
            model_id=self.model_id,
            language=self.language,
            attack_type=self.attack_type,
            risk_category=self.risk_category,
            risk_subcategory=self.risk_subcategory,
            prompt=prompt,
            prompt_english=prompt_english,
            response=response,
            generation_params=gen_params,
            device=str(device),
            notes=self.notes,
            validated=validated,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize attack metadata (without result) to a dictionary.

        Returns:
            Dictionary with all constructor fields. Useful for logging or
            building attack manifests before running.
        """
        return {
            "attack_id": self.attack_id,
            "model_id": self.model_id,
            "language": self.language,
            "attack_type": self.attack_type,
            "risk_category": self.risk_category,
            "risk_subcategory": self.risk_subcategory,
            "notes": self.notes,
        }
