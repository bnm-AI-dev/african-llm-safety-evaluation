"""Configuration dataclasses for the African LLM Safety Evaluation framework.

All config objects are plain ``dataclasses`` so they are serializable,
inspectable, and carry no external dependencies beyond the stdlib.  Pydantic
is intentionally avoided here to keep the runtime footprint minimal; validation
is done explicitly in ``__post_init__``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Language → model registry
# ---------------------------------------------------------------------------

#: Maps lowercase language names to their canonical HuggingFace model IDs.
#: Keep this in sync with the competition's official model registry.
MODEL_REGISTRY: dict[str, str] = {
    "swahili": "sartifyllc/Pawa-Gemma-Swahili-2B",
    "hausa": "UBC-NLP/AfriInstruct-Hausa-7B",
    "yoruba": "UBC-NLP/AfriInstruct-Yoruba-7B",
    "igbo": "UBC-NLP/AfriInstruct-Igbo-7B",
    "amharic": "EthioNLP/Amharic_LLAMA",
    "oromo": "EthioNLP/Amharic_LLAMA",
    "somali": "EthioNLP/Amharic_LLAMA",
    "tigrinya": "EthioNLP/Amharic_LLAMA",
    "zulu": "tobiolatunji/InkubaLM-0.4B",
    "afrikaans": "tobiolatunji/InkubaLM-0.4B",
}


# ---------------------------------------------------------------------------
# GenerationConfig
# ---------------------------------------------------------------------------


@dataclass
class GenerationConfig:
    """Hyperparameters forwarded directly to ``model.generate()``.

    Attributes:
        max_new_tokens: Maximum tokens the model may generate beyond the prompt.
        temperature: Sampling temperature. Higher = more random. Set to ``1.0``
            with ``do_sample=False`` for greedy decoding.
        top_p: Nucleus sampling probability mass cutoff.
        do_sample: Whether to use sampling (``True``) or greedy decoding
            (``False``).  Greedy decoding ignores ``temperature`` and
            ``top_p``.
        repetition_penalty: Penalise token repetition. ``1.0`` disables it.

    Example::

        cfg = GenerationConfig(max_new_tokens=512, temperature=0.3)
        output = model.generate(**inputs, **cfg.to_dict())
    """

    max_new_tokens: int = 800
    temperature: float = 0.4
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if not (0.0 < self.temperature <= 2.0):
            raise ValueError(f"temperature must be in (0, 2], got {self.temperature}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0, got {self.repetition_penalty}")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict suitable for unpacking into ``model.generate``.

        Returns:
            Dictionary of all generation parameters.
        """
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
        }


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Identifies and configures a model for loading.

    Attributes:
        model_id: HuggingFace repo ID (e.g. ``"sartifyllc/Pawa-Gemma-Swahili-2B"``).
        language: Language this model is being used for.  Used to look up
            ``model_id`` from :data:`MODEL_REGISTRY` when ``model_id`` is
            not provided directly.
        torch_dtype: Torch dtype string (``"float16"``, ``"bfloat16"``,
            ``"float32"``).  ``float16`` works on MPS and CUDA; ``bfloat16``
            is preferred on Ampere+ CUDA GPUs.
        low_cpu_mem_usage: Pass ``low_cpu_mem_usage=True`` to ``from_pretrained``
            to avoid doubling CPU RAM during load.
        device: Target device string.  ``"auto"`` resolves to MPS → CUDA →
            CPU at runtime.

    Example::

        cfg = ModelConfig(language="swahili")
        # cfg.model_id is automatically resolved from MODEL_REGISTRY
    """

    model_id: str = ""
    language: str = ""
    torch_dtype: str = "float16"
    low_cpu_mem_usage: bool = True
    device: str = "auto"

    _VALID_DTYPES: frozenset[str] = field(
        default_factory=lambda: frozenset({"float16", "bfloat16", "float32"}),
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.model_id:
            if not self.language:
                raise ValueError("Provide either model_id or language.")
            lang = self.language.lower()
            if lang not in MODEL_REGISTRY:
                raise ValueError(
                    f"Unknown language '{lang}'. "
                    f"Known languages: {sorted(MODEL_REGISTRY)}"
                )
            self.model_id = MODEL_REGISTRY[lang]

        if self.torch_dtype not in self._VALID_DTYPES:
            raise ValueError(
                f"torch_dtype must be one of {sorted(self._VALID_DTYPES)}, "
                f"got '{self.torch_dtype}'"
            )

    def resolve_device(self) -> str:
        """Resolve ``"auto"`` to a concrete device string.

        Resolution order: MPS (Apple Silicon) → CUDA → CPU.

        Returns:
            One of ``"mps"``, ``"cuda"``, or ``"cpu"``.
        """
        if self.device != "auto":
            return self.device

        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"


# ---------------------------------------------------------------------------
# AttackConfig
# ---------------------------------------------------------------------------


@dataclass
class AttackConfig:
    """Metadata describing a single attack run.

    This is the data-layer representation of an attack — it does not contain
    any executable logic.  Use it to drive :class:`~attacks.base.AttackBase`
    subclasses or to deserialize attack definitions from JSON/YAML manifests.

    Attributes:
        attack_id: Unique identifier (e.g. ``"C4"``).
        attack_type: Taxonomy attack type (``"context_poisoning"``,
            ``"indirect_request"``, ``"roleplay"``, …).
        risk_category: Top-level risk category.
        risk_subcategory: Subcategory within ``risk_category``.
        language: Target language.
        notes: Free-text annotation for the submission write-up.
        tags: Optional freeform labels for filtering/grouping.

    Example::

        cfg = AttackConfig(
            attack_id="C4",
            attack_type="indirect_request",
            risk_category="self_harm",
            risk_subcategory="encouragement",
            language="swahili",
        )
    """

    attack_id: str
    attack_type: str
    risk_category: str
    risk_subcategory: str
    language: str
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for required in ("attack_id", "attack_type", "risk_category", "risk_subcategory", "language"):
            if not getattr(self, required):
                raise ValueError(f"AttackConfig.{required} must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            All fields as a flat dictionary.
        """
        return {
            "attack_id": self.attack_id,
            "attack_type": self.attack_type,
            "risk_category": self.risk_category,
            "risk_subcategory": self.risk_subcategory,
            "language": self.language,
            "notes": self.notes,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttackConfig:
        """Deserialize from a dictionary (e.g. loaded from JSON).

        Args:
            data: Dictionary with at minimum the required fields.

        Returns:
            A new :class:`AttackConfig` instance.
        """
        return cls(
            attack_id=data["attack_id"],
            attack_type=data["attack_type"],
            risk_category=data["risk_category"],
            risk_subcategory=data["risk_subcategory"],
            language=data["language"],
            notes=data.get("notes", ""),
            tags=data.get("tags", []),
        )
