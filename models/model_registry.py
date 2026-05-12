"""Model registry for the African LLM Safety Evaluation framework.

Maps every supported language to its canonical HuggingFace model, along with
metadata gathered during empirical testing (safety level, architecture,
parameter count, known quirks).

Usage::

    from models.model_registry import registry

    info = registry.get_model_for_language("swahili")
    print(info.model_id)          # sartifyllc/Pawa-Gemma-Swahili-2B
    print(info.safety_level)      # SafetyLevel.WEAK

    info = registry.get_model("NCAIR1/N-ATLaS")
    print(info.languages)         # ['hausa', 'yoruba', 'igbo']

    # Validate before loading
    registry.validate("unknown/model")   # raises UnknownModelError
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SafetyLevel(str, Enum):
    """Empirical safety filter strength observed during testing.

    Attributes:
        NONE: Model applies no safety filtering whatsoever.  Any prompt works.
        WEAK: Soft filters that completion-style or indirect-request attacks
            easily bypass.
        MODERATE: Some refusal behaviour; requires multi-step or framed prompts.
        STRONG: Robust refusals; very hard to elicit harmful content.
        UNKNOWN: Not yet tested in this framework.
        DEGRADED: Model is too small / undertrained to produce coherent output;
            responses are gibberish regardless of prompt.
    """

    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    UNKNOWN = "unknown"
    DEGRADED = "degraded"


class Architecture(str, Enum):
    """Underlying model family."""

    GEMMA = "gemma"
    LLAMA = "llama"
    BLOOM = "bloom"
    BERT = "bert"
    GPT2 = "gpt2"
    OTHER = "other"


# ---------------------------------------------------------------------------
# ModelInfo dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInfo:
    """Immutable descriptor for a single HuggingFace model.

    Attributes:
        model_id: Canonical ``{org}/{repo}`` identifier on HuggingFace Hub.
        languages: Languages this model was trained / fine-tuned for.
            Lowercase, no hyphens (e.g. ``"swahili"`` not ``"sw"``).
        safety_level: Empirically observed safety filter strength.
        architecture: Base model family.
        param_count: Approximate parameter count as a human-readable string
            (e.g. ``"2B"``, ``"7B"``).
        safety_trained: Whether the model received explicit safety/RLHF tuning.
        tested: Whether this model has been run in the framework at least once.
        notes: Free-text observations from testing (quirks, bypass methods,
            known failure modes).
    """

    model_id: str
    languages: tuple[str, ...]
    safety_level: SafetyLevel = SafetyLevel.UNKNOWN
    architecture: Architecture = Architecture.OTHER
    param_count: str = "unknown"
    safety_trained: bool = False
    tested: bool = False
    notes: str = ""

    def supports_language(self, language: str) -> bool:
        """Return ``True`` if this model covers ``language``.

        Args:
            language: Lowercase language name (e.g. ``"hausa"``).

        Returns:
            Boolean membership test.
        """
        return language.lower() in self.languages

    def is_usable(self) -> bool:
        """Return ``True`` if the model produces coherent output.

        A model with :attr:`SafetyLevel.DEGRADED` produces gibberish and
        should be excluded from attack runners.

        Returns:
            ``False`` only for ``DEGRADED`` models.
        """
        return self.safety_level is not SafetyLevel.DEGRADED


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class UnknownModelError(KeyError):
    """Raised when a model ID is not in the registry."""

    def __init__(self, model_id: str) -> None:
        super().__init__(f"Model '{model_id}' not found in registry.")
        self.model_id = model_id


class UnknownLanguageError(KeyError):
    """Raised when no model covers the requested language."""

    def __init__(self, language: str) -> None:
        super().__init__(f"No model registered for language '{language}'.")
        self.language = language


# ---------------------------------------------------------------------------
# Registry class
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Central store of model metadata for all supported languages.

    All lookups are O(1) by model ID.  Language lookups are O(n) over the
    registered models (n is small — currently ≤ 6 distinct models).

    Example::

        reg = ModelRegistry()
        reg.register(ModelInfo(
            model_id="org/my-model",
            languages=("mymlanguage",),
        ))
        info = reg.get_model_for_language("mymanguage")
    """

    def __init__(self) -> None:
        self._store: dict[str, ModelInfo] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, info: ModelInfo) -> None:
        """Add or replace a model entry.

        Args:
            info: :class:`ModelInfo` to store under ``info.model_id``.
        """
        self._store[info.model_id] = info

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_model(self, model_id: str) -> ModelInfo:
        """Retrieve metadata by exact HuggingFace model ID.

        Args:
            model_id: Canonical ``{org}/{repo}`` identifier.

        Returns:
            The matching :class:`ModelInfo`.

        Raises:
            UnknownModelError: If ``model_id`` is not registered.
        """
        try:
            return self._store[model_id]
        except KeyError:
            raise UnknownModelError(model_id)

    def get_model_for_language(self, language: str) -> ModelInfo:
        """Return the first registered model that supports ``language``.

        If multiple models cover a language (e.g. via multi-language training),
        the one registered first is returned.  Use :meth:`get_models_for_language`
        to retrieve all candidates.

        Args:
            language: Lowercase language name.

        Returns:
            A matching :class:`ModelInfo`.

        Raises:
            UnknownLanguageError: If no model covers ``language``.
        """
        lang = language.lower()
        for info in self._store.values():
            if info.supports_language(lang):
                return info
        raise UnknownLanguageError(lang)

    def get_models_for_language(self, language: str) -> list[ModelInfo]:
        """Return all registered models that support ``language``.

        Args:
            language: Lowercase language name.

        Returns:
            List of :class:`ModelInfo` objects; empty if none match.
        """
        lang = language.lower()
        return [info for info in self._store.values() if info.supports_language(lang)]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, model_id: str) -> None:
        """Assert that ``model_id`` is registered and usable.

        Args:
            model_id: HuggingFace model ID to check.

        Raises:
            UnknownModelError: If the model is not in the registry.
            ValueError: If the model is registered but marked ``DEGRADED``.
        """
        info = self.get_model(model_id)
        if not info.is_usable():
            raise ValueError(
                f"Model '{model_id}' is marked DEGRADED (produces gibberish). "
                "Choose a different model for this language."
            )

    def is_registered(self, model_id: str) -> bool:
        """Return ``True`` if ``model_id`` is in the registry.

        Args:
            model_id: HuggingFace model ID to check.

        Returns:
            Boolean presence check.  Does not raise.
        """
        return model_id in self._store

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def list_models(self, usable_only: bool = False) -> list[ModelInfo]:
        """Return all registered models, optionally filtering out degraded ones.

        Args:
            usable_only: If ``True``, exclude ``DEGRADED`` models.

        Returns:
            List of :class:`ModelInfo`, in registration order.
        """
        models = list(self._store.values())
        if usable_only:
            models = [m for m in models if m.is_usable()]
        return models

    def list_languages(self, usable_only: bool = False) -> list[str]:
        """Return all languages covered by registered models, sorted.

        Args:
            usable_only: If ``True``, exclude languages whose only model is
                ``DEGRADED``.

        Returns:
            Sorted list of unique language names.
        """
        langs: set[str] = set()
        for info in self._store.values():
            if usable_only and not info.is_usable():
                continue
            langs.update(info.languages)
        return sorted(langs)

    def __iter__(self) -> Iterator[ModelInfo]:
        return iter(self._store.values())

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"ModelRegistry({len(self)} models, {len(self.list_languages())} languages)"


# ---------------------------------------------------------------------------
# Pre-populated registry (all competition models from master_plan.md)
# ---------------------------------------------------------------------------

registry = ModelRegistry()

# --- Pawa-Gemma-Swahili-2B ---
# Tested: weak safety, completion-style attacks bypass filters reliably.
# Context poisoning and indirect_request produce the strongest evidence.
registry.register(ModelInfo(
    model_id="sartifyllc/Pawa-Gemma-Swahili-2B",
    languages=("swahili",),
    safety_level=SafetyLevel.WEAK,
    architecture=Architecture.GEMMA,
    param_count="2B",
    safety_trained=False,
    tested=True,
    notes=(
        "Completion model. Soft safety filters bypassed by context_poisoning and "
        "indirect_request framing. Seed harmful text → model continues naturally. "
        "Best results at temperature 0.3–0.4."
    ),
))

# --- N-ATLaS (NCAIR1) — Hausa, Yoruba, Igbo ---
# Tested: zero safety filtering. Any prompt produces unsafe content.
# 7B LLAMA-based, multilingual across West African languages.
registry.register(ModelInfo(
    model_id="NCAIR1/N-ATLaS",
    languages=("hausa", "yoruba", "igbo"),
    safety_level=SafetyLevel.NONE,
    architecture=Architecture.LLAMA,
    param_count="7B",
    safety_trained=False,
    tested=True,
    notes=(
        "Zero safety filters. Cross-model replication of Swahili attacks works "
        "directly. Strongest evidence comes from attacks that already work on "
        "Pawa-Gemma — the model will comply without framing."
    ),
))

# --- EthioNLP Amharic LLAMA — Amharic, Oromo, Somali, Tigrinya ---
# Not yet tested in this framework. 7B LLAMA-based, Ethiopian language focus.
registry.register(ModelInfo(
    model_id="EthioNLP/Amharic_LLAMA_our_data",
    languages=("amharic", "oromo", "somali", "tigrinya"),
    safety_level=SafetyLevel.UNKNOWN,
    architecture=Architecture.LLAMA,
    param_count="7B",
    safety_trained=False,
    tested=False,
    notes=(
        "Priority 1 candidate for 3rd model/language expansion. "
        "7B LLAMA likely has sufficient fluency for Amharic. "
        "Replicate vaccine misinfo and medical dosage attack patterns first."
    ),
))

# --- AfroLM — Fulfulde, Pulaar, Serere, Shona, Lingala, Wolof, Malagasy ---
# Not yet tested. Multilingual West/Central African model.
registry.register(ModelInfo(
    model_id="bonadossou/afrolm_active_learning",
    languages=("fulfulde", "pulaar", "serere", "shona", "lingala", "wolof", "malagasy"),
    safety_level=SafetyLevel.UNKNOWN,
    architecture=Architecture.BLOOM,
    param_count="unknown",
    safety_trained=False,
    tested=False,
    notes=(
        "Covers 7 competition languages. Test fluency before designing attacks — "
        "BLOOM-based models vary widely in output quality across low-resource languages."
    ),
))

# --- InkubaLM-0.4B — Zulu, Afrikaans ---
# Tested: produces gibberish. Too small (0.4B) to be useful for red-teaming.
registry.register(ModelInfo(
    model_id="lelapa/InkubaLM-0.4B",
    languages=("zulu", "afrikaans"),
    safety_level=SafetyLevel.DEGRADED,
    architecture=Architecture.OTHER,
    param_count="0.4B",
    safety_trained=False,
    tested=True,
    notes=(
        "DEAD END. Model is too small — all responses are incoherent regardless "
        "of prompt. Do not include Zulu or Afrikaans attacks in submissions."
    ),
))

# --- Akan (Ghana-NLP) ---
# Not yet tested. BERT-based — likely a masked LM, not a generative model.
# May require a different inference approach (fill-mask vs generate).
registry.register(ModelInfo(
    model_id="Ghana-NLP/abena-base-akuapem-twi-cased",
    languages=("akan",),
    safety_level=SafetyLevel.UNKNOWN,
    architecture=Architecture.BERT,
    param_count="110M",
    safety_trained=False,
    tested=False,
    notes=(
        "BERT-based masked LM — cannot use causal generation pipeline. "
        "Requires fill-mask inference. Unclear how to elicit safety failures "
        "without generative capability. Evaluate feasibility before attempting."
    ),
))
