"""Model loading with in-process caching and memory optimisation.

Provides a single :func:`load_model` entry point that returns a
``(tokenizer, model)`` pair.  Repeated calls for the same ``model_id`` on the
same ``device`` are served from the in-process cache with zero overhead.

Memory management:
    - ``low_cpu_mem_usage=True`` avoids doubling CPU RAM during ``from_pretrained``
      by loading shard-by-shard directly into the target dtype.
    - MPS / CUDA models are moved to device after load.
    - :func:`unload_model` frees GPU/MPS memory and removes the cache entry.
    - :func:`clear_cache` unloads every cached model (useful between attack
      batches that target different models).

Usage::

    import torch
    from models.model_loader import load_model, unload_model, clear_cache

    device = torch.device("mps")

    # First call — downloads/loads from disk (~10–30 s for 7B)
    tokenizer, model = load_model("NCAIR1/N-ATLaS", device)

    # Second call — instant cache hit
    tokenizer, model = load_model("NCAIR1/N-ATLaS", device)

    # Free memory when done
    unload_model("NCAIR1/N-ATLaS")

    # Or nuke everything
    clear_cache()
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal cache
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    """One cached model + tokenizer pair."""

    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    device: str
    dtype: str


# ``model_id`` → ``_CacheEntry``.  Module-level so it persists for the
# lifetime of the interpreter process.
_CACHE: dict[str, _CacheEntry] = {}


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

_BEST_DTYPE: dict[str, str] = {
    "mps": "float16",    # MPS does not support bfloat16
    "cuda": "float16",   # safe default; use bfloat16 on Ampere+ if preferred
    "cpu": "float32",    # float16 on CPU is slow; only use if RAM is critical
}


def _resolve_dtype(dtype: str | None, device: torch.device) -> torch.dtype:
    """Return the torch dtype to use, falling back to a device-appropriate default.

    Args:
        dtype: Caller-requested dtype string, or ``None`` for auto-select.
        device: Target device (used to pick the default when ``dtype`` is ``None``).

    Returns:
        Resolved :class:`torch.dtype`.

    Raises:
        ValueError: If ``dtype`` is not one of ``float16``, ``bfloat16``,
            ``float32``.
    """
    if dtype is None:
        dtype = _BEST_DTYPE.get(device.type, "float16")

    if dtype not in _DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype '{dtype}'. Choose from: {sorted(_DTYPE_MAP)}"
        )
    return _DTYPE_MAP[dtype]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_model(
    model_id: str,
    device: torch.device,
    *,
    dtype: str | None = None,
    low_cpu_mem_usage: bool = True,
    force_reload: bool = False,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load (or retrieve from cache) a tokenizer and causal LM.

    The function validates the model against :data:`~models.model_registry.registry`
    before attempting to load, so degraded models (e.g. InkubaLM-0.4B) are
    rejected immediately without hitting HuggingFace Hub.

    Args:
        model_id: HuggingFace ``{org}/{repo}`` identifier.
        device: Target torch device.  The model is moved to this device after
            loading.
        dtype: Weight dtype (``"float16"``, ``"bfloat16"``, ``"float32"``).
            Defaults to ``"float16"`` on MPS/CUDA, ``"float32"`` on CPU.
        low_cpu_mem_usage: Pass ``low_cpu_mem_usage=True`` to ``from_pretrained``
            to avoid peak RAM doubling during load.
        force_reload: If ``True``, evict any existing cache entry and reload
            from disk.  Useful after OOM recovery.

    Returns:
        ``(tokenizer, model)`` tuple.  The model is in ``eval()`` mode and
        gradient computation is disabled.

    Raises:
        ValueError: If the model is registered as ``DEGRADED``.
        RuntimeError: If loading fails (missing weights, CUDA OOM, etc.).

    Example::

        device = torch.device("mps")
        tok, mdl = load_model("sartifyllc/Pawa-Gemma-Swahili-2B", device)
        inputs = tok("Hello", return_tensors="pt").to(device)
        with torch.no_grad():
            out = mdl.generate(**inputs, max_new_tokens=50)
    """
    # --- Registry validation (import here to avoid circular imports) ---
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from models.model_registry import registry, UnknownModelError
        if registry.is_registered(model_id):
            registry.validate(model_id)   # raises ValueError for DEGRADED
        else:
            logger.warning(
                "Model '%s' is not in the registry — proceeding without metadata.",
                model_id,
            )
    except ImportError:
        logger.debug("Registry not available; skipping validation.")

    # --- Cache hit ---
    cache_key = model_id
    if not force_reload and cache_key in _CACHE:
        entry = _CACHE[cache_key]
        if entry.device == str(device):
            logger.debug("Cache hit: %s on %s", model_id, device)
            return entry.tokenizer, entry.model
        else:
            # Device mismatch — evict and reload on new device
            logger.info(
                "Cache device mismatch for %s (%s → %s) — reloading.",
                model_id,
                entry.device,
                device,
            )
            _evict(cache_key)

    # --- Load ---
    resolved_dtype = _resolve_dtype(dtype, device)
    logger.info(
        "Loading %s → %s [dtype=%s, low_cpu_mem=%s]",
        model_id,
        device,
        resolved_dtype,
        low_cpu_mem_usage,
    )

    try:
        tokenizer = _load_tokenizer(model_id)
        model = _load_causal_lm(model_id, resolved_dtype, low_cpu_mem_usage, device)
    except Exception as exc:
        _raise_load_error(model_id, exc)

    _CACHE[cache_key] = _CacheEntry(
        tokenizer=tokenizer,
        model=model,
        device=str(device),
        dtype=str(resolved_dtype),
    )
    logger.info("Loaded and cached: %s", model_id)
    return tokenizer, model


def unload_model(model_id: str) -> None:
    """Remove a model from the cache and free device memory.

    Calls ``del`` on the model tensor, then runs the Python garbage collector
    and empties the CUDA/MPS allocator cache so the memory is returned to the
    OS promptly.

    Args:
        model_id: HuggingFace model ID to evict.  No-op if not cached.
    """
    if model_id not in _CACHE:
        logger.debug("unload_model: '%s' not in cache — nothing to do.", model_id)
        return
    _evict(model_id)
    logger.info("Unloaded: %s", model_id)


def clear_cache() -> None:
    """Unload every cached model and free all device memory.

    Use this between attack batches that target different models to avoid
    holding multiple large models in memory simultaneously.
    """
    keys = list(_CACHE)
    for key in keys:
        _evict(key)
    logger.info("Model cache cleared (%d model(s) unloaded).", len(keys))


def list_cached_models() -> list[str]:
    """Return the model IDs currently held in cache.

    Returns:
        List of ``model_id`` strings; empty if the cache is cold.
    """
    return list(_CACHE)


def get_memory_footprint(model_id: str) -> dict[str, Any]:
    """Return memory usage statistics for a cached model.

    Args:
        model_id: HuggingFace model ID.

    Returns:
        Dictionary with ``"param_bytes"`` (total parameter bytes) and
        ``"param_millions"`` (parameter count in millions).  Empty dict if the
        model is not cached.
    """
    if model_id not in _CACHE:
        return {}
    model = _CACHE[model_id].model
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return {
        "param_bytes": param_bytes,
        "param_mb": round(param_bytes / 1024 ** 2, 1),
        "param_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    """Download and return the tokenizer for ``model_id``.

    Args:
        model_id: HuggingFace model ID.

    Returns:
        Loaded tokenizer.

    Raises:
        Exception: Propagated from ``AutoTokenizer.from_pretrained``.
    """
    logger.debug("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Ensure a pad token exists (required for batched generation)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug("Set pad_token = eos_token for %s", model_id)

    return tokenizer


def _load_causal_lm(
    model_id: str,
    dtype: torch.dtype,
    low_cpu_mem_usage: bool,
    device: torch.device,
) -> PreTrainedModel:
    """Download, instantiate, and move the causal LM to ``device``.

    Args:
        model_id: HuggingFace model ID.
        dtype: Weight tensor dtype.
        low_cpu_mem_usage: Passed directly to ``from_pretrained``.
        device: Target device.

    Returns:
        Model in ``eval()`` mode with gradients disabled.
    """
    logger.debug("Loading weights: %s (dtype=%s)", model_id, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    model.to(device)
    model.eval()

    # Disable gradient tracking globally for inference — saves memory and
    # removes the overhead of building the autograd graph on every forward pass.
    for param in model.parameters():
        param.requires_grad_(False)

    return model


def _evict(model_id: str) -> None:
    """Remove a cache entry and aggressively free device memory.

    Args:
        model_id: Cache key to evict.
    """
    entry = _CACHE.pop(model_id, None)
    if entry is None:
        return

    device_type = entry.model.device.type if hasattr(entry.model, "device") else "cpu"
    del entry.model
    del entry.tokenizer
    del entry

    gc.collect()

    if device_type == "cuda":
        torch.cuda.empty_cache()
        logger.debug("CUDA cache emptied after unloading %s", model_id)
    elif device_type == "mps":
        # MPS does not expose empty_cache in all torch versions
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
            logger.debug("MPS cache emptied after unloading %s", model_id)


def _raise_load_error(model_id: str, exc: Exception) -> None:
    """Produce a helpful RuntimeError with context about the failure.

    Args:
        model_id: The model that failed to load.
        exc: Original exception from ``from_pretrained`` or ``.to(device)``.

    Raises:
        RuntimeError: Always.
    """
    msg = str(exc)

    if "out of memory" in msg.lower() or "oom" in msg.lower():
        hint = (
            "Out of memory. Try: --dtype float16, a smaller model, "
            "or clear other cached models with clear_cache() first."
        )
    elif "does not exist" in msg.lower() or "404" in msg.lower():
        hint = (
            f"Model '{model_id}' not found on HuggingFace Hub. "
            "Check the model ID in model_registry.py."
        )
    elif "connection" in msg.lower() or "timeout" in msg.lower():
        hint = (
            "Network error. Check your internet connection or set "
            "TRANSFORMERS_OFFLINE=1 to use a locally cached copy."
        )
    else:
        hint = f"Original error: {exc}"

    raise RuntimeError(f"Failed to load '{model_id}'. {hint}") from exc
