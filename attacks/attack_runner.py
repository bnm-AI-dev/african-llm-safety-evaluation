"""CLI runner for red-teaming African LLMs.

Loads one or more attack classes from the registry, runs them against the
target model, and writes structured JSON results.  Handles CTRL+C gracefully
by flushing partial results before exit.

Usage examples::

    # Run all registered attacks for Swahili (model auto-resolved)
    python attacks/attack_runner.py --language swahili

    # Run a single named attack class
    python attacks/attack_runner.py --attack-class SelfHarmEncouragementAttack

    # Override model and generation parameters
    python attacks/attack_runner.py \\
        --attack-class SelfHarmEncouragementAttack \\
        --model-id sartifyllc/Pawa-Gemma-Swahili-2B \\
        --max-new-tokens 512 --temperature 0.3 --top-p 0.9

    # Write output and structured log to custom paths
    python attacks/attack_runner.py \\
        --attack-class SelfHarmEncouragementAttack \\
        --output results/c4_run.json \\
        --log-file logs/c4_run.log

    # Run every registered attack and be verbose
    python attacks/attack_runner.py --verbose

Notes:
    - Run from the repo root: ``python attacks/attack_runner.py ...``
    - Activate the virtual environment first: ``source .venv/bin/activate``
    - New attack classes must be added to :data:`ATTACK_REGISTRY`.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Type

# ---------------------------------------------------------------------------
# Path bootstrap — must happen before any local imports so that
# "from base import ..." (non-relative, linter-normalised style) resolves
# to attacks/base.py regardless of the working directory.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local imports (non-relative, consistent with linter-normalised style)
from base import AttackBase, AttackResult
from context_poisoning import SelfHarmEncouragementAttack

# Parent directory on path so config.py is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GenerationConfig, ModelConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger("attack_runner")

# ---------------------------------------------------------------------------
# Attack registry — add new classes here as they are implemented.
# ---------------------------------------------------------------------------
ATTACK_REGISTRY: dict[str, Type[AttackBase]] = {
    "SelfHarmEncouragementAttack": SelfHarmEncouragementAttack,
}


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(log_file: str | None, verbose: bool) -> None:
    """Configure root logger with console and optional file handlers.

    Args:
        log_file: Path to write a DEBUG-level log file.  ``None`` disables
            file logging.
        verbose: If ``True``, set the console handler to DEBUG; otherwise INFO.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)
        logger.debug("File logging → %s", log_file)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    model_cfg: ModelConfig,
    device: torch.device,
) -> tuple[Any, Any]:
    """Load tokenizer and causal LM onto ``device``.

    Args:
        model_cfg: Describes which model to load and how.
        device: Target torch device.

    Returns:
        ``(tokenizer, model)`` tuple, model already in eval mode.

    Raises:
        RuntimeError: If the model fails to load (missing weights, OOM, etc.).
    """
    logger.info("Loading %s …", model_cfg.model_id)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_id)

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.model_id,
            torch_dtype=dtype_map[model_cfg.torch_dtype],
            low_cpu_mem_usage=model_cfg.low_cpu_mem_usage,
        )
        model.to(device)
        model.eval()
        logger.info("Model ready on %s", device)
        return tokenizer, model

    except Exception as exc:
        raise RuntimeError(f"Failed to load {model_cfg.model_id}: {exc}") from exc


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------


def save_results(
    results: list[AttackResult],
    output_path: str,
    run_metadata: dict[str, Any],
) -> None:
    """Write results to a structured JSON file.

    The output format is::

        {
          "run_id": "...",
          "timestamp": "...",
          "model_id": "...",
          "device": "...",
          "generation_params": {...},
          "total_attacks": N,
          "validated_count": N,
          "results": [...]
        }

    Args:
        results: Completed (or partial, on interrupt) attack results.
        output_path: Destination file path.  Parent directories are created.
        run_metadata: Top-level metadata dict to merge into the output.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **run_metadata,
        "total_attacks": len(results),
        "validated_count": sum(1 for r in results if r.validated),
        "results": [r.to_dict() for r in results],
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    logger.info("Results saved → %s  (%d attacks)", output_path, len(results))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="attack_runner",
        description="Run red-teaming attacks against African LLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Attack selection
    sel = parser.add_argument_group("attack selection")
    sel.add_argument(
        "--attack-class",
        metavar="CLASS",
        nargs="+",
        choices=list(ATTACK_REGISTRY),
        help=(
            "One or more attack class names to run.  "
            f"Available: {', '.join(ATTACK_REGISTRY)}.  "
            "Omit to run all registered attacks."
        ),
    )
    sel.add_argument(
        "--language",
        metavar="LANG",
        help="Filter attacks to those targeting this language (e.g. swahili).",
    )

    # Model
    mdl = parser.add_argument_group("model")
    mdl.add_argument(
        "--model-id",
        metavar="HF_REPO",
        help="HuggingFace model ID.  Overrides the language-based registry lookup.",
    )
    mdl.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model weights.",
    )
    mdl.add_argument(
        "--device",
        default="auto",
        help="Torch device (auto, cpu, cuda, mps).",
    )

    # Generation
    gen = parser.add_argument_group("generation")
    gen.add_argument("--max-new-tokens", type=int, default=800)
    gen.add_argument("--temperature", type=float, default=0.4)
    gen.add_argument("--top-p", type=float, default=0.9)
    gen.add_argument("--no-sample", action="store_true", help="Use greedy decoding.")
    gen.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty passed to model.generate (>= 1.0).",
    )

    # Output
    out = parser.add_argument_group("output")
    out.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help=(
            "JSON output file.  Defaults to "
            "results/attack_results_<timestamp>.json."
        ),
    )
    out.add_argument(
        "--log-file",
        metavar="PATH",
        default=None,
        help="Write DEBUG-level log to this file in addition to stdout.",
    )
    out.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Set console log level to DEBUG.",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: parse args → load model → run attacks → save results."""
    parser = build_parser()
    args = parser.parse_args()

    # Default output path
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"results/attack_results_{ts}.json"

    setup_logging(args.log_file, args.verbose)
    logger.info("=== African LLM Safety Evaluation — Attack Runner ===")

    # --- Build config objects ---
    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample,
        repetition_penalty=args.repetition_penalty,
    )
    gen_params = gen_cfg.to_dict()
    logger.debug("Generation config: %s", gen_params)

    model_cfg = ModelConfig(
        model_id=args.model_id or "",
        language=args.language or "",
        torch_dtype=args.dtype,
        device=args.device,
    )
    device = torch.device(model_cfg.resolve_device())

    # --- Instantiate attack classes ---
    class_names: list[str] = args.attack_class or list(ATTACK_REGISTRY)
    attacks: list[AttackBase] = []
    for name in class_names:
        instance = ATTACK_REGISTRY[name]()
        if args.language and instance.language != args.language.lower():
            logger.debug("Skipping %s (language mismatch)", name)
            continue
        if args.model_id:
            instance.model_id = args.model_id
        attacks.append(instance)

    if not attacks:
        logger.error("No attacks matched the given filters.  Exiting.")
        sys.exit(1)

    logger.info(
        "Running %d attack(s): %s",
        len(attacks),
        ", ".join(a.attack_id for a in attacks),
    )

    # Resolve model_id from first attack if not overridden
    effective_model_id = args.model_id or attacks[0].model_id

    # --- Load model ---
    model_cfg.model_id = effective_model_id
    try:
        tokenizer, model = load_model(model_cfg, device)
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    run_metadata: dict[str, Any] = {
        "run_id": ts,
        "timestamp": datetime.now().isoformat(),
        "model_id": effective_model_id,
        "device": str(device),
        "generation_params": gen_params,
    }

    # --- SIGINT handler: flush partial results ---
    completed: list[AttackResult] = []

    def _handle_interrupt(sig: int, frame: Any) -> None:  # noqa: ANN001
        logger.warning("\nInterrupted — saving %d partial result(s)…", len(completed))
        save_results(completed, output_path, run_metadata)
        sys.exit(130)

    signal.signal(signal.SIGINT, _handle_interrupt)

    # --- Run attacks ---
    progress = tqdm(attacks, unit="attack", desc="Attacks", colour="green")
    for attack in progress:
        progress.set_description(f"{attack.attack_id} ({attack.risk_category})")
        logger.debug("Starting %s", attack.attack_id)

        try:
            result = attack.run(tokenizer, model, device, gen_params)
            completed.append(result)
            status = "UNSAFE" if result.validated else "safe"
            tqdm.write(
                f"[{result.attack_id}] {status} — "
                f"{len(result.response)} chars"
            )
            logger.debug("Response preview: %s", result.response[:120])

        except RuntimeError as exc:
            logger.error("Attack %s failed: %s — skipping", attack.attack_id, exc)

    progress.close()

    # --- Persist ---
    save_results(completed, output_path, run_metadata)

    validated = sum(1 for r in completed if r.validated)
    logger.info(
        "Done. %d/%d attacks produced unsafe responses.",
        validated,
        len(completed),
    )


if __name__ == "__main__":
    main()
