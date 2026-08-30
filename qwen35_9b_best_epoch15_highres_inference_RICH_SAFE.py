#!/usr/bin/env python3
"""
Inference-only resolution sweep for the BEST Qwen3.5-9B checkpoint.

Target checkpoint:
    outputs/qwen35_9b_fulltrain/checkpoints/epoch_1.5

NO training is performed.
Default inference: r768 and r896.

The script imports the original successful training module so the exact prompt,
VQADataset, collator, answer-token scoring, checkpoint restore, and submission
formatting are reused.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from dotenv import load_dotenv
except ImportError as exc:
    raise RuntimeError(
        "python-dotenv is required. Install with: pip install -U python-dotenv"
    ) from exc

try:
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
except ImportError as exc:
    raise RuntimeError(
        "A recent Transformers build is required. Install with:\n"
        'pip install -U "transformers @ git+https://github.com/huggingface/transformers.git"'
    ) from exc

from peft import PeftModel


console = Console(highlight=False)
DEFAULT_OUTPUT_DIR = Path("outputs/qwen35_9b_fulltrain")
DEFAULT_CHECKPOINT = Path("outputs/qwen35_9b_fulltrain/checkpoints/epoch_1.5")
DEFAULT_SIDES = (768, 896)
DEFAULT_TRAINING_MODULE = "train_qwen35_9b_fulltrain_checkpoints_wandb"


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(sorted({int(x.strip()) for x in value.split(",") if x.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--sides must be comma-separated integers") from exc
    if not result or any(x <= 0 for x in result):
        raise argparse.ArgumentTypeError("--sides must contain positive integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inference-only 768/896 pixel-budget sweep for the previous best "
            "Qwen3.5-9B epoch_1.5 LoRA checkpoint."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-csv", type=str, default="test.csv")
    parser.add_argument("--sample-submission", type=str, default="sample_submission.csv")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--training-module",
        type=str,
        default=DEFAULT_TRAINING_MODULE,
        help=(
            "Original successful training script module name without .py. "
            "Its prompt/dataset/collator/scoring functions are reused."
        ),
    )
    parser.add_argument(
        "--sides",
        type=parse_int_list,
        default=DEFAULT_SIDES,
        help="Inference max-pixel budget sides. Default: 768,896",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Default: value saved in the original run_config.json (normally 4).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Default: value saved in the original run_config.json (normally 2).",
    )
    parser.add_argument(
        "--check-all-images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        parser.error("--eval-batch-size must be > 0")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    return args


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        if default is not None:
            return default
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_path(path: Path, data_root: Path) -> Path:
    if path.is_absolute():
        return path
    candidate = Path.cwd() / path
    if candidate.exists():
        return candidate.resolve()
    return (data_root / path).resolve()


def print_info(
    *,
    gpu_name: str,
    model_id: str,
    checkpoint: Path,
    output_dir: Path,
    sides: tuple[int, ...],
    eval_batch_size: int,
    num_workers: int,
) -> None:
    console.print(
        Panel(
            "[bold cyan]Qwen3.5-9B · BEST epoch 1.5 High-Resolution Inference[/bold cyan]\n"
            "[white]Inference only · no training · same checkpoint · r768/r896[/white]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
    table = Table(title="Inference configuration", box=box.ROUNDED, border_style="green", show_header=False)
    table.add_column("Key", style="bold white", no_wrap=True)
    table.add_column("Value", style="cyan")
    table.add_row("GPU", gpu_name)
    table.add_row("Model", model_id)
    table.add_row("Checkpoint", str(checkpoint))
    table.add_row("Output root", str(output_dir))
    table.add_row("Pixel budgets", ", ".join(f"{s}²" for s in sides))
    table.add_row("Eval batch", str(eval_batch_size))
    table.add_row("Workers", str(num_workers))
    console.print(table)


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file, override=False)

    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    checkpoint_dir = resolve_path(args.checkpoint, args.data_root)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA GPU is required")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)

    # Reuse the successful run's exact inference semantics.
    training = importlib.import_module(args.training_module)

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    adapter_state = checkpoint_dir / "adapter_state.pt"
    if not adapter_state.is_file():
        raise FileNotFoundError(
            f"adapter_state.pt not found: {adapter_state}\n"
            "This script intentionally restores the exact raw PEFT state from the run."
        )

    run_config = load_json(args.output_dir / "run_config.json", default={})
    model_id = str(run_config.get("model_id", "Qwen/Qwen3.5-9B"))
    min_pixels = int(run_config.get("min_pixels", 256 * 256))
    lora_r = int(run_config.get("lora_r", 16))
    lora_alpha = int(run_config.get("lora_alpha", 32))
    lora_dropout = float(run_config.get("lora_dropout", 0.05))
    load_in_4bit = bool(run_config.get("load_in_4bit", False))
    eval_batch_size = int(
        args.eval_batch_size if args.eval_batch_size is not None else run_config.get("eval_batch_size", 4)
    )
    num_workers = int(
        args.num_workers if args.num_workers is not None else run_config.get("num_workers", 2)
    )

    print_info(
        gpu_name=gpu_name,
        model_id=model_id,
        checkpoint=checkpoint_dir,
        output_dir=args.output_dir,
        sides=args.sides,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )

    test_path = args.data_root / args.test_csv
    sample_path = args.data_root / args.sample_submission
    test_df = pd.read_csv(test_path)
    training.require_columns(test_df, training.TEST_COLUMNS, str(test_path))
    training.verify_image_paths(
        (test_df,),
        data_root=args.data_root,
        check_all=args.check_all_images,
    )

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }

    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        console.print("[yellow]Loading NF4 4-bit base from original run_config[/yellow]")
    else:
        console.print("[green]Loading BF16 base model[/green]")

    model = AutoModelForMultimodalLM.from_pretrained(model_id, **model_kwargs)
    model.config.use_cache = True

    # Load the standard PEFT checkpoint directly from adapter_model.safetensors.
    # This intentionally avoids adapter_state.pt because older/copied .pt files may be
    # Git-LFS pointer text or otherwise incompatible with the current torch.load parser.
    adapter_model_path = checkpoint_dir / "adapter_model.safetensors"
    adapter_config_path = checkpoint_dir / "adapter_config.json"
    if not adapter_model_path.is_file():
        raise FileNotFoundError(f"adapter_model.safetensors not found: {adapter_model_path}")
    if not adapter_config_path.is_file():
        raise FileNotFoundError(f"adapter_config.json not found: {adapter_config_path}")

    # Detect a common failure mode before PEFT emits a cryptic safetensors error.
    with adapter_model_path.open("rb") as fh:
        prefix = fh.read(64)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            "adapter_model.safetensors is only a Git LFS pointer, not the real weights. "
            "Run `git lfs pull` in the repository (or restore the real checkpoint file) "
            f"before inference: {adapter_model_path}"
        )

    console.print(
        f"[magenta]Loading PEFT adapter from safetensors[/magenta] · {adapter_model_path}"
    )
    model = PeftModel.from_pretrained(
        model,
        checkpoint_dir,
        is_trainable=False,
    )
    model.eval()
    model.config.use_cache = True
    torch.cuda.empty_cache()
    console.print("[green]✓ epoch_1.5 checkpoint restored[/green]")

    test_dataset = training.VQADataset(test_df, args.data_root, has_answer=False)

    reference_answer_ids: list[int] | None = None
    reference_ids: list[str] | None = None
    results: list[dict[str, Any]] = []
    total_started = time.time()

    for side in args.sides:
        pixel_budget = side * side
        console.rule(f"[bold magenta]r{side} · max_pixels={pixel_budget:,}[/bold magenta]")

        processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=min_pixels,
            max_pixels=pixel_budget,
            trust_remote_code=True,
        )
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        answer_ids = training.get_answer_token_ids(processor)
        if reference_answer_ids is None:
            reference_answer_ids = answer_ids
            console.print(f"[green]✓ Answer token IDs[/green] {answer_ids}")
        elif answer_ids != reference_answer_ids:
            raise RuntimeError(
                f"Answer-token IDs changed between processors: reference={reference_answer_ids}, r{side}={answer_ids}"
            )

        current_batch_size = eval_batch_size

        while True:
            test_loader = training.build_dataloader(
                test_dataset,
                processor,
                train=False,
                batch_size=current_batch_size,
                num_workers=num_workers,
            )
            inference_started = time.time()
            try:
                ids, probabilities, predictions = training.predict_choice_probabilities(
                    model=model,
                    processor=processor,
                    loader=test_loader,
                    device=device,
                    description=f"BEST epoch_1.5 · r{side} · bs{current_batch_size}",
                )
                inference_minutes = (time.time() - inference_started) / 60.0
                break
            except torch.cuda.OutOfMemoryError:
                if current_batch_size <= 1:
                    raise
                new_batch_size = max(1, current_batch_size // 2)
                console.print(
                    f"[yellow]⚠ CUDA OOM at r{side}: batch {current_batch_size} → {new_batch_size}; retrying[/yellow]"
                )
                current_batch_size = new_batch_size
                torch.cuda.empty_cache()

        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise RuntimeError(f"Test ID order changed at r{side}; refusing to save submission.")

        submission_df = training.build_submission_frame(ids, predictions, sample_path)

        submission_dir = args.output_dir / "submissions" / f"r{side}"
        probability_dir = args.output_dir / "probabilities" / f"r{side}"
        submission_dir.mkdir(parents=True, exist_ok=True)
        probability_dir.mkdir(parents=True, exist_ok=True)

        submission_path = submission_dir / "submission_epoch_1.5.csv"
        probability_path = probability_dir / "probabilities_epoch_1.5.csv"

        submission_df.to_csv(submission_path, index=False, encoding="utf-8-sig")
        training.save_probability_frame(probability_path, ids, probabilities, predictions)

        prediction_counts = {
            label: int(sum(pred == label for pred in predictions))
            for label in training.LABELS
        }

        result = {
            "checkpoint": str(checkpoint_dir),
            "epoch": 1.5,
            "inference_side": side,
            "max_pixels": pixel_budget,
            "batch_size": current_batch_size,
            "inference_minutes": inference_minutes,
            "prediction_counts": prediction_counts,
            "submission_path": str(submission_path),
            "probability_path": str(probability_path),
        }
        results.append(result)

        table = Table(title=f"r{side} completed", box=box.ROUNDED, border_style="green", show_header=False)
        table.add_column("Key", style="bold")
        table.add_column("Value")
        table.add_row("Submission", str(submission_path))
        table.add_row("Probabilities", str(probability_path))
        table.add_row("Batch", str(current_batch_size))
        table.add_row("Time", f"{inference_minutes:.1f} min")
        table.add_row("Pred counts", str(prediction_counts))
        console.print(table)

        del test_loader, processor
        torch.cuda.empty_cache()

    results_path = args.output_dir / "best_epoch15_highres_inference_results.json"
    save_json(results_path, results)
    total_minutes = (time.time() - total_started) / 60.0

    summary = Table(title="High-resolution submissions ready", box=box.ROUNDED, border_style="cyan")
    summary.add_column("Resolution", justify="right")
    summary.add_column("Submission")
    summary.add_column("Time", justify="right")
    for result in results:
        summary.add_row(
            f"r{result['inference_side']}",
            result["submission_path"],
            f"{result['inference_minutes']:.1f}m",
        )
    console.print(summary)
    console.print(
        Panel(
            "[bold green]Inference complete[/bold green]\n"
            "Original r640 submission is untouched.\n"
            f"Result metadata: [cyan]{results_path}[/cyan]\n"
            f"Total high-res inference time: [bold]{total_minutes:.1f} min[/bold]",
            border_style="green",
            box=box.ROUNDED,
        )
    )


if __name__ == "__main__":
    main()
