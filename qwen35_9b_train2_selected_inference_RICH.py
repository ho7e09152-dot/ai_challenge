"""Inference-only script for selected dense checkpoints from the FINAL Qwen3.5-9B run.

This script DOES NOT train anything.

It reuses the exact prompt / dataset / collator / answer-token scoring / submission logic
from ``train_qwen35_9b_train2_FINAL_RICH_4SUB.py`` and evaluates only the dense
checkpoints around the region that was not submitted automatically:

    1.10, 1.20, 1.30, 1.35, 1.40, 1.45 epochs

Default project layout
----------------------
/workspace/ai_challenge/
├── train_qwen35_9b_train2_FINAL_RICH_4SUB.py
├── qwen35_9b_train2_selected_inference_RICH.py
├── test.csv
├── sample_submission.csv
├── .env
└── outputs/qwen35_9b_train2_FINAL/
    ├── checkpoints.json
    ├── checkpoints/
    │   ├── epoch_1.10_update_XXXX/
    │   ├── ...
    │   └── epoch_1.45_update_XXXX/
    ├── submissions/r640/
    └── probabilities/r640/

Outputs are written to the SAME locations used by the training script:

    outputs/qwen35_9b_train2_FINAL/submissions/r640/
    outputs/qwen35_9b_train2_FINAL/probabilities/r640/

The existing ``submission_manifest.csv`` is updated in-place (new rows are appended,
then deduplicated by checkpoint/resolution). Existing submissions are not deleted.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
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
        "python-dotenv is required. Install it with: pip install -U python-dotenv"
    ) from exc

try:
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
except ImportError as exc:
    raise RuntimeError(
        "A recent Transformers build is required. Install it with:\n"
        'pip install -U "transformers @ git+https://github.com/huggingface/transformers.git"'
    ) from exc

from peft import LoraConfig, get_peft_model


console = Console()
DEFAULT_EPOCHS = (1.10, 1.20, 1.30, 1.35, 1.40, 1.45)
DEFAULT_OUTPUT_DIR = Path("outputs/qwen35_9b_train2_FINAL")


def parse_float_list(value: str) -> tuple[float, ...]:
    try:
        values = tuple(
            sorted({float(item.strip()) for item in value.split(",") if item.strip()})
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--epochs must be comma-separated numbers"
        ) from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("--epochs must contain positive values")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inference-only r640 submissions for selected dense checkpoints from "
            "the FINAL Qwen3.5-9B train2 run."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-csv", type=str, default="test.csv")
    parser.add_argument(
        "--sample-submission", type=str, default="sample_submission.csv"
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--training-module",
        type=str,
        default="train_qwen35_9b_train2_FINAL_RICH_4SUB",
        help=(
            "Module name of the training script, without .py. Its exact prompt, "
            "dataset, collator, checkpoint restore, and inference functions are reused."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=parse_float_list,
        default=DEFAULT_EPOCHS,
        help="Checkpoint epochs to infer. Default: 1.10,1.20,1.30,1.35,1.40,1.45",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Defaults to the value saved in run_config.json (normally 4).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Defaults to the value saved in run_config.json (normally 2).",
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


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def close_epoch(a: float, b: float, tol: float = 5e-4) -> bool:
    return abs(float(a) - float(b)) <= tol


def select_checkpoint_records(
    checkpoint_records: list[dict[str, Any]],
    requested_epochs: tuple[float, ...],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    missing: list[float] = []

    for target_epoch in requested_epochs:
        matches = []
        for record in checkpoint_records:
            record_epoch = float(
                record.get("requested_epoch", record.get("current_data_epoch", math.nan))
            )
            if close_epoch(record_epoch, target_epoch):
                matches.append(record)

        if len(matches) == 0:
            missing.append(target_epoch)
            continue
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple checkpoints matched epoch {target_epoch:.2f}: {matches}"
            )

        record = dict(matches[0])
        checkpoint_dir = Path(str(record["checkpoint_dir"]))
        if not checkpoint_dir.is_absolute():
            # Old/portable checkpoint metadata may contain a relative path.
            # Prefer it relative to the current project directory if needed.
            candidate = Path.cwd() / checkpoint_dir
            if candidate.exists():
                checkpoint_dir = candidate
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"Checkpoint directory for epoch {target_epoch:.2f} does not exist: "
                f"{checkpoint_dir}"
            )
        if not (checkpoint_dir / "adapter_state.pt").is_file():
            raise FileNotFoundError(
                f"adapter_state.pt missing for epoch {target_epoch:.2f}: {checkpoint_dir}"
            )

        record["checkpoint_dir"] = str(checkpoint_dir.resolve())
        selected.append(record)

    if missing:
        available = [
            float(r.get("requested_epoch", r.get("current_data_epoch", math.nan)))
            for r in checkpoint_records
        ]
        raise RuntimeError(
            "Requested checkpoint epoch(s) not found in checkpoints.json: "
            f"{[round(v, 3) for v in missing]}\n"
            f"Available: {[round(v, 3) for v in available]}"
        )

    return selected


def print_plan(
    selected: list[dict[str, Any]],
    output_dir: Path,
    model_id: str,
    eval_batch_size: int,
    max_pixels: int,
) -> None:
    console.print(
        Panel(
            "[bold]Qwen3.5-9B · Selected Checkpoint Inference[/bold]\n"
            "[cyan]Inference only[/cyan] · no training · r640 · same output directory",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    info = Table(box=box.ROUNDED, border_style="cyan", show_header=False)
    info.add_column("Key", style="bold")
    info.add_column("Value")
    info.add_row("GPU", torch.cuda.get_device_name(0))
    info.add_row("Model", model_id)
    info.add_row("Output", str(output_dir))
    info.add_row("Resolution", f"max_pixels={max_pixels:,} (640² budget)")
    info.add_row("Eval batch", str(eval_batch_size))
    console.print(info)

    table = Table(title="Selected checkpoints", box=box.ROUNDED, border_style="magenta")
    table.add_column("Epoch", justify="right")
    table.add_column("Update", justify="right")
    table.add_column("Checkpoint")
    table.add_column("Submission")
    for record in selected:
        epoch = float(record.get("requested_epoch", record["current_data_epoch"]))
        update = int(record["global_update"])
        tag = f"epoch_{epoch:.2f}_update_{update:04d}"
        table.add_row(
            f"{epoch:.2f}",
            f"{update:,}",
            Path(str(record["checkpoint_dir"])).name,
            f"submission_{tag}.csv",
        )
    console.print(table)


def append_manifest(
    manifest_path: Path,
    new_rows: list[dict[str, Any]],
) -> None:
    new_df = pd.DataFrame(new_rows)
    if manifest_path.is_file():
        old_df = pd.read_csv(manifest_path)
        combined = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df

    # Normalize the keys used for safe de-duplication. Keep the newest path/result.
    subset = [
        col
        for col in ("kind", "requested_epoch", "update", "inference_side")
        if col in combined.columns
    ]
    if subset:
        combined = combined.drop_duplicates(subset=subset, keep="last")

    if "requested_epoch" in combined.columns:
        combined = combined.sort_values(
            by=["inference_side", "requested_epoch"],
            na_position="last",
        ).reset_index(drop=True)

    combined.to_csv(manifest_path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file, override=False)

    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This inference configuration requires BF16-capable CUDA")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Import the actual training module so prompt/rendering/token-scoring stays identical.
    training = importlib.import_module(args.training_module)

    run_config_path = args.output_dir / "run_config.json"
    checkpoints_json_path = args.output_dir / "checkpoints.json"
    run_config = load_json(run_config_path)
    checkpoint_records = load_json(checkpoints_json_path)
    if not isinstance(checkpoint_records, list):
        raise TypeError(f"{checkpoints_json_path} must contain a JSON list")

    selected = select_checkpoint_records(checkpoint_records, args.epochs)

    model_id = str(run_config.get("model_id", "Qwen/Qwen3.5-9B"))
    min_pixels = int(run_config.get("min_pixels", 256 * 256))
    # Deliberately use the proven r640 inference only.
    max_pixels = 640 * 640
    eval_batch_size = int(
        args.eval_batch_size
        if args.eval_batch_size is not None
        else run_config.get("eval_batch_size", 4)
    )
    num_workers = int(
        args.num_workers
        if args.num_workers is not None
        else run_config.get("num_workers", 2)
    )

    lora_r = int(run_config.get("lora_r", 16))
    lora_alpha = int(run_config.get("lora_alpha", 32))
    lora_dropout = float(run_config.get("lora_dropout", 0.05))
    load_in_4bit = bool(run_config.get("load_in_4bit", False))

    print_plan(
        selected=selected,
        output_dir=args.output_dir,
        model_id=model_id,
        eval_batch_size=eval_batch_size,
        max_pixels=max_pixels,
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

    console.print("[cyan]Loading r640 processor...[/cyan]")
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=True,
    )
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

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

    console.print("[cyan]Loading base model...[/cyan]")
    model = AutoModelForMultimodalLM.from_pretrained(model_id, **model_kwargs)
    model.config.use_cache = True

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.eval()

    training_answer_ids = training.get_answer_token_ids(processor)
    console.print(
        f"[green]Answer token IDs verified:[/green] {training_answer_ids}"
    )

    test_dataset = training.VQADataset(test_df, args.data_root, has_answer=False)

    submissions_dir = args.output_dir / "submissions" / "r640"
    probabilities_dir = args.output_dir / "probabilities" / "r640"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    probabilities_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    started_all = time.time()

    # One loader is reused unless OOM forces a smaller batch size.
    current_batch_size = eval_batch_size
    test_loader = training.build_dataloader(
        test_dataset,
        processor,
        train=False,
        batch_size=current_batch_size,
        num_workers=num_workers,
    )

    for index, record in enumerate(selected, start=1):
        requested_epoch = float(
            record.get("requested_epoch", record["current_data_epoch"])
        )
        update = int(record["global_update"])
        checkpoint_dir = Path(str(record["checkpoint_dir"]))
        tag = training.checkpoint_name(requested_epoch, update)

        console.rule(
            f"[bold magenta]{index}/{len(selected)} · {tag}[/bold magenta]"
        )
        console.print(f"[yellow]Restoring[/yellow] {checkpoint_dir}")
        training.restore_checkpoint(model, checkpoint_dir)
        model.eval()
        model.config.use_cache = True
        torch.cuda.empty_cache()

        inference_started = time.time()
        while True:
            try:
                ids, probabilities, predictions = training.predict_choice_probabilities(
                    model=model,
                    processor=processor,
                    loader=test_loader,
                    device=torch.device("cuda:0"),
                    description=f"test {tag} r640 bs{current_batch_size}",
                )
                break
            except torch.cuda.OutOfMemoryError:
                if current_batch_size <= 1:
                    raise
                new_batch_size = max(1, current_batch_size // 2)
                console.print(
                    f"[yellow]CUDA OOM[/yellow] · batch {current_batch_size} → "
                    f"{new_batch_size}; restarting this checkpoint inference"
                )
                current_batch_size = new_batch_size
                torch.cuda.empty_cache()
                test_loader = training.build_dataloader(
                    test_dataset,
                    processor,
                    train=False,
                    batch_size=current_batch_size,
                    num_workers=num_workers,
                )

        inference_minutes = (time.time() - inference_started) / 60.0

        submission_df = training.build_submission_frame(ids, predictions, sample_path)
        submission_path = submissions_dir / f"submission_{tag}.csv"
        probability_path = probabilities_dir / f"probabilities_{tag}.csv"

        submission_df.to_csv(
            submission_path,
            index=False,
            encoding="utf-8-sig",
        )
        training.save_probability_frame(
            probability_path,
            ids,
            probabilities,
            predictions,
        )

        prediction_counts = {
            label: int(sum(pred == label for pred in predictions))
            for label in training.LABELS
        }

        result = {
            "requested_epoch": requested_epoch,
            "global_update": update,
            "current_data_epoch": float(record.get("current_data_epoch", requested_epoch)),
            "baseline_equivalent_epoch": float(
                record.get("baseline_equivalent_epoch", math.nan)
            ),
            "inference_side": 640,
            "inference_max_pixels": max_pixels,
            "inference_batch_size": current_batch_size,
            "submission_path": str(submission_path),
            "probability_path": str(probability_path),
            "inference_minutes": inference_minutes,
            "prediction_counts": prediction_counts,
        }
        results.append(result)
        manifest_rows.append(
            {
                "kind": "checkpoint",
                "requested_epoch": requested_epoch,
                "actual_epoch": float(record.get("current_data_epoch", requested_epoch)),
                "update": update,
                "inference_side": 640,
                "max_pixels": max_pixels,
                "inference_batch_size": current_batch_size,
                "submission_path": str(submission_path),
                "probability_path": str(probability_path),
            }
        )

        console.print(f"[bold green]✓ Submission[/bold green]  {submission_path}")
        console.print(f"[green]✓ Probabilities[/green] {probability_path}")
        console.print(
            f"[dim]Inference time {inference_minutes:.1f} min · "
            f"pred counts {prediction_counts}[/dim]"
        )

    results_path = args.output_dir / "selected_inference_results.json"
    save_json(results_path, results)

    manifest_path = args.output_dir / "submission_manifest.csv"
    append_manifest(manifest_path, manifest_rows)

    elapsed_minutes = (time.time() - started_all) / 60.0

    summary = Table(title="Completed submissions", box=box.ROUNDED, border_style="green")
    summary.add_column("Epoch", justify="right")
    summary.add_column("Update", justify="right")
    summary.add_column("File")
    summary.add_column("Time", justify="right")
    for result in results:
        summary.add_row(
            f"{result['requested_epoch']:.2f}",
            f"{result['global_update']:,}",
            Path(result["submission_path"]).name,
            f"{result['inference_minutes']:.1f}m",
        )
    console.print(summary)

    console.print(
        Panel(
            f"[bold green]Selected inference complete[/bold green]\n"
            f"Submissions: [cyan]{submissions_dir}[/cyan]\n"
            f"Manifest: [cyan]{manifest_path}[/cyan]\n"
            f"Total inference time: [bold]{elapsed_minutes:.1f} min[/bold]",
            border_style="green",
            box=box.ROUNDED,
        )
    )


if __name__ == "__main__":
    main()
