"""Qwen3.5-9B BF16 LoRA FINAL run for Korean multiple-choice VQA.

Terminal UI: Rich panels/tables + colored tqdm progress (training semantics unchanged).

Final strategy
--------------
Data:
  * train.csv: all original labeled rows
  * train_aug.csv: synthetic-only rows after exact-original removal
  * train2_aug.csv: all unique extra labeled QA rows (images under train2/)

Proven recipe kept unchanged:
  * Qwen3.5-9B
  * BF16 base + LoRA r=16, alpha=32, dropout=0.05, all-linear
  * assistant-only normal LM SFT
  * same English system prompt + Korean one-letter answer instruction
  * LR 3e-5, AdamW, cosine schedule, warmup 5%, effective batch 16
  * training max_pixels = 640 x 640
  * no validation split, no dynamic choice shuffle, no TTA

Final safeguards / improvements:
  * verify trainable LoRA dtype after PEFT creation; only if PEFT leaves a trainable
    adapter in BF16/FP16, upcast that trainable parameter to FP32 before AdamW.
  * assert that the first assistant target token used by training is exactly the same
    a/b/c/d token scored during inference.
  * recompute optimizer updates from the ACTUAL merged dataset, preserving a full
    2.0-epoch cosine schedule after train2 is added instead of freezing 1,270 steps.
  * save dense LoRA checkpoints, especially around 1.3~1.8 epochs.
  * automatically run test inference after training and write submission + probability
    CSVs for every checkpoint.
  * no choice-rotation TTA. Optional multi-resolution inference is enabled by default
    at 640/768/896 square-pixel budgets; training itself remains at 640.

Default dense checkpoint epochs:
  0.50, 0.75, 1.00, 1.10, 1.20, 1.30, 1.35, 1.40, 1.45,
  1.50, 1.55, 1.60, 1.65, 1.70, 1.75, 1.80, 1.90, 2.00

Outputs:
  * checkpoints/epoch_X.XX_update_XXXX/
  * submissions/r640|r768|r896/submission_epoch_X.XX_update_XXXX.csv
  * probabilities/r640|r768|r896/probabilities_epoch_X.XX_update_XXXX.csv
  * one soft-checkpoint ensemble per inference resolution (convenience only)
  * submission_manifest.csv, checkpoints.json, inference_results.json, run_config.json

Use --inference-sides 640 if you want only the proven 640 inference and want to save
inference time. The default 640,768,896 does NOT change training and does NOT use TTA.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError as exc:
    raise RuntimeError(
        "rich is required for the terminal UI. Install it with: pip install -U rich"
    ) from exc

try:
    from dotenv import load_dotenv
except ImportError as exc:
    raise RuntimeError(
        "python-dotenv is required. Install it with: pip install -U python-dotenv"
    ) from exc

try:
    from transformers import (
        AutoModelForMultimodalLM,
        AutoProcessor,
        BitsAndBytesConfig,
        get_cosine_schedule_with_warmup,
    )
except ImportError as exc:
    raise RuntimeError(
        "Qwen3.5 requires a recent Transformers build. Install it with:\n"
        'pip install -U "transformers @ '
        'git+https://github.com/huggingface/transformers.git"'
    ) from exc

import peft

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)


MODEL_ID = "Qwen/Qwen3.5-9B"
LABELS = ("a", "b", "c", "d")
TRAIN_COLUMNS = ("id", "path", "question", "a", "b", "c", "d", "answer")
TEST_COLUMNS = ("id", "path", "question", "a", "b", "c", "d")
SYSTEM_PROMPT = (
    "You are a visual multiple-choice question answering assistant. "
    "Inspect the image and answer using exactly one lowercase letter: "
    "a, b, c, or d. Do not explain your answer."
)


# Successful pre-train2 baseline reference only (for logging / comparison).
BASELINE_UPDATES_PER_EPOCH = 635
CONTENT_COLUMNS = ("path", "question", "a", "b", "c", "d", "answer")


console = Console(highlight=False)


def rich_banner() -> None:
    title = Text("Qwen3.5-9B · VQA Final Training", style="bold cyan")
    subtitle = Text(
        "BF16 LoRA · train + augmentation + train2 · dense checkpoints · auto submission",
        style="white",
    )
    body = Text.assemble(title, "\n", subtitle)
    console.print(
        Panel(
            body,
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def rich_kv_table(title: str, rows: list[tuple[str, str]], border_style: str = "cyan") -> None:
    table = Table(
        title=title,
        box=box.ROUNDED,
        border_style=border_style,
        header_style="bold",
        show_header=False,
        pad_edge=True,
    )
    table.add_column("Key", style="bold white", no_wrap=True)
    table.add_column("Value", style="cyan")
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)


def rich_notice(message: str, *, kind: str = "info") -> None:
    styles = {
        "info": ("cyan", "ℹ"),
        "success": ("green", "✓"),
        "warning": ("yellow", "⚠"),
        "error": ("red", "✗"),
        "checkpoint": ("magenta", "◆"),
    }
    style, icon = styles.get(kind, styles["info"])
    console.print(f"[{style}]{icon}[/{style}] {message}")


def parse_float_list(value: str) -> tuple[float, ...]:
    try:
        values = tuple(sorted({float(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "values must be comma-separated numbers"
        ) from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must contain positive numbers")
    return values


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "values must be comma-separated integers"
        ) from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must contain positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FINAL Qwen3.5-9B BF16 LoRA: proven recipe + train2, actual-dataset "
            "2-epoch schedule, dense checkpoints, immediate multi-checkpoint submissions."
        )
    )

    # Paths / data
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--train-csv", type=str, default="train.csv")
    parser.add_argument("--aug-csv", type=str, default="train_aug.csv")
    parser.add_argument("--extra-train-csv", type=str, default="train2_aug.csv")
    parser.add_argument("--extra-image-dir", type=str, default="train2")
    parser.add_argument("--test-csv", type=str, default="test.csv")
    parser.add_argument(
        "--sample-submission", type=str, default="sample_submission.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/qwen35_9b_train2_FINAL")
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))

    # Model / reproducibility
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=42)

    # Proven training hyperparameters
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Proven LoRA capacity
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Training image budget remains the proven 640^2.
    parser.add_argument("--min-pixels", type=int, default=256 * 256)
    parser.add_argument("--max-pixels", type=int, default=640 * 640)

    # Dense checkpointing by CURRENT-DATA epoch, not hard-coded old optimizer steps.
    parser.add_argument(
        "--checkpoint-epochs",
        type=parse_float_list,
        default=parse_float_list(
            "0.50,0.75,1.00,1.10,1.20,1.30,1.35,1.40,1.45,"
            "1.50,1.55,1.60,1.65,1.70,1.75,1.80,1.90,2.00"
        ),
        help=(
            "Comma-separated current-dataset epochs at which LoRA adapters are saved. "
            "The final --epochs point is always added automatically."
        ),
    )
    parser.add_argument(
        "--baseline-updates-per-epoch",
        type=int,
        default=BASELINE_UPDATES_PER_EPOCH,
        help="Reference only: 635 updates/epoch for the previous 10,146-row best run.",
    )

    # Inference resolution sweep. These are square-pixel budgets, NOT forced resize sides.
    # No TTA / no choice rotation is performed.
    parser.add_argument(
        "--inference-sides",
        type=parse_int_list,
        default=parse_int_list("640,768,896"),
        help=(
            "Comma-separated square pixel-budget sides for inference. "
            "Default 640,768,896. Use 640 for fastest proven-only inference."
        ),
    )

    # Runtime / switches
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Leave disabled: BF16 LoRA is the proven A100 80GB configuration.",
    )
    parser.add_argument(
        "--use-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use synthetic-only rows extracted from train_aug.csv.",
    )
    parser.add_argument(
        "--use-extra-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include train2_aug.csv; its images are resolved under train2/ if needed.",
    )
    parser.add_argument(
        "--check-all-images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--data-check-only",
        action="store_true",
        help=(
            "Validate/merge train + train_aug + train2, verify image paths and exit "
            "before loading the model."
        ),
    )
    parser.add_argument(
        "--run-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically generate submissions for every saved checkpoint after training.",
    )

    # W&B
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable W&B when WANDB_API_KEY is available, or when WANDB_MODE=offline. "
            "If neither is configured, training continues with W&B disabled."
        ),
    )
    parser.add_argument("--wandb-project", type=str, default="qwen35-vqa")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be > 0")
    if args.baseline_updates_per_epoch <= 0:
        parser.error("--baseline-updates-per-epoch must be > 0")
    if args.grad_accum_steps <= 0:
        parser.error("--grad-accum-steps must be > 0")
    if args.train_batch_size <= 0 or args.eval_batch_size <= 0:
        parser.error("batch sizes must be > 0")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("--warmup-ratio must be in [0, 1)")
    if args.min_pixels <= 0 or args.max_pixels <= 0:
        parser.error("pixel limits must be > 0")
    if args.min_pixels > args.max_pixels:
        parser.error("--min-pixels cannot exceed --max-pixels")

    invalid_ckpt_epochs = [ep for ep in args.checkpoint_epochs if ep > args.epochs + 1e-9]
    if invalid_ckpt_epochs:
        parser.error(
            f"checkpoint epochs cannot exceed --epochs={args.epochs}: {invalid_ckpt_epochs}"
        )

    # Always preserve the exact final state.
    if not any(abs(ep - args.epochs) < 1e-9 for ep in args.checkpoint_epochs):
        args.checkpoint_epochs = tuple(sorted((*args.checkpoint_epochs, args.epochs)))

    return args


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def normalize_answer_column(df: pd.DataFrame, name: str) -> pd.DataFrame:
    result = df.copy()
    result["answer"] = result["answer"].astype(str).str.strip().str.lower()
    invalid = ~result["answer"].isin(LABELS)
    if invalid.any():
        examples = result.loc[invalid, ["id", "answer"]].head().to_dict("records")
        raise ValueError(f"{name} contains invalid answers: {examples}")
    return result


def exact_row_keys(df: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(df[list(TRAIN_COLUMNS)].astype(str))


def content_row_keys(df: pd.DataFrame) -> pd.MultiIndex:
    """Content identity excluding `id`, which may restart in independently-made data."""
    return pd.MultiIndex.from_frame(df[list(CONTENT_COLUMNS)].astype(str))


def resolve_extra_image_paths(
    frame: pd.DataFrame,
    data_root: Path,
    extra_image_dir: str,
) -> pd.DataFrame:
    """Resolve train2_aug.csv image paths without changing normal train/test behavior.

    Accepted forms for an image physically located at `train2/foo.jpg`:
      * `train2/foo.jpg`
      * `foo.jpg`
      * a relative nested path such as `subdir/foo.jpg`

    Resolution order is conservative: use the CSV path as-is if it already exists,
    otherwise try it under `train2/`, then try its basename under `train2/`.
    """
    result = frame.copy()
    resolved: list[str] = []
    missing: list[str] = []

    extra_root = data_root / extra_image_dir

    for raw_value in result["path"].astype(str):
        raw = Path(raw_value)
        candidates: list[Path] = []

        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(data_root / raw)
            candidates.append(extra_root / raw)
            candidates.append(extra_root / raw.name)

        chosen: Path | None = None
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if candidate.is_file():
                chosen = candidate
                break

        if chosen is None:
            missing.append(raw_value)
            resolved.append(raw_value)
            continue

        try:
            normalized = chosen.relative_to(data_root)
            resolved.append(normalized.as_posix())
        except ValueError:
            # Absolute paths outside data_root are still supported by VQADataset below.
            resolved.append(str(chosen))

    if missing:
        raise FileNotFoundError(
            "Could not resolve train2 image paths (first 10): " + str(missing[:10])
        )

    result["path"] = resolved
    return result


def _answer_distribution(df: pd.DataFrame) -> dict[str, int]:
    counts = df["answer"].astype(str).str.lower().value_counts()
    return {label: int(counts.get(label, 0)) for label in LABELS}


def build_full_train(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the final training frame while keeping the proven training recipe.

    The only intended experiment variable is data:
      1) all train.csv originals
      2) synthetic-only rows from train_aug.csv
      3) every unique labeled row from train2_aug.csv

    No validation split is created.
    """
    train_path = args.data_root / args.train_csv
    base_df = pd.read_csv(train_path)
    require_columns(base_df, TRAIN_COLUMNS, str(train_path))
    base_df = normalize_answer_column(base_df, str(train_path)).reset_index(drop=True)

    if base_df["id"].duplicated().any():
        duplicated = base_df.loc[base_df["id"].duplicated(), "id"].head().tolist()
        raise ValueError(
            "train.csv must have one original row per image ID. "
            f"Example duplicated IDs: {duplicated}"
        )

    base_count = len(base_df)
    pieces: list[pd.DataFrame] = [base_df]
    synthetic_df = base_df.iloc[0:0].copy()

    # Existing augmentation: preserve the exact successful behavior by removing
    # rows that are exact copies of train.csv and keeping the synthetic siblings.
    if args.use_augmentation:
        aug_path = args.data_root / args.aug_csv
        if not aug_path.exists():
            raise FileNotFoundError(
                f"augmentation requested but file does not exist: {aug_path}"
            )

        aug_df = pd.read_csv(aug_path)
        require_columns(aug_df, TRAIN_COLUMNS, str(aug_path))
        aug_df = normalize_answer_column(aug_df, str(aug_path)).reset_index(drop=True)

        known_ids = set(base_df["id"].astype(str))
        unknown_ids = sorted(set(aug_df["id"].astype(str)) - known_ids)
        if unknown_ids:
            raise ValueError(
                "train_aug.csv contains IDs not found in train.csv: "
                f"{unknown_ids[:10]}"
            )

        original_keys = exact_row_keys(base_df)
        aug_keys = exact_row_keys(aug_df)
        synthetic_df = aug_df.loc[~aug_keys.isin(original_keys)].copy()
        synthetic_df = synthetic_df.drop_duplicates(
            subset=list(CONTENT_COLUMNS)
        ).reset_index(drop=True)
        pieces.append(synthetic_df)

    # New data: all unique QA rows are valid training examples. train2 IDs may
    # overlap with train.csv IDs because IDs are metadata only during training.
    extra_df = base_df.iloc[0:0].copy()
    extra_raw_count = 0
    if args.use_extra_data:
        extra_path = args.data_root / args.extra_train_csv
        if not extra_path.exists():
            raise FileNotFoundError(
                f"extra data is enabled but CSV does not exist: {extra_path}"
            )

        extra_df = pd.read_csv(extra_path)
        require_columns(extra_df, TRAIN_COLUMNS, str(extra_path))
        extra_df = normalize_answer_column(extra_df, str(extra_path))
        extra_df = resolve_extra_image_paths(
            extra_df,
            data_root=args.data_root,
            extra_image_dir=args.extra_image_dir,
        )
        extra_raw_count = len(extra_df)
        extra_df = extra_df.drop_duplicates(
            subset=list(CONTENT_COLUMNS)
        ).reset_index(drop=True)
        pieces.append(extra_df)

    merged = pd.concat(pieces, ignore_index=True)
    before_cross_dedup = len(merged)

    # Remove only exact content duplicates across sources. This is a data-safety
    # improvement, not a modeling change. Different paraphrases remain separate rows.
    train_df = merged.drop_duplicates(
        subset=list(CONTENT_COLUMNS)
    ).reset_index(drop=True)
    cross_duplicates_removed = before_cross_dedup - len(train_df)

    train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    stats: dict[str, Any] = {
        "original_rows": base_count,
        "synthetic_rows": len(synthetic_df),
        "extra_rows_raw": extra_raw_count,
        "extra_rows_unique": len(extra_df),
        "extra_unique_images": int(extra_df["path"].astype(str).nunique()) if len(extra_df) else 0,
        "cross_source_duplicates_removed": cross_duplicates_removed,
        "total_train_rows": len(train_df),
        "total_unique_image_paths": int(train_df["path"].astype(str).nunique()),
        "answer_distribution_original": _answer_distribution(base_df),
        "answer_distribution_synthetic": _answer_distribution(synthetic_df) if len(synthetic_df) else {},
        "answer_distribution_extra": _answer_distribution(extra_df) if len(extra_df) else {},
        "answer_distribution_total": _answer_distribution(train_df),
    }

    data_table = Table(
        title="Training Data Summary · NO validation split",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
    )
    data_table.add_column("Source / Metric", style="bold white")
    data_table.add_column("Rows", justify="right", style="green")
    data_table.add_column("Details", style="dim")
    data_table.add_row("train.csv", f"{base_count:,}", "original labeled rows")
    data_table.add_row(
        "train_aug.csv",
        f"{len(synthetic_df):,}",
        "synthetic-only after exact-original removal",
    )
    if args.use_extra_data:
        data_table.add_row(
            "train2_aug.csv",
            f"{len(extra_df):,}",
            f"{extra_raw_count:,} raw · {stats['extra_unique_images']:,} unique images",
        )
    data_table.add_section()
    data_table.add_row("cross-source dedup", f"-{cross_duplicates_removed:,}", "exact content duplicates")
    data_table.add_row("TOTAL", f"{len(train_df):,}", f"{stats['total_unique_image_paths']:,} unique image paths")
    console.print(data_table)

    dist_table = Table(
        title="Answer Distribution",
        box=box.SIMPLE_HEAVY,
        border_style="blue",
        header_style="bold blue",
    )
    dist_table.add_column("Dataset", style="bold white")
    for label in LABELS:
        dist_table.add_column(label, justify="right")
    for name, distribution in (
        ("original", stats["answer_distribution_original"]),
        ("train2", stats["answer_distribution_extra"] if args.use_extra_data else {}),
        ("total", stats["answer_distribution_total"]),
    ):
        if name == "train2" and not args.use_extra_data:
            continue
        dist_table.add_row(name, *(f"{int(distribution.get(label, 0)):,}" for label in LABELS))
    console.print(dist_table)

    return train_df, stats


def verify_image_paths(
    frames: Iterable[pd.DataFrame], data_root: Path, check_all: bool
) -> None:
    missing: list[str] = []
    checked: set[str] = set()

    for frame in frames:
        paths = frame["path"].astype(str)
        if not check_all:
            paths = paths.head(16)

        for relative in paths:
            if relative in checked:
                continue
            checked.add(relative)
            raw_path = Path(relative)
            image_path = raw_path if raw_path.is_absolute() else data_root / raw_path
            if not image_path.is_file():
                missing.append(relative)
                if len(missing) >= 10:
                    break

        if len(missing) >= 10:
            break

    if missing:
        raise FileNotFoundError(f"missing image paths (first 10): {missing}")

    rich_notice(f"Verified {len(checked):,} unique image paths", kind="success")


def build_mc_prompt(row: dict[str, Any]) -> str:
    return (
        f"{str(row['question']).strip()}\n"
        f"(a) {str(row['a']).strip()}\n"
        f"(b) {str(row['b']).strip()}\n"
        f"(c) {str(row['c']).strip()}\n"
        f"(d) {str(row['d']).strip()}\n\n"
        "정답을 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."
    )


class VQADataset(Dataset):
    def __init__(self, frame: pd.DataFrame, data_root: Path, has_answer: bool):
        self.records = frame.to_dict("records")
        self.data_root = data_root
        self.has_answer = has_answer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        raw_image_path = Path(str(row["path"]))
        image_path = (
            raw_image_path if raw_image_path.is_absolute()
            else self.data_root / raw_image_path
        )

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB").copy()

        sample = {
            "id": str(row["id"]),
            "image": rgb_image,
            "prompt": build_mc_prompt(row),
        }
        if self.has_answer:
            sample["answer"] = str(row["answer"]).strip().lower()
        return sample


def system_message() -> dict[str, Any]:
    return {
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
    }


def user_message(prompt: str, image: Image.Image) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }


def assistant_message(answer: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": answer}],
    }


def render_chat(
    processor: Any,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool,
) -> str:
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


@dataclass
class VQACollator:
    processor: Any
    train: bool

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        # Right-padding simplifies exact prompt-prefix masking during training.
        # Left-padding makes the final prompt token sit at index -1 for inference,
        # which lets us score the next-token a/b/c/d logits directly.
        self.processor.tokenizer.padding_side = "right" if self.train else "left"

        images = [sample["image"] for sample in samples]
        prompt_texts: list[str] = []
        full_texts: list[str] = []

        for sample in samples:
            prompt_messages = [
                system_message(),
                user_message(sample["prompt"], sample["image"]),
            ]
            prompt_texts.append(
                render_chat(
                    self.processor,
                    prompt_messages,
                    add_generation_prompt=True,
                )
            )

            if self.train:
                full_messages = prompt_messages + [assistant_message(sample["answer"])]
                full_texts.append(
                    render_chat(
                        self.processor,
                        full_messages,
                        add_generation_prompt=False,
                    )
                )

        if not self.train:
            encoded = self.processor(
                text=prompt_texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
            return {
                "model_inputs": encoded,
                "ids": [sample["id"] for sample in samples],
            }

        encoded = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        prompt_encoded = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100

        for index in range(len(samples)):
            prompt_len = int(prompt_encoded["attention_mask"][index].sum())
            prompt_ids = prompt_encoded["input_ids"][index, :prompt_len]
            full_prefix = encoded["input_ids"][index, :prompt_len]

            if not torch.equal(prompt_ids, full_prefix):
                raise RuntimeError(
                    "chat-template prefix mismatch; update masking logic for "
                    "the installed Transformers version"
                )

            labels[index, :prompt_len] = -100

            # Critical correctness guard: the first token actually trained as the
            # assistant answer must be the exact same single token scored at inference.
            target_positions = torch.nonzero(labels[index] != -100, as_tuple=False)
            if target_positions.numel() == 0:
                raise RuntimeError(
                    f"sample {samples[index]['id']!r} has no unmasked assistant target token"
                )
            first_target_pos = int(target_positions[0].item())
            actual_target_id = int(labels[index, first_target_pos].item())
            expected_ids = self.processor.tokenizer.encode(
                samples[index]["answer"], add_special_tokens=False
            )
            if len(expected_ids) != 1:
                raise RuntimeError(
                    f"answer {samples[index]['answer']!r} is not one tokenizer token: "
                    f"{expected_ids}"
                )
            expected_target_id = int(expected_ids[0])
            if actual_target_id != expected_target_id:
                actual_text = self.processor.tokenizer.decode([actual_target_id])
                expected_text = self.processor.tokenizer.decode([expected_target_id])
                raise RuntimeError(
                    "TRAIN/INFERENCE ANSWER TOKEN MISMATCH: "
                    f"id={samples[index]['id']!r}, answer={samples[index]['answer']!r}, "
                    f"trained_token={actual_target_id}({actual_text!r}), "
                    f"scored_token={expected_target_id}({expected_text!r}). "
                    "Do not train until chat-template/token scoring is aligned."
                )

        if not torch.any(labels != -100):
            raise RuntimeError("all labels were masked")

        encoded["labels"] = labels
        return encoded


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


def build_dataloader(
    dataset: Dataset,
    processor: Any,
    train: bool,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        collate_fn=VQACollator(processor=processor, train=train),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def cpu_adapter_state(model: Any) -> dict[str, torch.Tensor]:
    state = get_peft_model_state_dict(model)
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def save_checkpoint(
    model: Any,
    checkpoint_dir: Path,
    epoch_progress: float,
    global_update: int,
    learning_rate: float,
    elapsed_minutes: float,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Standard PEFT checkpoint for normal reuse.
    model.save_pretrained(checkpoint_dir, safe_serialization=True)

    # Raw PEFT state makes fast in-process switching between checkpoints robust.
    raw_state_path = checkpoint_dir / "adapter_state.pt"
    torch.save(cpu_adapter_state(model), raw_state_path)

    metadata = {
        "epoch_progress": epoch_progress,
        "global_update": global_update,
        "learning_rate": learning_rate,
        "elapsed_minutes": elapsed_minutes,
    }
    save_json(checkpoint_dir / "checkpoint_meta.json", metadata)
    rich_notice(f"Saved checkpoint → [bold]{checkpoint_dir}[/bold]", kind="checkpoint")


def load_raw_adapter_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility with older PyTorch versions.
        return torch.load(path, map_location="cpu")


def restore_checkpoint(model: Any, checkpoint_dir: Path) -> None:
    raw_state_path = checkpoint_dir / "adapter_state.pt"
    if not raw_state_path.exists():
        raise FileNotFoundError(f"missing adapter state: {raw_state_path}")

    state = load_raw_adapter_state(raw_state_path)
    set_result = set_peft_model_state_dict(model, state)

    unexpected = getattr(set_result, "unexpected_keys", None)
    if unexpected:
        raise RuntimeError(
            f"unexpected adapter keys while restoring {checkpoint_dir}: {unexpected}"
        )


def get_answer_token_ids(processor: Any) -> list[int]:
    token_ids: list[int] = []
    for label in LABELS:
        encoded = processor.tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(
                f"answer label {label!r} is not a single tokenizer token: {encoded}"
            )
        token_ids.append(encoded[0])
    return token_ids



def ensure_trainable_parameters_fp32(model: Any) -> dict[str, Any]:
    """Verify LoRA/trainable precision and guard against low-precision adapter updates.

    Modern PEFT normally autocasts FP16/BF16 adapter weights to FP32. We preserve that
    behavior. If the installed PEFT leaves any *trainable* parameter in FP16/BF16, only
    that trainable parameter is upcast to FP32 before the optimizer is created.
    Frozen BF16 base weights are never changed.
    """
    before: dict[str, int] = {}
    for parameter in model.parameters():
        if parameter.requires_grad:
            key = str(parameter.dtype)
            before[key] = before.get(key, 0) + parameter.numel()

    non_fp32 = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.dtype != torch.float32
    ]

    cast_count = 0
    cast_numel = 0
    if non_fp32:
        rich_notice(
            "PEFT left some trainable parameters below FP32; "
            "upcasting trainable parameters only before AdamW.",
            kind="warning",
        )
        for parameter in non_fp32:
            cast_count += 1
            cast_numel += parameter.numel()
            parameter.data = parameter.data.float()

    after: dict[str, int] = {}
    for parameter in model.parameters():
        if parameter.requires_grad:
            key = str(parameter.dtype)
            after[key] = after.get(key, 0) + parameter.numel()

    if any(
        parameter.requires_grad and parameter.dtype != torch.float32
        for parameter in model.parameters()
    ):
        raise RuntimeError("trainable LoRA parameters are still not FP32 after precision guard")

    info = {
        "peft_version": getattr(peft, "__version__", "unknown"),
        "trainable_dtype_numel_before": before,
        "trainable_dtype_numel_after": after,
        "upcast_parameter_tensors": cast_count,
        "upcast_parameter_numel": cast_numel,
    }
    rich_kv_table(
        "LoRA Precision Guard",
        [
            ("PEFT version", str(info["peft_version"])),
            ("before", str(before)),
            ("after", str(after)),
            ("upcast tensors", f"{cast_count:,}"),
            ("upcast parameters", f"{cast_numel:,}"),
        ],
        border_style="magenta",
    )
    if cast_count == 0:
        rich_notice("All trainable LoRA parameters were already FP32", kind="success")
    else:
        rich_notice(
            f"Upcast {cast_count} trainable tensors / {cast_numel:,} parameters to FP32",
            kind="warning",
        )
    return info


def build_checkpoint_targets_from_epochs(
    checkpoint_epochs: tuple[float, ...],
    updates_per_epoch: int,
    total_updates: int,
) -> dict[int, float]:
    """Map requested current-dataset epoch positions to exact optimizer updates."""
    if updates_per_epoch <= 0 or total_updates <= 0:
        raise ValueError("updates_per_epoch and total_updates must be > 0")

    targets: dict[int, float] = {}
    for requested_epoch in checkpoint_epochs:
        update = max(1, int(round(requested_epoch * updates_per_epoch)))
        update = min(update, total_updates)
        if update in targets and abs(targets[update] - requested_epoch) > 1e-9:
            raise ValueError(
                "checkpoint epochs are too dense for the current optimizer-step count: "
                f"{targets[update]} and {requested_epoch} both map to update {update}"
            )
        targets[update] = requested_epoch

    # Always preserve final optimizer state.
    targets[total_updates] = total_updates / updates_per_epoch
    return dict(sorted(targets.items()))


def checkpoint_name(epoch_progress: float, update: int) -> str:
    return f"epoch_{epoch_progress:.2f}_update_{update:04d}"


@torch.inference_mode()
def predict_choice_probabilities(
    model: Any,
    processor: Any,
    loader: DataLoader,
    device: torch.device,
    description: str,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Return IDs, normalized a/b/c/d probabilities, and chosen labels.

    Because inference batches are left-padded, index -1 corresponds to the last
    prompt token for every sample. The logits at that position are therefore the
    next-token logits. We select the four answer-token logits and softmax only
    over those candidates.
    """

    model.eval()
    previous_use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = True

    answer_token_ids = get_answer_token_ids(processor)
    all_ids: list[str] = []
    probability_chunks: list[np.ndarray] = []

    for batch in tqdm(loader, desc=description, leave=False):
        ids = batch["ids"]
        inputs = move_to_device(batch["model_inputs"], device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**inputs)

        next_token_logits = outputs.logits[:, -1, answer_token_ids].float()
        probabilities = torch.softmax(next_token_logits, dim=-1)

        all_ids.extend(ids)
        probability_chunks.append(probabilities.cpu().numpy())

    model.config.use_cache = previous_use_cache

    if probability_chunks:
        all_probabilities = np.concatenate(probability_chunks, axis=0)
    else:
        all_probabilities = np.empty((0, len(LABELS)), dtype=np.float32)

    prediction_indices = all_probabilities.argmax(axis=1)
    predictions = [LABELS[index] for index in prediction_indices]
    return all_ids, all_probabilities, predictions


def build_submission_frame(
    ids: list[str],
    predictions: list[str],
    sample_path: Path,
) -> pd.DataFrame:
    prediction_df = pd.DataFrame(
        {
            "_id_key": pd.Series(ids, dtype="string"),
            "answer": predictions,
        }
    )

    if prediction_df["_id_key"].duplicated().any():
        raise RuntimeError("test predictions contain duplicated IDs")

    if not sample_path.exists():
        return pd.DataFrame({"id": ids, "answer": predictions})

    sample_df = pd.read_csv(sample_path)
    require_columns(sample_df, ("id", "answer"), str(sample_path))

    sample_order = sample_df[["id"]].copy()
    sample_order["_id_key"] = sample_order["id"].astype(str)
    prediction_df["_id_key"] = prediction_df["_id_key"].astype(str)

    submission_df = sample_order.merge(
        prediction_df,
        on="_id_key",
        how="left",
        validate="one_to_one",
    )

    if submission_df["answer"].isna().any():
        missing = submission_df.loc[
            submission_df["answer"].isna(), "id"
        ].head().tolist()
        raise RuntimeError(
            "some sample-submission IDs have no prediction. "
            f"Examples: {missing}"
        )

    return submission_df[["id", "answer"]]


def save_probability_frame(
    path: Path,
    ids: list[str],
    probabilities: np.ndarray,
    predictions: list[str],
) -> None:
    frame = pd.DataFrame(
        {
            "id": ids,
            "p_a": probabilities[:, 0],
            "p_b": probabilities[:, 1],
            "p_c": probabilities[:, 2],
            "p_d": probabilities[:, 3],
            "answer": predictions,
        }
    )
    frame.to_csv(path, index=False, encoding="utf-8-sig")



def init_wandb(
    args: argparse.Namespace,
    run_config: dict[str, Any],
) -> Any | None:
    if not args.wandb:
        rich_notice("W&B disabled by --no-wandb", kind="info")
        return None

    mode = os.getenv("WANDB_MODE", "online").strip().lower() or "online"
    api_key = os.getenv("WANDB_API_KEY", "").strip()

    if mode == "disabled":
        rich_notice("W&B disabled because WANDB_MODE=disabled", kind="info")
        return None

    if mode == "online" and not api_key:
        rich_notice(
            "W&B online logging requested, but WANDB_API_KEY is not set. "
            "Continuing with W&B disabled. Set the key in .env or use WANDB_MODE=offline.",
            kind="warning",
        )
        return None

    try:
        import wandb
    except ImportError:
        rich_notice(
            "W&B requested but wandb is not installed. Continuing without W&B. "
            "Install with: pip install -U wandb",
            kind="warning",
        )
        return None

    project = os.getenv("WANDB_PROJECT", "").strip() or args.wandb_project
    entity = os.getenv("WANDB_ENTITY", "").strip() or None
    run_name = (
        os.getenv("WANDB_RUN_NAME", "").strip() or args.wandb_run_name or None
    )

    tags_raw = os.getenv("WANDB_TAGS", "").strip()
    tags = [item.strip() for item in tags_raw.split(",") if item.strip()] or None

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            mode=mode,
            config=run_config,
            tags=tags,
        )
    except Exception as exc:
        rich_notice(f"W&B initialization failed ({exc!r}); continuing without W&B", kind="warning")
        return None

    rich_notice(
        f"W&B enabled · project={project!r} · mode={mode!r} · "
        f"run={getattr(run, 'name', None)!r}",
        kind="success",
    )
    return run


def wandb_log(run: Any | None, values: dict[str, Any], step: int | None = None) -> None:
    if run is None:
        return
    run.log(values, step=step)


def main() -> None:
    args = parse_args()

    # Load .env before reading any W&B environment variables.
    load_dotenv(args.env_file, override=False)

    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_root = args.output_dir / "checkpoints"
    submissions_root = args.output_dir / "submissions"
    probabilities_root = args.output_dir / "probabilities"
    processor_dir = args.output_dir / "processor"

    checkpoints_root.mkdir(parents=True, exist_ok=True)
    submissions_root.mkdir(parents=True, exist_ok=True)
    probabilities_root.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires a BF16-capable GPU")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 2**30

    rich_banner()
    rich_kv_table(
        "Runtime",
        [
            ("GPU", gpu_name),
            ("VRAM", f"{total_vram:.1f} GiB"),
            ("Model", args.model_id),
            ("Output", str(args.output_dir)),
            ("Seed", str(args.seed)),
        ],
        border_style="green",
    )

    train_df, data_stats = build_full_train(args)

    test_path = args.data_root / args.test_csv
    test_df = pd.read_csv(test_path)
    require_columns(test_df, TEST_COLUMNS, str(test_path))

    verify_image_paths(
        (train_df, test_df),
        data_root=args.data_root,
        check_all=args.check_all_images,
    )

    if args.data_check_only:
        rich_notice("Data check complete · exiting before model load/training", kind="success")
        return

    rich_notice(f"Loading processor · [bold]{args.model_id}[/bold]", kind="info")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        trust_remote_code=True,
    )
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    processor.save_pretrained(processor_dir)

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }

    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        rich_notice("Loading NF4 4-bit base model", kind="warning")
    else:
        rich_notice("Loading BF16 base model · proven configuration", kind="success")

    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id,
        **model_kwargs,
    )
    model.config.use_cache = False

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,
        )

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # IMPORTANT: exactly the LoRA configuration from the successful run.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    precision_info = ensure_trainable_parameters_fp32(model)
    model.print_trainable_parameters()

    # IMPORTANT: no dynamic choice shuffling; normal assistant-only LM SFT is kept.
    train_dataset = VQADataset(train_df, args.data_root, has_answer=True)
    test_dataset = VQADataset(test_df, args.data_root, has_answer=False)

    train_loader = build_dataloader(
        train_dataset,
        processor,
        train=True,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
    )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # FINAL scheduling rule: preserve a full cosine schedule over the ACTUAL merged
    # dataset. With train2 added, both epoch exposure and the relative cosine position
    # of the old 1.5/2.0 best point are preserved (1.5 epoch remains 75% of schedule).
    current_updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_updates = max(1, int(round(args.epochs * current_updates_per_epoch)))
    warmup_steps = int(total_updates * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    checkpoint_target_map = build_checkpoint_targets_from_epochs(
        checkpoint_epochs=args.checkpoint_epochs,
        updates_per_epoch=current_updates_per_epoch,
        total_updates=total_updates,
    )
    checkpoint_targets = set(checkpoint_target_map)

    effective_batch = args.train_batch_size * args.grad_accum_steps
    rich_kv_table(
        "Training Schedule · actual merged dataset",
        [
            ("training rows", f"{len(train_df):,}"),
            ("micro-batches / epoch", f"{len(train_loader):,}"),
            ("optimizer updates / epoch", f"{current_updates_per_epoch:,}"),
            ("effective batch", str(effective_batch)),
            ("epochs", f"{args.epochs:.3f}"),
            ("total updates", f"{total_updates:,}"),
            ("warmup updates", f"{warmup_steps:,}"),
            ("learning rate", f"{args.learning_rate:.2e}"),
            ("train max pixels", f"{args.max_pixels:,} ({int(math.sqrt(args.max_pixels))}² budget)"),
            ("old baseline updates / ep", f"{args.baseline_updates_per_epoch:,} · reference only"),
        ],
        border_style="cyan",
    )

    checkpoint_table = Table(
        title=f"Dense Checkpoint Plan · {len(checkpoint_target_map)} checkpoints",
        box=box.ROUNDED,
        border_style="magenta",
        header_style="bold magenta",
    )
    checkpoint_table.add_column("Epoch", justify="right")
    checkpoint_table.add_column("Update", justify="right")
    checkpoint_table.add_column("Progress", justify="right")
    checkpoint_table.add_column("Old-step equiv.", justify="right", style="dim")
    for update, requested_epoch in checkpoint_target_map.items():
        current_ep = update / current_updates_per_epoch
        old_step_equivalent = update / args.baseline_updates_per_epoch
        checkpoint_table.add_row(
            f"{requested_epoch:.2f}",
            f"{update:,}",
            f"{update / total_updates:.1%}",
            f"{old_step_equivalent:.3f}",
        )
    console.print(checkpoint_table)

    if effective_batch != 16:
        rich_notice(
            "Effective batch size differs from the successful baseline (16); "
            "update-count equivalence is no longer exact.",
            kind="warning",
        )

    run_config = vars(args).copy()
    run_config["data_root"] = str(args.data_root)
    run_config["output_dir"] = str(args.output_dir)
    run_config["env_file"] = str(args.env_file)
    run_config["checkpoint_epochs"] = list(args.checkpoint_epochs)
    run_config["checkpoint_target_map"] = {str(k): v for k, v in checkpoint_target_map.items()}
    run_config["gpu"] = gpu_name
    run_config["vram_gib"] = total_vram
    run_config["current_updates_per_epoch"] = current_updates_per_epoch
    run_config["baseline_updates_per_epoch"] = args.baseline_updates_per_epoch
    run_config["total_updates"] = total_updates
    run_config["warmup_steps"] = warmup_steps
    run_config["effective_batch_size"] = effective_batch
    run_config["training_objective"] = "assistant-only normal LM SFT (proven baseline)"
    run_config["training_epochs"] = args.epochs
    run_config["training_max_pixels"] = args.max_pixels
    run_config["inference_sides"] = list(args.inference_sides)
    run_config["lora_precision"] = precision_info
    run_config["system_prompt"] = SYSTEM_PROMPT
    run_config.update(data_stats)

    save_json(args.output_dir / "run_config.json", run_config)
    wandb_run = init_wandb(args, run_config)

    if wandb_run is not None:
        for key in (
            "original_rows",
            "synthetic_rows",
            "extra_rows_unique",
            "extra_unique_images",
            "total_train_rows",
            "total_unique_image_paths",
        ):
            if key in data_stats:
                wandb_run.summary[f"data/{key}"] = data_stats[key]

    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    global_update = 0
    micro_step_total = 0
    running_raw_loss = 0.0
    running_micro_steps = 0
    checkpoint_records: list[dict[str, Any]] = []

    console.print(
        Panel(
            "[bold green]Training start[/bold green]\n"
            "NO validation · NO TTA · assistant-only LM SFT · 640² training budget\n"
            "Cosine schedule spans the full actual merged dataset.",
            border_style="green",
            box=box.ROUNDED,
        )
    )

    pass_index = 0
    while global_update < total_updates:
        pass_index += 1
        model.train()
        model.config.use_cache = False

        progress = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"train pass {pass_index}",
            dynamic_ncols=True,
            colour="cyan",
        )

        for step, batch in enumerate(progress, start=1):
            if global_update >= total_updates:
                break

            micro_step_total += 1
            batch = move_to_device(batch, device)

            # Preserve the proven gradient-accumulation behavior, including a short
            # final group at the end of each full data pass.
            group_start = (
                ((step - 1) // args.grad_accum_steps) * args.grad_accum_steps + 1
            )
            group_end = min(
                group_start + args.grad_accum_steps - 1,
                len(train_loader),
            )
            group_size = group_end - group_start + 1

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                raw_loss = outputs.loss
                loss = raw_loss / group_size

            loss.backward()

            running_raw_loss += float(raw_loss.detach())
            running_micro_steps += 1

            if step != group_end:
                continue

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                args.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1

            current_lr = scheduler.get_last_lr()[0]
            current_epoch_progress = global_update / current_updates_per_epoch
            baseline_epoch_progress = global_update / args.baseline_updates_per_epoch

            if global_update % args.log_every == 0 or global_update == 1:
                mean_loss = running_raw_loss / max(1, running_micro_steps)
                progress.set_postfix(
                    loss=f"{mean_loss:.4f}",
                    lr=f"{current_lr:.2e}",
                    cur_ep=f"{current_epoch_progress:.3f}",
                    base_ep=f"{baseline_epoch_progress:.3f}",
                )
                wandb_log(
                    wandb_run,
                    {
                        "train/loss": mean_loss,
                        "train/lr": current_lr,
                        "train/current_data_epoch": current_epoch_progress,
                        "train/baseline_equivalent_epoch": baseline_epoch_progress,
                        "train/micro_step": micro_step_total,
                    },
                    step=global_update,
                )
                running_raw_loss = 0.0
                running_micro_steps = 0

            if global_update in checkpoint_targets:
                requested_epoch = checkpoint_target_map[global_update]
                checkpoint_dir = checkpoints_root / checkpoint_name(
                    requested_epoch, global_update
                )
                elapsed_minutes = (time.time() - started) / 60

                save_checkpoint(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    epoch_progress=current_epoch_progress,
                    global_update=global_update,
                    learning_rate=current_lr,
                    elapsed_minutes=elapsed_minutes,
                )

                record = {
                    "requested_epoch": requested_epoch,
                    "global_update": global_update,
                    "current_data_epoch": current_epoch_progress,
                    "baseline_equivalent_epoch": baseline_epoch_progress,
                    "learning_rate": current_lr,
                    "elapsed_minutes": elapsed_minutes,
                    "checkpoint_dir": str(checkpoint_dir),
                }
                checkpoint_records.append(record)
                save_json(args.output_dir / "checkpoints.json", checkpoint_records)

                wandb_log(
                    wandb_run,
                    {
                        "checkpoint/update": global_update,
                        "checkpoint/current_data_epoch": current_epoch_progress,
                        "checkpoint/baseline_equivalent_epoch": baseline_epoch_progress,
                        "checkpoint/lr": current_lr,
                        "checkpoint/elapsed_minutes": elapsed_minutes,
                    },
                    step=global_update,
                )

                if wandb_run is not None:
                    wandb_run.summary[f"checkpoint_update_{global_update:04d}"] = str(
                        checkpoint_dir
                    )

            if global_update >= total_updates:
                break

    if global_update != total_updates:
        raise RuntimeError(
            f"training stopped at update {global_update}, expected {total_updates}"
        )

    if len(checkpoint_records) != len(checkpoint_target_map):
        raise RuntimeError(
            f"saved {len(checkpoint_records)} checkpoints, "
            f"expected {len(checkpoint_target_map)}"
        )

    train_elapsed = (time.time() - started) / 60
    rich_notice(f"Training complete · {train_elapsed:.1f} min", kind="success")

    if wandb_run is not None:
        wandb_run.summary["train/elapsed_minutes"] = train_elapsed
        wandb_run.summary["train/final_update"] = global_update

    del optimizer, scheduler
    torch.cuda.empty_cache()

    if args.run_inference:
        sample_path = args.data_root / args.sample_submission

        # Build one inference processor/loader per pixel budget. This is NOT TTA:
        # each resolution produces its own independent submission file.
        inference_contexts: dict[int, tuple[Any, int, DataLoader]] = {}
        for side in args.inference_sides:
            pixel_budget = side * side
            if pixel_budget == args.max_pixels:
                inference_processor = processor
            else:
                rich_notice(
                    f"Loading inference processor · r{side} · max_pixels={pixel_budget:,}",
                    kind="info",
                )
                inference_processor = AutoProcessor.from_pretrained(
                    args.model_id,
                    min_pixels=args.min_pixels,
                    max_pixels=pixel_budget,
                    trust_remote_code=True,
                )
                inference_processor.tokenizer.padding_side = "left"
                if inference_processor.tokenizer.pad_token_id is None:
                    inference_processor.tokenizer.pad_token = (
                        inference_processor.tokenizer.eos_token
                    )

            test_loader = build_dataloader(
                test_dataset,
                inference_processor,
                train=False,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
            )
            inference_contexts[side] = (inference_processor, args.eval_batch_size, test_loader)

        probabilities_by_side: dict[int, list[np.ndarray]] = {
            side: [] for side in args.inference_sides
        }
        reference_ids_by_side: dict[int, list[str] | None] = {
            side: None for side in args.inference_sides
        }
        inference_records: list[dict[str, Any]] = []
        manifest_rows: list[dict[str, Any]] = []

        # Restore each LoRA checkpoint only once, then evaluate all requested resolutions.
        for record in checkpoint_records:
            update = int(record["global_update"])
            requested_epoch = float(record.get("requested_epoch", record["current_data_epoch"]))
            checkpoint_dir = Path(record["checkpoint_dir"])
            tag = checkpoint_name(requested_epoch, update)

            console.rule(f"[bold magenta]{tag}[/bold magenta]")
            rich_notice(
                f"Restoring checkpoint · actual ep {record['current_data_epoch']:.3f} · "
                f"old-step-equivalent {record['baseline_equivalent_epoch']:.3f}",
                kind="checkpoint",
            )
            restore_checkpoint(model, checkpoint_dir)
            model.config.use_cache = True
            torch.cuda.empty_cache()

            for side in args.inference_sides:
                inference_processor, inference_batch_size, test_loader = inference_contexts[side]
                answer_ids = get_answer_token_ids(inference_processor)
                training_answer_ids = get_answer_token_ids(processor)
                if answer_ids != training_answer_ids:
                    raise RuntimeError(
                        f"answer token IDs changed at inference side {side}: "
                        f"training={training_answer_ids}, inference={answer_ids}"
                    )

                inference_started = time.time()
                while True:
                    try:
                        ids, probabilities, predictions = predict_choice_probabilities(
                            model=model,
                            processor=inference_processor,
                            loader=test_loader,
                            device=device,
                            description=(
                                f"test {tag} r{side} bs{inference_batch_size}"
                            ),
                        )
                        break
                    except torch.cuda.OutOfMemoryError:
                        if inference_batch_size <= 1:
                            raise
                        new_batch_size = max(1, inference_batch_size // 2)
                        rich_notice(
                            f"CUDA OOM · r{side} batch {inference_batch_size} → "
                            f"retry batch {new_batch_size}",
                            kind="warning",
                        )
                        inference_batch_size = new_batch_size
                        torch.cuda.empty_cache()
                        test_loader = build_dataloader(
                            test_dataset,
                            inference_processor,
                            train=False,
                            batch_size=inference_batch_size,
                            num_workers=args.num_workers,
                        )
                        inference_contexts[side] = (
                            inference_processor,
                            inference_batch_size,
                            test_loader,
                        )

                inference_minutes = (time.time() - inference_started) / 60

                reference_ids = reference_ids_by_side[side]
                if reference_ids is None:
                    reference_ids_by_side[side] = ids
                elif ids != reference_ids:
                    raise RuntimeError(
                        f"test ID order changed between checkpoint runs at r{side}"
                    )

                probabilities_by_side[side].append(probabilities)

                side_submission_dir = submissions_root / f"r{side}"
                side_probability_dir = probabilities_root / f"r{side}"
                side_submission_dir.mkdir(parents=True, exist_ok=True)
                side_probability_dir.mkdir(parents=True, exist_ok=True)

                submission_df = build_submission_frame(ids, predictions, sample_path)
                submission_path = side_submission_dir / f"submission_{tag}.csv"
                submission_df.to_csv(
                    submission_path,
                    index=False,
                    encoding="utf-8-sig",
                )

                probability_path = side_probability_dir / f"probabilities_{tag}.csv"
                save_probability_frame(
                    probability_path,
                    ids,
                    probabilities,
                    predictions,
                )

                prediction_counts = {
                    label: int(sum(pred == label for pred in predictions))
                    for label in LABELS
                }
                inference_record = {
                    "requested_epoch": requested_epoch,
                    "global_update": update,
                    "current_data_epoch": record["current_data_epoch"],
                    "baseline_equivalent_epoch": record["baseline_equivalent_epoch"],
                    "inference_side": side,
                    "inference_max_pixels": side * side,
                    "inference_batch_size": inference_batch_size,
                    "submission_path": str(submission_path),
                    "probability_path": str(probability_path),
                    "inference_minutes": inference_minutes,
                    "prediction_counts": prediction_counts,
                }
                inference_records.append(inference_record)
                manifest_rows.append(
                    {
                        "kind": "checkpoint",
                        "requested_epoch": requested_epoch,
                        "actual_epoch": record["current_data_epoch"],
                        "update": update,
                        "inference_side": side,
                        "max_pixels": side * side,
                        "inference_batch_size": inference_batch_size,
                        "submission_path": str(submission_path),
                        "probability_path": str(probability_path),
                    }
                )

                rich_notice(f"Submission saved → [bold]{submission_path}[/bold]", kind="success")
                rich_notice(f"Probabilities saved → {probability_path}", kind="info")

        save_json(args.output_dir / "inference_results.json", inference_records)

        # Convenience-only checkpoint soft ensemble, separately for each resolution.
        # Previous leaderboard runs showed that a single checkpoint can be better,
        # so every individual submission above is always retained.
        ensemble_summaries: dict[str, Any] = {}
        for side in args.inference_sides:
            reference_ids = reference_ids_by_side[side]
            side_probs = probabilities_by_side[side]
            if reference_ids is None or not side_probs:
                raise RuntimeError(f"no checkpoint predictions were produced for r{side}")

            stacked = np.stack(side_probs, axis=0)
            ensemble_probabilities = stacked.mean(axis=0)
            ensemble_indices = ensemble_probabilities.argmax(axis=1)
            ensemble_predictions = [LABELS[index] for index in ensemble_indices]

            side_submission_dir = submissions_root / f"r{side}"
            side_probability_dir = probabilities_root / f"r{side}"
            ensemble_submission_path = side_submission_dir / "submission_ensemble_soft.csv"
            ensemble_probability_path = side_probability_dir / "probabilities_ensemble_soft.csv"

            ensemble_submission = build_submission_frame(
                reference_ids,
                ensemble_predictions,
                sample_path,
            )
            ensemble_submission.to_csv(
                ensemble_submission_path,
                index=False,
                encoding="utf-8-sig",
            )
            save_probability_frame(
                ensemble_probability_path,
                reference_ids,
                ensemble_probabilities,
                ensemble_predictions,
            )

            checkpoint_pred_indices = stacked.argmax(axis=2)
            unanimous = np.all(
                checkpoint_pred_indices == checkpoint_pred_indices[0:1, :],
                axis=0,
            )
            disagreement_count = int((~unanimous).sum())
            disagreement_ratio = disagreement_count / len(reference_ids)

            ensemble_summaries[f"r{side}"] = {
                "checkpoint_count": len(side_probs),
                "submission_path": str(ensemble_submission_path),
                "probability_path": str(ensemble_probability_path),
                "disagreement_count": disagreement_count,
                "disagreement_ratio": disagreement_ratio,
            }
            manifest_rows.append(
                {
                    "kind": "ensemble_soft",
                    "requested_epoch": np.nan,
                    "actual_epoch": np.nan,
                    "update": np.nan,
                    "inference_side": side,
                    "max_pixels": side * side,
                    "inference_batch_size": np.nan,
                    "submission_path": str(ensemble_submission_path),
                    "probability_path": str(ensemble_probability_path),
                }
            )

            rich_notice(f"r{side} soft-checkpoint ensemble saved → [bold]{ensemble_submission_path}[/bold]", kind="success")
            rich_notice(
                f"r{side} checkpoint disagreement · {disagreement_count:,}/{len(reference_ids):,} "
                f"({disagreement_ratio:.2%})",
                kind="info",
            )

        save_json(args.output_dir / "ensemble_summary.json", ensemble_summaries)
        manifest_df = pd.DataFrame(manifest_rows)
        manifest_path = args.output_dir / "submission_manifest.csv"
        manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
        rich_notice(f"Submission manifest saved → [bold]{manifest_path}[/bold]", kind="success")

        if wandb_run is not None:
            wandb_run.summary["inference/resolutions"] = list(args.inference_sides)
            wandb_run.summary["inference/submission_manifest"] = str(manifest_path)

    total_elapsed = (time.time() - started) / 60
    console.print(
        Panel(
            f"[bold green]Run complete[/bold green]\n"
            f"Total elapsed: [bold]{total_elapsed:.1f} minutes[/bold]\n"
            f"Outputs: [cyan]{args.output_dir}[/cyan]",
            title="✓ Finished",
            border_style="green",
            box=box.DOUBLE,
            padding=(1, 2),
        )
    )

    if wandb_run is not None:
        wandb_run.summary["total_elapsed_minutes"] = total_elapsed
        wandb_run.finish()


if __name__ == "__main__":
    main()
