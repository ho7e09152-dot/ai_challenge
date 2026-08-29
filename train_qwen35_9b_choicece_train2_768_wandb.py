"""Qwen3.5-9B LoRA training for Korean multiple-choice VQA.

This version is tuned around the best previous run and adds a second image/QA set.

Main changes from the previous 9B script
----------------------------------------
1. NO validation split. Every labeled row is used for training.
2. Base data:
      train.csv
      + synthetic-only rows extracted from train_aug.csv
3. Extra data:
      train2_aug.csv
      + images under train2/
   `train2_aug.csv` must have the same columns as train.csv. Its `path` may already
   contain `train2/...`, or may contain only a filename/relative path; the script
   will try to resolve it under train2/ automatically.
4. Dynamic choice shuffling during training.
   The four answer choices are re-permuted every time a training example is read,
   and the answer index is remapped accordingly. This reduces answer-position bias
   without generating more CSV rows.
5. 4-choice classification loss instead of normal language-model SFT loss.
   We score only the next-token logits for a/b/c/d and apply CrossEntropyLoss to the
   correct choice. This aligns the training objective with the competition metric.
6. max_pixels defaults to 768 x 768 for both training and inference.
7. Actual training defaults to 1.75 epochs, while the cosine LR scheduler keeps a
   2.0-epoch horizon. This preserves the LR trajectory that worked well previously
   instead of forcing LR to zero at 1.75 epochs.
8. Checkpoints default to 1.0 / 1.25 / 1.5 / 1.75 epochs. Each checkpoint gets its
   own submission CSV and probability CSV after training.
9. W&B logging is retained and now also logs 4-choice training accuracy.

Expected input layout
---------------------
    .
    ├── train.csv
    ├── train_aug.csv
    ├── train2_aug.csv
    ├── test.csv
    ├── sample_submission.csv
    ├── train2/
    │   ├── ... extra images ...
    │   └── ...
    └── other image directories referenced by train.csv/test.csv

Recommended environment (A100 80GB)
-----------------------------------
    pip install -U "transformers @ git+https://github.com/huggingface/transformers.git"
    pip install -U peft accelerate bitsandbytes pandas pillow tqdm wandb \
        python-dotenv safetensors torchvision

Typical run
-----------
    python train_qwen35_9b_choicece_train2_768_wandb.py

If 768^2 + batch=2 is unexpectedly OOM, keep the effective batch size at 16:
    python train_qwen35_9b_choicece_train2_768_wandb.py \
        --train-batch-size 1 --grad-accum-steps 16

Important leaderboard note
--------------------------
There is intentionally no local validation set. Checkpoint selection therefore uses
external evaluation (for example the public leaderboard). Repeatedly tuning to the
public leaderboard can overfit that public subset, so retain multiple sensible
checkpoints for final/private evaluation.
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
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Load the default .env before importing Transformers/Hugging Face Hub.
# This is useful when HF_HOME / HF_HUB_CACHE are defined in .env.
try:
    from dotenv import load_dotenv
except ImportError as exc:
    raise RuntimeError(
        "python-dotenv is required. Install it with: pip install -U python-dotenv"
    ) from exc

load_dotenv(Path(".env"), override=False)

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

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)


MODEL_ID = "Qwen/Qwen3.5-9B"
LABELS = ("a", "b", "c", "d")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
TRAIN_COLUMNS = ("id", "path", "question", "a", "b", "c", "d", "answer")
TEST_COLUMNS = ("id", "path", "question", "a", "b", "c", "d")
CONTENT_COLUMNS = ("path", "question", "a", "b", "c", "d", "answer")

SYSTEM_PROMPT = (
    "You are a visual multiple-choice question answering assistant. "
    "Inspect the image and answer using exactly one lowercase letter: "
    "a, b, c, or d. Do not explain your answer."
)


def parse_checkpoint_epochs(text: str) -> list[float]:
    """Parse a comma-separated checkpoint list such as '1.0,1.25,1.5,1.75'."""
    values: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value <= 0:
            raise argparse.ArgumentTypeError("checkpoint epochs must be > 0")
        values.append(value)

    if not values:
        raise argparse.ArgumentTypeError("at least one checkpoint epoch is required")

    # Preserve order but remove duplicates.
    return list(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qwen3.5-9B full-train VQA: train+aug+train2, dynamic choice shuffle, "
            "4-choice CE loss, 768^2 vision, dense checkpoints, W&B."
        )
    )

    # ------------------------------------------------------------------
    # Paths / data
    # ------------------------------------------------------------------
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
        "--output-dir",
        type=Path,
        default=Path("outputs/qwen35_9b_choicece_train2_768"),
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))

    # ------------------------------------------------------------------
    # Model / reproducibility
    # ------------------------------------------------------------------
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=42)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    # Actual optimization stops at 1.75 epochs by default because the previous
    # experiment peaked around 1.5 and fell at 2.0.
    parser.add_argument("--epochs", type=float, default=1.75)

    # The scheduler still behaves as if training would continue to 2.0 epochs.
    # This avoids changing the LR curve merely because we stop earlier.
    parser.add_argument("--scheduler-epochs", type=float, default=2.0)

    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Keep the already-proven LoRA capacity, but use slightly more dropout as a
    # modest regularizer because the previous 2.0-epoch checkpoint degraded.
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.08)

    # Vision resolution: upgrade the previous 640^2 maximum to 768^2.
    parser.add_argument("--min-pixels", type=int, default=256 * 256)
    parser.add_argument("--max-pixels", type=int, default=768 * 768)

    # Save only the range that is plausible given the previous leaderboard curve.
    parser.add_argument(
        "--checkpoint-epochs",
        type=parse_checkpoint_epochs,
        default=parse_checkpoint_epochs("1.0,1.25,1.5,1.75"),
        help="Comma-separated epoch checkpoints, default: 1.0,1.25,1.5,1.75",
    )

    # ------------------------------------------------------------------
    # Runtime / switches
    # ------------------------------------------------------------------
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use NF4 QLoRA. BF16 LoRA is recommended on an A100 80GB.",
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
        help="Include train2_aug.csv and resolve its images under train2/.",
    )
    parser.add_argument(
        "--dynamic-choice-shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomly permute a/b/c/d every time a training row is read.",
    )
    parser.add_argument(
        "--check-all-images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--run-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run test inference for every saved checkpoint after training.",
    )

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable W&B when WANDB_API_KEY is available, or when "
            "WANDB_MODE=offline. Otherwise training continues without W&B."
        ),
    )
    parser.add_argument("--wandb-project", type=str, default="qwen35-vqa")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be > 0")
    if args.scheduler_epochs <= 0:
        parser.error("--scheduler-epochs must be > 0")
    if args.scheduler_epochs < args.epochs:
        parser.error("--scheduler-epochs should be >= --epochs")
    if args.grad_accum_steps <= 0:
        parser.error("--grad-accum-steps must be > 0")
    if args.train_batch_size <= 0 or args.eval_batch_size <= 0:
        parser.error("batch sizes must be > 0")
    if args.min_pixels <= 0 or args.max_pixels <= 0:
        parser.error("pixel limits must be > 0")
    if args.min_pixels > args.max_pixels:
        parser.error("--min-pixels cannot exceed --max-pixels")

    too_late = [value for value in args.checkpoint_epochs if value > args.epochs + 1e-9]
    if too_late:
        parser.error(
            "checkpoint epochs cannot exceed actual training epochs; "
            f"invalid values: {too_late}"
        )

    return args


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Make Python/NumPy RNGs deterministic inside DataLoader workers."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


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


def content_keys(df: pd.DataFrame) -> pd.MultiIndex:
    """Exact training-content key, intentionally excluding ID."""
    return pd.MultiIndex.from_frame(df[list(CONTENT_COLUMNS)].astype(str))


def resolve_extra_image_paths(
    frame: pd.DataFrame,
    data_root: Path,
    extra_image_dir: str,
) -> pd.DataFrame:
    """Resolve train2_aug.csv paths robustly.

    Accepted examples for a file physically located at train2/foo.jpg:
      - path = "train2/foo.jpg"
      - path = "foo.jpg"
      - path = "some/subdir/foo.jpg" if train2/some/subdir/foo.jpg exists

    If the original path already resolves under data_root, it is kept unchanged.
    """
    result = frame.copy()
    resolved: list[str] = []
    unresolved_examples: list[str] = []

    for raw_value in result["path"].astype(str):
        # Normalize Windows separators because RunPod is Linux.
        normalized = raw_value.strip().replace("\\", "/")
        original = Path(normalized)

        candidates: list[Path] = []

        # Absolute path or path already relative to data_root.
        if original.is_absolute():
            candidates.append(original)
        else:
            candidates.append(data_root / original)
            candidates.append(data_root / extra_image_dir / original)
            # Useful if CSV contains an unrelated directory prefix but the actual
            # file was copied flat into train2/.
            candidates.append(data_root / extra_image_dir / original.name)

        chosen: Path | None = None
        for candidate in candidates:
            if candidate.is_file():
                chosen = candidate
                break

        if chosen is None:
            unresolved_examples.append(raw_value)
            # Keep the most likely relative path so verify_image_paths produces a
            # useful error later.
            chosen = data_root / extra_image_dir / original

        try:
            relative = chosen.resolve().relative_to(data_root.resolve())
            resolved.append(relative.as_posix())
        except ValueError:
            # Absolute paths outside data_root are valid too.
            resolved.append(str(chosen.resolve()))

    result["path"] = resolved

    if unresolved_examples:
        print(
            "WARNING: some train2 paths could not be resolved immediately; "
            "the full image verification step will fail if they truly do not exist. "
            f"Examples: {unresolved_examples[:5]}"
        )

    return result


def build_full_train(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one full training frame without creating a validation split."""

    # ------------------------------------------------------------------
    # 1) Original train.csv
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2) train_aug.csv
    #    The known file contains exact originals + one synthetic sibling per row.
    #    Remove exact originals first so they are not double-counted.
    # ------------------------------------------------------------------
    synthetic_count = 0
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

        original_keys = content_keys(base_df)
        aug_keys = content_keys(aug_df)
        synthetic_df = aug_df.loc[~aug_keys.isin(original_keys)].copy()
        synthetic_df = synthetic_df.drop_duplicates(
            subset=list(CONTENT_COLUMNS)
        ).reset_index(drop=True)

        synthetic_count = len(synthetic_df)
        pieces.append(synthetic_df)

    # ------------------------------------------------------------------
    # 3) New train2_aug.csv
    #    There is no separate train2.csv requirement. Every valid unique row in
    #    this file is treated as additional labeled training data.
    # ------------------------------------------------------------------
    extra_count_raw = 0
    extra_count_unique = 0
    extra_unique_images = 0
    id_overlap_with_base = 0

    if args.use_extra_data:
        extra_path = args.data_root / args.extra_train_csv
        if not extra_path.exists():
            raise FileNotFoundError(
                "extra data is enabled but the CSV does not exist: "
                f"{extra_path}. Use --no-use-extra-data to disable it."
            )

        extra_df = pd.read_csv(extra_path)
        require_columns(extra_df, TRAIN_COLUMNS, str(extra_path))
        extra_df = normalize_answer_column(extra_df, str(extra_path))
        extra_df = resolve_extra_image_paths(
            extra_df,
            data_root=args.data_root,
            extra_image_dir=args.extra_image_dir,
        )

        extra_count_raw = len(extra_df)
        extra_df = extra_df.drop_duplicates(
            subset=list(CONTENT_COLUMNS)
        ).reset_index(drop=True)
        extra_count_unique = len(extra_df)
        extra_unique_images = extra_df["path"].astype(str).nunique()

        # Extra IDs do NOT need to be globally unique for training, because IDs are
        # metadata only. Report collisions rather than failing; independently-built
        # train2 datasets commonly restart IDs from 0/1.
        id_overlap_with_base = len(
            set(extra_df["id"].astype(str)) & set(base_df["id"].astype(str))
        )
        if id_overlap_with_base:
            print(
                f"NOTE: train2 has {id_overlap_with_base:,} ID values that also "
                "appear in train.csv. This is allowed because training IDs are "
                "not used as labels."
            )

        pieces.append(extra_df)

    # ------------------------------------------------------------------
    # 4) Merge and remove any accidental exact content duplicates across sources.
    # ------------------------------------------------------------------
    merged_before_dedup = pd.concat(pieces, ignore_index=True)
    before_count = len(merged_before_dedup)

    train_df = merged_before_dedup.drop_duplicates(
        subset=list(CONTENT_COLUMNS)
    ).reset_index(drop=True)
    cross_source_duplicates_removed = before_count - len(train_df)

    # Shuffle once here. DataLoader also shuffles on every pass.
    train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    stats = {
        "base_original_rows": base_count,
        "base_synthetic_rows": synthetic_count,
        "extra_rows_raw": extra_count_raw,
        "extra_rows_unique": extra_count_unique,
        "extra_unique_images": extra_unique_images,
        "extra_id_overlap_with_base": id_overlap_with_base,
        "cross_source_duplicates_removed": cross_source_duplicates_removed,
        "total_train_rows": len(train_df),
        "total_unique_image_paths": train_df["path"].astype(str).nunique(),
    }

    print("Training data summary (NO validation split):")
    print(f"  train.csv original       : {base_count:,}")
    print(f"  train_aug synthetic-only : {synthetic_count:,}")
    if args.use_extra_data:
        print(
            f"  train2_aug.csv           : {extra_count_unique:,} unique rows "
            f"({extra_count_raw:,} raw)"
        )
        print(f"  train2 unique images     : {extra_unique_images:,}")
    print(f"  cross-source dedup       : -{cross_source_duplicates_removed:,}")
    print(f"  TOTAL training rows      : {len(train_df):,}")
    print(
        f"  TOTAL unique image paths : "
        f"{train_df['path'].astype(str).nunique():,}"
    )

    return train_df, stats


def verify_image_paths(
    frames: Iterable[pd.DataFrame],
    data_root: Path,
    check_all: bool,
) -> None:
    missing: list[str] = []
    checked: set[str] = set()

    for frame in frames:
        paths = frame["path"].astype(str)
        if not check_all:
            paths = paths.head(32)

        for raw_path in paths:
            if raw_path in checked:
                continue
            checked.add(raw_path)

            path = Path(raw_path)
            actual = path if path.is_absolute() else data_root / path
            if not actual.is_file():
                missing.append(raw_path)
                if len(missing) >= 20:
                    break

        if len(missing) >= 20:
            break

    if missing:
        raise FileNotFoundError(
            "missing image paths (first 20): " + repr(missing)
        )

    print(f"Verified {len(checked):,} unique image paths")


def build_mc_prompt(question: str, choices: Sequence[str]) -> str:
    return (
        f"{str(question).strip()}\n"
        f"(a) {str(choices[0]).strip()}\n"
        f"(b) {str(choices[1]).strip()}\n"
        f"(c) {str(choices[2]).strip()}\n"
        f"(d) {str(choices[3]).strip()}\n\n"
        "정답을 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."
    )


class VQADataset(Dataset):
    """VQA dataset with optional on-the-fly answer-choice permutation.

    Dynamic permutation is applied only to training rows. The image and question are
    untouched. We permute the four choice VALUES and remap the target index using the
    original answer label, so semantic correctness is preserved even if choice texts
    are duplicated.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: Path,
        has_answer: bool,
        dynamic_choice_shuffle: bool = False,
    ):
        self.records = frame.to_dict("records")
        self.data_root = data_root
        self.has_answer = has_answer
        self.dynamic_choice_shuffle = dynamic_choice_shuffle and has_answer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]

        raw_image_path = Path(str(row["path"]))
        image_path = (
            raw_image_path
            if raw_image_path.is_absolute()
            else self.data_root / raw_image_path
        )
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB").copy()

        original_choices = [str(row[label]).strip() for label in LABELS]

        if self.has_answer:
            original_answer_label = str(row["answer"]).strip().lower()
            original_answer_index = LABEL_TO_INDEX[original_answer_label]
        else:
            original_answer_index = -1

        if self.dynamic_choice_shuffle:
            # DataLoader workers each have their own deterministic torch RNG stream.
            # Because this is called on every access, the same row can receive a new
            # permutation on a later epoch/pass.
            permutation = torch.randperm(len(LABELS)).tolist()
            choices = [original_choices[i] for i in permutation]
            target_index = permutation.index(original_answer_index)
        else:
            choices = original_choices
            target_index = original_answer_index

        sample: dict[str, Any] = {
            "id": str(row["id"]),
            "image": rgb_image,
            "prompt": build_mc_prompt(str(row["question"]), choices),
        }

        if self.has_answer:
            sample["choice_target"] = int(target_index)

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


def render_prompt(processor: Any, prompt: str, image: Image.Image) -> str:
    messages = [system_message(), user_message(prompt, image)]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@dataclass
class ChoiceCollator:
    """Create prompt-only batches for both training and inference.

    We use LEFT padding for a crucial reason: every sample's final real prompt token
    then sits at tensor index -1. `outputs.logits[:, -1, :]` is therefore exactly the
    next-token distribution where the model should emit a/b/c/d.
    """

    processor: Any
    train: bool

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        self.processor.tokenizer.padding_side = "left"

        images = [sample["image"] for sample in samples]
        prompt_texts = [
            render_prompt(self.processor, sample["prompt"], sample["image"])
            for sample in samples
        ]

        encoded = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        # With left padding, the final position must always be a real token.
        if "attention_mask" in encoded and not bool(
            torch.all(encoded["attention_mask"][:, -1] == 1)
        ):
            raise RuntimeError(
                "left-padding invariant failed: final prompt token is padded"
            )

        batch: dict[str, Any] = {
            "model_inputs": encoded,
            "ids": [sample["id"] for sample in samples],
        }

        if self.train:
            batch["choice_targets"] = torch.tensor(
                [sample["choice_target"] for sample in samples],
                dtype=torch.long,
            )

        return batch


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
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        collate_fn=ChoiceCollator(processor=processor, train=train),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator if train else None,
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


def format_epoch_label(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def save_checkpoint(
    model: Any,
    checkpoint_dir: Path,
    epoch_progress: float,
    global_update: int,
    learning_rate: float,
    elapsed_minutes: float,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Standard PEFT files.
    model.save_pretrained(checkpoint_dir, safe_serialization=True)

    # Raw state enables quick in-process switching during multi-checkpoint inference.
    raw_state_path = checkpoint_dir / "adapter_state.pt"
    torch.save(cpu_adapter_state(model), raw_state_path)

    save_json(
        checkpoint_dir / "checkpoint_meta.json",
        {
            "epoch_progress": epoch_progress,
            "global_update": global_update,
            "learning_rate": learning_rate,
            "elapsed_minutes": elapsed_minutes,
        },
    )
    print(f"Saved checkpoint: {checkpoint_dir}")


def load_raw_adapter_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
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


@torch.inference_mode()
def predict_choice_probabilities(
    model: Any,
    processor: Any,
    loader: DataLoader,
    device: torch.device,
    answer_token_ids: Sequence[int],
    description: str,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Score only the next-token logits for a/b/c/d."""
    model.eval()
    previous_use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = True

    all_ids: list[str] = []
    probability_chunks: list[np.ndarray] = []

    for batch in tqdm(loader, desc=description, leave=False):
        ids = batch["ids"]
        inputs = move_to_device(batch["model_inputs"], device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**inputs)

        choice_logits = outputs.logits[:, -1, list(answer_token_ids)].float()
        probabilities = torch.softmax(choice_logits, dim=-1)

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


def build_checkpoint_targets(
    checkpoint_epochs: Sequence[float],
    actual_epochs: float,
    updates_per_epoch: int,
    total_actual_updates: int,
) -> dict[int, float]:
    """Map desired epoch values to optimizer-update indices."""
    if updates_per_epoch <= 0:
        raise ValueError("updates_per_epoch must be > 0")

    desired = list(checkpoint_epochs)
    if not any(math.isclose(value, actual_epochs, abs_tol=1e-9) for value in desired):
        desired.append(actual_epochs)

    targets: dict[int, float] = {}
    for epoch_progress in sorted(desired):
        if epoch_progress > actual_epochs + 1e-9:
            continue
        update = max(1, round(epoch_progress * updates_per_epoch))
        update = min(update, total_actual_updates)
        if update in targets and not math.isclose(
            targets[update], epoch_progress, abs_tol=1e-9
        ):
            raise ValueError(
                "checkpoint spacing is too fine for this dataset/update count: "
                f"{targets[update]} and {epoch_progress} both map to update {update}"
            )
        targets[update] = epoch_progress

    return targets


def init_wandb(args: argparse.Namespace, run_config: dict[str, Any]) -> Any | None:
    if not args.wandb:
        print("W&B disabled by --no-wandb")
        return None

    mode = os.getenv("WANDB_MODE", "online").strip().lower() or "online"
    api_key = os.getenv("WANDB_API_KEY", "").strip()

    if mode == "disabled":
        print("W&B disabled because WANDB_MODE=disabled")
        return None
    if mode == "online" and not api_key:
        print(
            "W&B online logging requested, but WANDB_API_KEY is not set. "
            "Continuing without W&B."
        )
        return None

    try:
        import wandb
    except ImportError:
        print("wandb package is unavailable; continuing without W&B")
        return None

    project = os.getenv("WANDB_PROJECT", "").strip() or args.wandb_project
    entity = os.getenv("WANDB_ENTITY", "").strip() or None
    run_name = (
        os.getenv("WANDB_RUN_NAME", "").strip()
        or args.wandb_run_name
        or "qwen35-9b-choicece-train2-768"
    )
    tags_raw = os.getenv("WANDB_TAGS", "").strip()
    tags = [item.strip() for item in tags_raw.split(",") if item.strip()] or [
        "qwen3.5-9b",
        "choice-ce",
        "dynamic-choice-shuffle",
        "train2",
        "768px",
    ]

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
        print(f"W&B initialization failed ({exc!r}); continuing without W&B")
        return None

    print(
        f"W&B enabled: project={project!r}, mode={mode!r}, "
        f"run={getattr(run, 'name', None)!r}"
    )
    return run


def wandb_log(run: Any | None, values: dict[str, Any], step: int | None = None) -> None:
    if run is not None:
        run.log(values, step=step)


def main() -> None:
    args = parse_args()

    # Load a custom env file too. The default .env was already loaded before
    # Transformers import, but this lets W&B settings come from --env-file.
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
    print(f"GPU: {gpu_name} ({total_vram:.1f} GiB)")

    # ------------------------------------------------------------------
    # Data: full train, no validation split.
    # ------------------------------------------------------------------
    train_df, data_stats = build_full_train(args)

    test_path = args.data_root / args.test_csv
    test_df = pd.read_csv(test_path)
    require_columns(test_df, TEST_COLUMNS, str(test_path))

    verify_image_paths(
        (train_df, test_df),
        data_root=args.data_root,
        check_all=args.check_all_images,
    )

    # ------------------------------------------------------------------
    # Processor / model
    # ------------------------------------------------------------------
    print(f"Loading processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        trust_remote_code=True,
    )
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.save_pretrained(processor_dir)

    answer_token_ids = get_answer_token_ids(processor)
    print(
        "Answer token IDs: "
        + ", ".join(
            f"{label}={token_id}" for label, token_id in zip(LABELS, answer_token_ids)
        )
    )

    model_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
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
        print("Loading NF4 4-bit base model")
    else:
        print("Loading BF16 base model")

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

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        # Keep all-linear because this is the already-proven configuration from the
        # successful 9B run. Changing module targeting at the same time would add a
        # second large experimental variable.
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Datasets / loaders
    # ------------------------------------------------------------------
    train_dataset = VQADataset(
        train_df,
        args.data_root,
        has_answer=True,
        dynamic_choice_shuffle=args.dynamic_choice_shuffle,
    )
    test_dataset = VQADataset(
        test_df,
        args.data_root,
        has_answer=False,
        dynamic_choice_shuffle=False,
    )

    train_loader = build_dataloader(
        train_dataset,
        processor,
        train=True,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # ------------------------------------------------------------------
    # LR / checkpoint schedule
    # ------------------------------------------------------------------
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_actual_updates = max(1, round(updates_per_epoch * args.epochs))
    total_scheduler_updates = max(
        total_actual_updates,
        round(updates_per_epoch * args.scheduler_epochs),
    )
    warmup_steps = int(total_scheduler_updates * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_scheduler_updates,
    )

    checkpoint_targets = build_checkpoint_targets(
        checkpoint_epochs=args.checkpoint_epochs,
        actual_epochs=args.epochs,
        updates_per_epoch=updates_per_epoch,
        total_actual_updates=total_actual_updates,
    )

    print("Training schedule:")
    print(f"  actual epochs          : {args.epochs}")
    print(f"  scheduler horizon      : {args.scheduler_epochs} epochs")
    print(f"  micro-batches / epoch  : {len(train_loader):,}")
    print(f"  optimizer updates/epoch: {updates_per_epoch:,}")
    print(f"  actual optimizer steps : {total_actual_updates:,}")
    print(f"  scheduler total steps  : {total_scheduler_updates:,}")
    print(f"  warmup steps           : {warmup_steps:,}")
    print(
        f"  effective batch size   : "
        f"{args.train_batch_size * args.grad_accum_steps}"
    )
    print("Checkpoint plan:")
    for update, epoch_progress in sorted(checkpoint_targets.items()):
        print(
            f"  update {update:,}/{total_actual_updates:,} "
            f"-> epoch_{format_epoch_label(epoch_progress)}"
        )

    run_config = vars(args).copy()
    run_config["data_root"] = str(args.data_root)
    run_config["output_dir"] = str(args.output_dir)
    run_config["env_file"] = str(args.env_file)
    run_config["checkpoint_epochs"] = list(args.checkpoint_epochs)
    run_config["gpu"] = gpu_name
    run_config["vram_gib"] = total_vram
    run_config["updates_per_epoch"] = updates_per_epoch
    run_config["total_actual_updates"] = total_actual_updates
    run_config["total_scheduler_updates"] = total_scheduler_updates
    run_config["warmup_steps"] = warmup_steps
    run_config["objective"] = "next-token 4-choice cross-entropy over a/b/c/d"
    run_config.update(data_stats)
    save_json(args.output_dir / "run_config.json", run_config)

    wandb_run = init_wandb(args, run_config)
    if wandb_run is not None:
        for key, value in data_stats.items():
            wandb_run.summary[f"data/{key}"] = value

    # ------------------------------------------------------------------
    # Training loop: 4-choice CE
    # ------------------------------------------------------------------
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    global_update = 0
    micro_step_total = 0
    running_loss_sum = 0.0
    running_micro_steps = 0
    running_correct = 0
    running_examples = 0
    checkpoint_records: list[dict[str, Any]] = []

    full_epochs = int(math.floor(args.epochs))
    final_fraction = args.epochs - full_epochs
    epoch_passes = full_epochs + (1 if final_fraction > 1e-9 else 0)

    print(
        "Training with NO validation split and 4-choice CE loss. "
        f"Dynamic choice shuffle={'ON' if args.dynamic_choice_shuffle else 'OFF'}."
    )

    stop_training = False

    for epoch_index in range(epoch_passes):
        if stop_training:
            break

        model.train()
        model.config.use_cache = False

        if epoch_index < full_epochs:
            max_micro_steps_this_epoch = len(train_loader)
        else:
            max_micro_steps_this_epoch = max(
                1,
                round(len(train_loader) * final_fraction),
            )

        progress = tqdm(
            train_loader,
            total=max_micro_steps_this_epoch,
            desc=f"train pass {epoch_index + 1}/{epoch_passes}",
        )

        for step, batch in enumerate(progress, start=1):
            if step > max_micro_steps_this_epoch:
                break
            if global_update >= total_actual_updates:
                stop_training = True
                break

            micro_step_total += 1

            inputs = move_to_device(batch["model_inputs"], device)
            targets = batch["choice_targets"].to(device, non_blocking=True)

            # Correctly normalize the final, possibly-short gradient accumulation
            # group so its gradients have the same scale as normal groups.
            group_start = (
                ((step - 1) // args.grad_accum_steps) * args.grad_accum_steps + 1
            )
            group_end = min(
                group_start + args.grad_accum_steps - 1,
                max_micro_steps_this_epoch,
            )
            group_size = group_end - group_start + 1

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)

                # Critical change: do NOT train normal LM labels. We keep only the
                # next-token logits for the four legal answers and directly optimize
                # the correct class index 0..3.
                choice_logits = outputs.logits[:, -1, answer_token_ids].float()
                raw_loss = F.cross_entropy(choice_logits, targets)
                loss = raw_loss / group_size

            loss.backward()

            with torch.no_grad():
                predicted = choice_logits.argmax(dim=-1)
                batch_correct = int((predicted == targets).sum().item())
                batch_examples = int(targets.numel())

            running_loss_sum += float(raw_loss.detach())
            running_micro_steps += 1
            running_correct += batch_correct
            running_examples += batch_examples

            should_update = step == group_end
            if not should_update:
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
            epoch_progress = global_update / updates_per_epoch

            if global_update % args.log_every == 0 or global_update == 1:
                mean_loss = running_loss_sum / max(1, running_micro_steps)
                mean_accuracy = running_correct / max(1, running_examples)

                progress.set_postfix(
                    loss=f"{mean_loss:.4f}",
                    acc=f"{mean_accuracy:.4f}",
                    lr=f"{current_lr:.2e}",
                    ep=f"{epoch_progress:.3f}",
                )

                wandb_log(
                    wandb_run,
                    {
                        "train/loss": mean_loss,
                        "train/choice_accuracy": mean_accuracy,
                        "train/lr": current_lr,
                        "train/epoch_progress": epoch_progress,
                        "train/micro_step": micro_step_total,
                    },
                    step=global_update,
                )

                running_loss_sum = 0.0
                running_micro_steps = 0
                running_correct = 0
                running_examples = 0

            if global_update in checkpoint_targets:
                target_progress = checkpoint_targets[global_update]
                label = format_epoch_label(target_progress)
                checkpoint_dir = checkpoints_root / f"epoch_{label}"
                elapsed_minutes = (time.time() - started) / 60

                save_checkpoint(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    epoch_progress=target_progress,
                    global_update=global_update,
                    learning_rate=current_lr,
                    elapsed_minutes=elapsed_minutes,
                )

                record = {
                    "epoch_progress": target_progress,
                    "global_update": global_update,
                    "learning_rate": current_lr,
                    "elapsed_minutes": elapsed_minutes,
                    "checkpoint_dir": str(checkpoint_dir),
                }
                checkpoint_records.append(record)
                save_json(args.output_dir / "checkpoints.json", checkpoint_records)

                wandb_log(
                    wandb_run,
                    {
                        "checkpoint/epoch_progress": target_progress,
                        "checkpoint/lr": current_lr,
                        "checkpoint/elapsed_minutes": elapsed_minutes,
                    },
                    step=global_update,
                )

                if wandb_run is not None:
                    wandb_run.summary[f"checkpoint_epoch_{label}"] = str(
                        checkpoint_dir
                    )

            if global_update >= total_actual_updates:
                stop_training = True
                break

    if global_update != total_actual_updates:
        raise RuntimeError(
            f"training stopped at update {global_update}, "
            f"expected {total_actual_updates}"
        )

    if len(checkpoint_records) != len(checkpoint_targets):
        raise RuntimeError(
            f"saved {len(checkpoint_records)} checkpoints, "
            f"expected {len(checkpoint_targets)}"
        )

    train_elapsed = (time.time() - started) / 60
    print(f"Training complete in {train_elapsed:.1f} minutes")

    if wandb_run is not None:
        wandb_run.summary["train/elapsed_minutes"] = train_elapsed
        wandb_run.summary["train/final_update"] = global_update

    # Optimizer/scheduler are unnecessary for inference.
    del optimizer, scheduler
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Multi-checkpoint test inference
    # ------------------------------------------------------------------
    if args.run_inference:
        test_loader = build_dataloader(
            test_dataset,
            processor,
            train=False,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )

        sample_path = args.data_root / args.sample_submission
        all_checkpoint_probabilities: list[np.ndarray] = []
        reference_ids: list[str] | None = None
        inference_records: list[dict[str, Any]] = []

        for record in checkpoint_records:
            epoch_progress = float(record["epoch_progress"])
            label = format_epoch_label(epoch_progress)
            checkpoint_dir = Path(record["checkpoint_dir"])

            print(f"Restoring checkpoint epoch_{label}")
            restore_checkpoint(model, checkpoint_dir)
            model.config.use_cache = True
            torch.cuda.empty_cache()

            inference_started = time.time()
            ids, probabilities, predictions = predict_choice_probabilities(
                model=model,
                processor=processor,
                loader=test_loader,
                device=device,
                answer_token_ids=answer_token_ids,
                description=f"test epoch_{label}",
            )
            inference_minutes = (time.time() - inference_started) / 60

            if reference_ids is None:
                reference_ids = ids
            elif ids != reference_ids:
                raise RuntimeError(
                    "test ID order changed between checkpoint inference runs"
                )

            all_checkpoint_probabilities.append(probabilities)

            submission_df = build_submission_frame(ids, predictions, sample_path)
            submission_path = submissions_root / f"submission_epoch_{label}.csv"
            submission_df.to_csv(
                submission_path,
                index=False,
                encoding="utf-8-sig",
            )

            probability_path = probabilities_root / f"probabilities_epoch_{label}.csv"
            save_probability_frame(
                probability_path,
                ids,
                probabilities,
                predictions,
            )

            prediction_counts = {
                candidate: int(sum(pred == candidate for pred in predictions))
                for candidate in LABELS
            }
            inference_record = {
                "epoch_progress": epoch_progress,
                "submission_path": str(submission_path),
                "probability_path": str(probability_path),
                "inference_minutes": inference_minutes,
                "prediction_counts": prediction_counts,
            }
            inference_records.append(inference_record)

            print(f"Saved submission: {submission_path}")
            print(f"Saved probabilities: {probability_path}")

            wandb_log(
                wandb_run,
                {
                    f"inference/epoch_{label}_minutes": inference_minutes,
                    f"inference/epoch_{label}_pred_a": prediction_counts["a"],
                    f"inference/epoch_{label}_pred_b": prediction_counts["b"],
                    f"inference/epoch_{label}_pred_c": prediction_counts["c"],
                    f"inference/epoch_{label}_pred_d": prediction_counts["d"],
                },
                step=global_update,
            )

        save_json(args.output_dir / "inference_results.json", inference_records)

        if reference_ids is None or not all_checkpoint_probabilities:
            raise RuntimeError("no checkpoint predictions were produced")

        # Keep the equal soft-vote artifact for completeness. The previous run showed
        # that a single checkpoint can outperform the ensemble, so do not assume this
        # file is automatically the best submission.
        stacked = np.stack(all_checkpoint_probabilities, axis=0)
        ensemble_probabilities = stacked.mean(axis=0)
        ensemble_indices = ensemble_probabilities.argmax(axis=1)
        ensemble_predictions = [LABELS[index] for index in ensemble_indices]

        ensemble_submission = build_submission_frame(
            reference_ids,
            ensemble_predictions,
            sample_path,
        )
        ensemble_submission_path = submissions_root / "submission_ensemble_soft.csv"
        ensemble_submission.to_csv(
            ensemble_submission_path,
            index=False,
            encoding="utf-8-sig",
        )

        ensemble_probability_path = probabilities_root / "probabilities_ensemble_soft.csv"
        save_probability_frame(
            ensemble_probability_path,
            reference_ids,
            ensemble_probabilities,
            ensemble_predictions,
        )

        print(f"Saved soft-voting ensemble: {ensemble_submission_path}")
        print(f"Saved ensemble probabilities: {ensemble_probability_path}")

        checkpoint_pred_indices = stacked.argmax(axis=2)
        unanimous = np.all(
            checkpoint_pred_indices == checkpoint_pred_indices[0:1, :],
            axis=0,
        )
        disagreement_count = int((~unanimous).sum())
        disagreement_ratio = disagreement_count / len(reference_ids)

        save_json(
            args.output_dir / "ensemble_summary.json",
            {
                "checkpoint_count": len(all_checkpoint_probabilities),
                "submission_path": str(ensemble_submission_path),
                "probability_path": str(ensemble_probability_path),
                "disagreement_count": disagreement_count,
                "disagreement_ratio": disagreement_ratio,
            },
        )

        print(
            f"Checkpoint disagreement: {disagreement_count:,}/{len(reference_ids):,} "
            f"({disagreement_ratio:.2%})"
        )

        if wandb_run is not None:
            wandb_run.summary["ensemble/submission"] = str(ensemble_submission_path)
            wandb_run.summary["ensemble/disagreement_count"] = disagreement_count
            wandb_run.summary["ensemble/disagreement_ratio"] = disagreement_ratio

    total_elapsed = (time.time() - started) / 60
    print(f"Total elapsed: {total_elapsed:.1f} minutes")

    if wandb_run is not None:
        wandb_run.summary["total_elapsed_minutes"] = total_elapsed
        wandb_run.finish()


if __name__ == "__main__":
    main()
