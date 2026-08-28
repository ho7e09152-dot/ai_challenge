"""Qwen3.5-4B LoRA full-train script for Korean multiple-choice VQA.

Strategy used by this version
-----------------------------
* Uses 100% of train.csv for training (no validation split).
* Adds synthetic rows from train_aug.csv after removing exact copies of originals.
* Trains for 2 epochs by default.
* Saves LoRA checkpoints every 0.5 epoch by default:
    epoch_0.5, epoch_1.0, epoch_1.5, epoch_2.0
* After training, reloads each saved LoRA state and runs test inference.
* Saves one submission CSV per checkpoint.
* Scores the a/b/c/d answer-token logits directly and saves probabilities.
* Creates an equal-weight soft-voting ensemble submission from all checkpoints.
* Logs training metrics/checkpoint events to Weights & Biases when configured.

Recommended RunPod setup
------------------------
    pip install -U "transformers @ git+https://github.com/huggingface/transformers.git"
    pip install -U peft accelerate bitsandbytes pandas pillow tqdm wandb python-dotenv

Example
-------
    cp .env.sample .env
    # Fill WANDB_API_KEY in .env if you want online W&B logging.

    python train_qwen35_4b_fulltrain_checkpoints_wandb.py \
        --data-root . \
        --output-dir outputs/qwen35_4b_fulltrain

Notes
-----
This script intentionally does not use a local validation set. That gives all labeled
original samples to training and leaves checkpoint selection to external evaluation.
If your competition has a private leaderboard, repeated selection on the public
leaderboard can overfit the public split, so keep that risk in mind.
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

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)


MODEL_ID = "Qwen/Qwen3.5-4B"
LABELS = ("a", "b", "c", "d")
TRAIN_COLUMNS = ("id", "path", "question", "a", "b", "c", "d", "answer")
TEST_COLUMNS = ("id", "path", "question", "a", "b", "c", "d")
SYSTEM_PROMPT = (
    "You are a visual multiple-choice question answering assistant. "
    "Inspect the image and answer using exactly one lowercase letter: "
    "a, b, c, or d. Do not explain your answer."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full-train Qwen3.5-4B LoRA with half-epoch checkpoints, "
            "multi-checkpoint inference, soft-voting ensemble, and W&B logging."
        )
    )

    # Paths / data
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--train-csv", type=str, default="train.csv")
    parser.add_argument("--aug-csv", type=str, default="train_aug.csv")
    parser.add_argument("--test-csv", type=str, default="test.csv")
    parser.add_argument(
        "--sample-submission", type=str, default="sample_submission.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/qwen35_4b_fulltrain")
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))

    # Model / reproducibility
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=42)

    # Training: deliberately conservative changes from the previous run
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # LoRA: keep the already-proven capacity unchanged
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Vision resolution: 512^2 -> 640^2 maximum by default
    parser.add_argument("--min-pixels", type=int, default=256 * 256)
    parser.add_argument("--max-pixels", type=int, default=640 * 640)

    # Runtime
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--checkpoint-every",
        type=float,
        default=0.5,
        help="Save a checkpoint every N epochs, e.g. 0.5 -> 0.5/1.0/1.5/2.0.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use NF4 QLoRA. BF16 LoRA is the default for an A6000 48GB.",
    )
    parser.add_argument(
        "--use-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
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
        help="Run inference for every checkpoint and build submissions after training.",
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
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be > 0")
    if args.grad_accum_steps <= 0:
        parser.error("--grad-accum-steps must be > 0")
    if args.train_batch_size <= 0 or args.eval_batch_size <= 0:
        parser.error("batch sizes must be > 0")

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


def build_full_train(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Use every original row and, optionally, every synthetic row.

    train_aug.csv in the original project may contain both exact original rows and
    synthetic siblings. Exact originals are removed from the augmentation frame so
    they are not accidentally double-counted.
    """

    train_path = args.data_root / args.train_csv
    base_df = pd.read_csv(train_path)
    require_columns(base_df, TRAIN_COLUMNS, str(train_path))
    base_df = normalize_answer_column(base_df, str(train_path))

    if base_df["id"].duplicated().any():
        duplicated = base_df.loc[base_df["id"].duplicated(), "id"].head().tolist()
        raise ValueError(
            "train.csv must have one original row per image ID. "
            f"Example duplicated IDs: {duplicated}"
        )

    base_df = base_df.reset_index(drop=True)
    base_count = len(base_df)

    if not args.use_augmentation:
        stats = {
            "original_rows": base_count,
            "synthetic_rows": 0,
            "total_train_rows": base_count,
            "synthetic_to_original_ratio": 0.0,
        }
        print(f"Full train: {base_count:,} original rows; augmentation disabled")
        return base_df, stats

    aug_path = args.data_root / args.aug_csv
    if not aug_path.exists():
        raise FileNotFoundError(
            f"augmentation requested but file does not exist: {aug_path}"
        )

    aug_df = pd.read_csv(aug_path)
    require_columns(aug_df, TRAIN_COLUMNS, str(aug_path))
    aug_df = normalize_answer_column(aug_df, str(aug_path))

    known_ids = set(base_df["id"].astype(str))
    unknown_ids = sorted(set(aug_df["id"].astype(str)) - known_ids)
    if unknown_ids:
        raise ValueError(
            "train_aug.csv contains IDs not found in train.csv: "
            f"{unknown_ids[:10]}"
        )

    # Remove exact copies of original rows from the augmentation file.
    original_keys = exact_row_keys(base_df)
    aug_keys = exact_row_keys(aug_df)
    synthetic_df = aug_df.loc[~aug_keys.isin(original_keys)].copy()
    synthetic_df = synthetic_df.reset_index(drop=True)

    train_df = pd.concat([base_df, synthetic_df], ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    synthetic_count = len(synthetic_df)
    ratio = synthetic_count / base_count if base_count else 0.0
    stats = {
        "original_rows": base_count,
        "synthetic_rows": synthetic_count,
        "total_train_rows": len(train_df),
        "synthetic_to_original_ratio": ratio,
    }

    print(
        f"Full train: {base_count:,} original + {synthetic_count:,} synthetic "
        f"= {len(train_df):,} rows (synthetic/original={ratio:.3f})"
    )
    if not 0.8 <= ratio <= 1.2:
        print(
            "WARNING: augmentation ratio is not close to 1:1. "
            "This may be intentional, but verify train_aug.csv."
        )

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
            if not (data_root / relative).is_file():
                missing.append(relative)
                if len(missing) >= 10:
                    break

        if len(missing) >= 10:
            break

    if missing:
        raise FileNotFoundError(f"missing image paths (first 10): {missing}")

    print(f"Verified {len(checked):,} unique image paths")


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
        image_path = self.data_root / str(row["path"])

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
    print(f"Saved checkpoint: {checkpoint_dir}")


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


def build_checkpoint_targets(
    epochs: float,
    checkpoint_every: float,
    updates_per_epoch: int,
) -> dict[int, float]:
    """Map optimizer update numbers to desired epoch-progress labels."""

    if updates_per_epoch <= 0:
        raise ValueError("updates_per_epoch must be > 0")

    desired: list[float] = []
    value = checkpoint_every
    epsilon = 1e-9

    while value < epochs - epsilon:
        desired.append(round(value, 10))
        value += checkpoint_every

    # Always include the exact final training point.
    desired.append(round(epochs, 10))

    targets: dict[int, float] = {}
    total_updates = max(1, round(epochs * updates_per_epoch))

    for epoch_progress in desired:
        update = max(1, round(epoch_progress * updates_per_epoch))
        update = min(update, total_updates)
        if update in targets:
            raise ValueError(
                "checkpoint spacing is too fine for this dataset/update count. "
                f"Both {targets[update]} and {epoch_progress} map to update {update}."
            )
        targets[update] = epoch_progress

    return targets


def init_wandb(
    args: argparse.Namespace,
    run_config: dict[str, Any],
) -> Any | None:
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
            "Continuing training with W&B disabled. Set the key in .env, "
            "or use WANDB_MODE=offline."
        )
        return None

    try:
        import wandb
    except ImportError:
        print(
            "W&B requested but the wandb package is not installed. "
            "Continuing without W&B. Install with: pip install -U wandb"
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
        print(f"W&B initialization failed ({exc!r}); continuing without W&B")
        return None

    print(
        f"W&B enabled: project={project!r}, mode={mode!r}, "
        f"run={getattr(run, 'name', None)!r}"
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
    print(f"GPU: {gpu_name} ({total_vram:.1f} GiB)")

    train_df, data_stats = build_full_train(args)

    test_path = args.data_root / args.test_csv
    test_df = pd.read_csv(test_path)
    require_columns(test_df, TEST_COLUMNS, str(test_path))

    verify_image_paths(
        (train_df, test_df),
        data_root=args.data_root,
        check_all=args.check_all_images,
    )

    print(f"Loading processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        trust_remote_code=True,
    )
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Save once; checkpoint folders only need adapter weights/config.
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
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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

    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_updates = max(1, round(updates_per_epoch * args.epochs))
    warmup_steps = int(total_updates * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    checkpoint_targets = build_checkpoint_targets(
        epochs=args.epochs,
        checkpoint_every=args.checkpoint_every,
        updates_per_epoch=updates_per_epoch,
    )

    print("Checkpoint plan:")
    for update, epoch_progress in sorted(checkpoint_targets.items()):
        print(f"  update {update:,}/{total_updates:,} -> epoch_{epoch_progress:.1f}")

    run_config = vars(args).copy()
    run_config["data_root"] = str(args.data_root)
    run_config["output_dir"] = str(args.output_dir)
    run_config["env_file"] = str(args.env_file)
    run_config["gpu"] = gpu_name
    run_config["vram_gib"] = total_vram
    run_config["updates_per_epoch"] = updates_per_epoch
    run_config["total_updates"] = total_updates
    run_config["warmup_steps"] = warmup_steps
    run_config["checkpoint_targets"] = {
        str(update): epoch_progress
        for update, epoch_progress in sorted(checkpoint_targets.items())
    }
    run_config.update(data_stats)

    save_json(args.output_dir / "run_config.json", run_config)
    wandb_run = init_wandb(args, run_config)

    if wandb_run is not None:
        wandb_run.summary["data/original_rows"] = data_stats["original_rows"]
        wandb_run.summary["data/synthetic_rows"] = data_stats["synthetic_rows"]
        wandb_run.summary["data/total_train_rows"] = data_stats["total_train_rows"]

    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    global_update = 0
    micro_step_total = 0
    running_raw_loss = 0.0
    running_micro_steps = 0
    checkpoint_records: list[dict[str, Any]] = []

    # epochs is float to keep checkpoint math general, but default is exactly 2.0.
    full_epochs = int(math.floor(args.epochs))
    final_fraction = args.epochs - full_epochs
    epoch_passes = full_epochs + (1 if final_fraction > 1e-9 else 0)

    print(
        "Training with NO validation split: all labeled originals are used for training."
    )

    stop_training = False

    for epoch_index in range(epoch_passes):
        if stop_training:
            break

        model.train()
        model.config.use_cache = False

        # For a fractional final epoch, process only the required prefix of the loader.
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
            if global_update >= total_updates:
                stop_training = True
                break

            micro_step_total += 1
            batch = move_to_device(batch, device)

            # Correctly normalize the final, possibly-short gradient accumulation group.
            group_start = ((step - 1) // args.grad_accum_steps) * args.grad_accum_steps + 1
            group_end = min(
                group_start + args.grad_accum_steps - 1,
                max_micro_steps_this_epoch,
            )
            group_size = group_end - group_start + 1

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                raw_loss = outputs.loss
                loss = raw_loss / group_size

            loss.backward()

            raw_loss_value = float(raw_loss.detach())
            running_raw_loss += raw_loss_value
            running_micro_steps += 1

            should_update = step == group_end

            if should_update:
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
                    mean_loss = running_raw_loss / max(1, running_micro_steps)
                    progress.set_postfix(
                        loss=f"{mean_loss:.4f}",
                        lr=f"{current_lr:.2e}",
                        ep=f"{epoch_progress:.3f}",
                    )
                    wandb_log(
                        wandb_run,
                        {
                            "train/loss": mean_loss,
                            "train/lr": current_lr,
                            "train/epoch_progress": epoch_progress,
                            "train/micro_step": micro_step_total,
                        },
                        step=global_update,
                    )
                    running_raw_loss = 0.0
                    running_micro_steps = 0

                if global_update in checkpoint_targets:
                    target_progress = checkpoint_targets[global_update]
                    checkpoint_dir = checkpoints_root / f"epoch_{target_progress:.1f}"
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
                        wandb_run.summary[
                            f"checkpoint_epoch_{target_progress:.1f}"
                        ] = str(checkpoint_dir)

                if global_update >= total_updates:
                    stop_training = True
                    break

    if global_update != total_updates:
        raise RuntimeError(
            f"training stopped at update {global_update}, expected {total_updates}"
        )

    expected_checkpoint_count = len(checkpoint_targets)
    if len(checkpoint_records) != expected_checkpoint_count:
        raise RuntimeError(
            f"saved {len(checkpoint_records)} checkpoints, "
            f"expected {expected_checkpoint_count}"
        )

    train_elapsed = (time.time() - started) / 60
    print(f"Training complete in {train_elapsed:.1f} minutes")

    if wandb_run is not None:
        wandb_run.summary["train/elapsed_minutes"] = train_elapsed
        wandb_run.summary["train/final_update"] = global_update

    # Optimizer/scheduler are no longer needed; free memory before 4x inference.
    del optimizer, scheduler
    torch.cuda.empty_cache()

    if args.run_inference:
        test_loader = build_dataloader(
            test_dataset,
            processor,
            train=False,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
        )

        sample_path = args.data_root / args.sample_submission
        all_checkpoint_probabilities: list[np.ndarray] = []
        reference_ids: list[str] | None = None
        inference_records: list[dict[str, Any]] = []

        for record in checkpoint_records:
            epoch_progress = float(record["epoch_progress"])
            checkpoint_dir = Path(record["checkpoint_dir"])

            print(f"Restoring checkpoint epoch_{epoch_progress:.1f}")
            restore_checkpoint(model, checkpoint_dir)
            model.config.use_cache = True
            torch.cuda.empty_cache()

            inference_started = time.time()
            ids, probabilities, predictions = predict_choice_probabilities(
                model=model,
                processor=processor,
                loader=test_loader,
                device=device,
                description=f"test epoch_{epoch_progress:.1f}",
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
            submission_path = (
                submissions_root / f"submission_epoch_{epoch_progress:.1f}.csv"
            )
            submission_df.to_csv(
                submission_path,
                index=False,
                encoding="utf-8-sig",
            )

            probability_path = (
                probabilities_root / f"probabilities_epoch_{epoch_progress:.1f}.csv"
            )
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
                    f"inference/epoch_{epoch_progress:.1f}_minutes": inference_minutes,
                    f"inference/epoch_{epoch_progress:.1f}_pred_a": prediction_counts["a"],
                    f"inference/epoch_{epoch_progress:.1f}_pred_b": prediction_counts["b"],
                    f"inference/epoch_{epoch_progress:.1f}_pred_c": prediction_counts["c"],
                    f"inference/epoch_{epoch_progress:.1f}_pred_d": prediction_counts["d"],
                },
                step=global_update,
            )

        save_json(args.output_dir / "inference_results.json", inference_records)

        if reference_ids is None or not all_checkpoint_probabilities:
            raise RuntimeError("no checkpoint predictions were produced")

        # Equal-weight soft voting over all four (default) checkpoints.
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

        # Useful diagnostic: how many test questions changed between checkpoints.
        checkpoint_pred_indices = stacked.argmax(axis=2)  # [num_ckpt, num_samples]
        unanimous = np.all(
            checkpoint_pred_indices == checkpoint_pred_indices[0:1, :],
            axis=0,
        )
        disagreement_count = int((~unanimous).sum())
        disagreement_ratio = disagreement_count / len(reference_ids)

        ensemble_summary = {
            "checkpoint_count": len(all_checkpoint_probabilities),
            "submission_path": str(ensemble_submission_path),
            "probability_path": str(ensemble_probability_path),
            "disagreement_count": disagreement_count,
            "disagreement_ratio": disagreement_ratio,
        }
        save_json(args.output_dir / "ensemble_summary.json", ensemble_summary)

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
