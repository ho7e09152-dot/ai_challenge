"""Qwen3.5-4B (about 5B total parameters) LoRA training for this VQA task.

Designed for the repository layout used by this project and a single RunPod
NVIDIA A6000 48GB GPU. The script:

* splits the original train.csv before using augmented rows (no image leakage),
* trains on original + synthetic rows from train_aug.csv,
* validates only on untouched original rows,
* saves the best LoRA adapter, and
* predicts test.csv and writes a submission CSV.

Recommended RunPod setup (run in a fresh environment):

    pip install -U "transformers @ git+https://github.com/huggingface/transformers.git"
    pip install -U peft accelerate bitsandbytes pandas scikit-learn pillow tqdm

Example:

    python train_qwen35_4b_runpod.py \
        --data-root . \
        --output-dir outputs/qwen35_4b_vqa

The official model ID is Qwen/Qwen3.5-4B. Hugging Face reports roughly 5B
total parameters for this checkpoint; there is no official Qwen3.5-5B ID.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import torch.nn as nn

try:
    from transformers import (
        AutoModelForMultimodalLM,
        AutoProcessor,
        BitsAndBytesConfig,
        get_cosine_schedule_with_warmup,
    )
except ImportError as exc:
    raise RuntimeError(
        "Qwen3.5 requires the latest Transformers. Install it with:\n"
        "pip install -U \"transformers @ "
        "git+https://github.com/huggingface/transformers.git\""
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
        description="Fine-tune Qwen3.5-4B for Korean multiple-choice VQA."
    )
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--train-csv", type=str, default="train.csv")
    parser.add_argument("--aug-csv", type=str, default="train_aug.csv")
    parser.add_argument("--test-csv", type=str, default="test.csv")
    parser.add_argument(
        "--sample-submission", type=str, default="sample_submission.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/qwen35_4b_vqa")
    )
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--min-pixels", type=int, default=256 * 256)
    parser.add_argument("--max-pixels", type=int, default=512 * 512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use NF4 QLoRA. Leave disabled for BF16 LoRA on an A6000 48GB.",
    )
    parser.add_argument(
        "--predict-test",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    return parser.parse_args()


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


def build_splits(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = args.data_root / args.train_csv
    base_df = pd.read_csv(train_path)
    require_columns(base_df, TRAIN_COLUMNS, str(train_path))
    base_df = normalize_answer_column(base_df, str(train_path))

    if base_df["id"].duplicated().any():
        raise ValueError("train.csv must have one original row per image ID")

    base_train_df, val_df = train_test_split(
        base_df,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=base_df["answer"],
    )
    base_train_df = base_train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    if not args.use_augmentation:
        return base_train_df, val_df

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
            f"{unknown_ids[:5]}"
        )

    # train_aug.csv in this repository contains both source and synthetic rows.
    # Remove exact copies of original rows, then attach only synthetic siblings
    # belonging to training IDs. This also works if aug-csv is synthetic-only.
    original_keys = exact_row_keys(base_df)
    aug_keys = exact_row_keys(aug_df)
    synthetic_df = aug_df.loc[~aug_keys.isin(original_keys)].copy()
    train_ids = set(base_train_df["id"].astype(str))
    synthetic_train_df = synthetic_df[
        synthetic_df["id"].astype(str).isin(train_ids)
    ].copy()

    train_df = pd.concat(
        [base_train_df, synthetic_train_df], ignore_index=True
    ).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    leaked = set(train_df["id"].astype(str)) & set(val_df["id"].astype(str))
    if leaked:
        raise AssertionError(f"image leakage detected for {len(leaked)} IDs")

    print(
        f"Data split: {len(base_train_df):,} original train + "
        f"{len(synthetic_train_df):,} synthetic train = {len(train_df):,}; "
        f"{len(val_df):,} untouched validation"
    )
    return train_df, val_df


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
        # Decoder-only generation needs left padding, while prompt-prefix label
        # masking during training is simplest and safest with right padding.
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
                full_messages = prompt_messages + [
                    assistant_message(sample["answer"])
                ]
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
                "answers": [sample.get("answer") for sample in samples],
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

        # Right padding is enforced in main(). The prompt-only encoding must be
        # an exact prefix of the full conversation; otherwise masking could
        # silently train on prompt tokens or hide the answer.
        for index in range(len(samples)):
            prompt_len = int(prompt_encoded["attention_mask"][index].sum())
            prompt_ids = prompt_encoded["input_ids"][index, :prompt_len]
            full_prefix = encoded["input_ids"][index, :prompt_len]
            if not torch.equal(prompt_ids, full_prefix):
                raise RuntimeError(
                    "chat-template prefix mismatch; update the masking logic "
                    "for the installed Transformers version"
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


def parse_answer(text: str) -> str | None:
    cleaned = text.strip().lower()
    exact = re.fullmatch(r"[\s\(\[]*([abcd])[\s\)\].,!]*", cleaned)
    if exact:
        return exact.group(1)
    bounded = re.search(r"(?<![a-z])([abcd])(?![a-z])", cleaned)
    return bounded.group(1) if bounded else None


@torch.inference_mode()
def predict_loader(
    model: Any,
    processor: Any,
    loader: DataLoader,
    device: torch.device,
    max_new_tokens: int,
    description: str,
) -> tuple[list[str], list[str | None], list[str | None]]:
    model.eval()
    previous_use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = True
    answer_token_ids: list[int] = []
    for label in LABELS:
        token_ids = processor.tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise RuntimeError(
                f"answer label {label!r} is not a single tokenizer token: "
                f"{token_ids}"
            )
        answer_token_ids.append(token_ids[0])
    eos_token_id = processor.tokenizer.eos_token_id
    if eos_token_id is None:
        raise RuntimeError("tokenizer has no EOS token")
    all_ids: list[str] = []
    all_predictions: list[str | None] = []
    all_answers: list[str | None] = []

    for batch in tqdm(loader, desc=description, leave=False):
        ids = batch.pop("ids")
        answers = batch.pop("answers")
        inputs = move_to_device(batch["model_inputs"], device)
        input_length = inputs["input_ids"].shape[1]

        def allowed_tokens(_batch_id: int, generated_ids: torch.Tensor) -> list[int]:
            generated_length = generated_ids.shape[-1] - input_length
            return answer_token_ids if generated_length == 0 else [eos_token_id]

        generated = model.generate(
            **inputs,
            max_new_tokens=max(2, max_new_tokens),
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
            prefix_allowed_tokens_fn=allowed_tokens,
        )
        decoded = processor.batch_decode(
            generated[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        predictions = [parse_answer(text) for text in decoded]
        all_ids.extend(ids)
        all_predictions.extend(predictions)
        all_answers.extend(answers)

    model.config.use_cache = previous_use_cache
    return all_ids, all_predictions, all_answers


def evaluate(
    model: Any,
    processor: Any,
    loader: DataLoader,
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, float]:
    _, predictions, answers = predict_loader(
        model,
        processor,
        loader,
        device,
        max_new_tokens,
        description="validation",
    )
    valid_pairs = [
        (prediction, answer)
        for prediction, answer in zip(predictions, answers)
        if prediction is not None and answer is not None
    ]
    correct = sum(prediction == answer for prediction, answer in valid_pairs)
    total = len(answers)
    invalid = total - len(valid_pairs)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": float(correct),
        "total": float(total),
        "invalid_predictions": float(invalid),
    }


def cpu_adapter_state(model: Any) -> dict[str, torch.Tensor]:
    state = get_peft_model_state_dict(model)
    return {key: value.detach().cpu().clone() for key, value in state.items()}


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
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
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

    train_df, val_df = build_splits(args)
    test_path = args.data_root / args.test_csv
    test_df = pd.read_csv(test_path)
    require_columns(test_df, TEST_COLUMNS, str(test_path))
    verify_image_paths(
        (train_df, val_df, test_df),
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
        print("Loading BF16 base model (recommended for A6000 48GB)")

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
    val_dataset = VQADataset(val_df, args.data_root, has_answer=True)
    test_dataset = VQADataset(test_df, args.data_root, has_answer=False)
    train_loader = build_dataloader(
        train_dataset,
        processor,
        train=True,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
    )
    val_loader = build_dataloader(
        val_dataset,
        processor,
        train=False,
        batch_size=args.eval_batch_size,
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
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    config_to_save = vars(args).copy()
    config_to_save["data_root"] = str(args.data_root)
    config_to_save["output_dir"] = str(args.output_dir)
    config_to_save["gpu"] = gpu_name
    config_to_save["vram_gib"] = total_vram
    save_json(args.output_dir / "run_config.json", config_to_save)

    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    global_update = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.config.use_cache = False
        running_loss = 0.0
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"train epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(progress, start=1):
            batch = move_to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                raw_loss = outputs.loss
                loss = raw_loss / args.grad_accum_steps
            loss.backward()
            running_loss += float(raw_loss.detach())
            epoch_loss += float(raw_loss.detach())

            should_update = (
                step % args.grad_accum_steps == 0 or step == len(train_loader)
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

                if global_update % args.log_every == 0:
                    mean_loss = running_loss / (
                        args.log_every * args.grad_accum_steps
                    )
                    progress.set_postfix(
                        loss=f"{mean_loss:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )
                    running_loss = 0.0

        metrics = evaluate(
            model,
            processor,
            val_loader,
            device,
            max_new_tokens=args.max_new_tokens,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": epoch_loss / len(train_loader),
            **metrics,
            "elapsed_minutes": (time.time() - started) / 60,
        }
        history.append(epoch_record)
        save_json(args.output_dir / "metrics.json", history)
        print(
            f"Epoch {epoch}: loss={epoch_record['train_loss']:.4f}, "
            f"val_acc={metrics['accuracy']:.4f}, "
            f"invalid={int(metrics['invalid_predictions'])}"
        )

        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_state = cpu_adapter_state(model)
            best_dir = args.output_dir / "best_adapter"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir, safe_serialization=True)
            processor.save_pretrained(best_dir)
            save_json(best_dir / "validation_metrics.json", epoch_record)
            print(f"Saved new best adapter to {best_dir}")

    if best_state is None:
        raise RuntimeError("training completed without a best adapter state")
    set_result = set_peft_model_state_dict(model, best_state)
    if getattr(set_result, "unexpected_keys", None):
        raise RuntimeError(
            f"unexpected adapter keys while restoring best state: "
            f"{set_result.unexpected_keys}"
        )
    model.config.use_cache = True

    if args.predict_test:
        test_loader = build_dataloader(
            test_dataset,
            processor,
            train=False,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
        )
        ids, predictions, _ = predict_loader(
            model,
            processor,
            test_loader,
            device,
            max_new_tokens=args.max_new_tokens,
            description="test prediction",
        )
        invalid_count = sum(prediction is None for prediction in predictions)
        if invalid_count:
            bad_ids = [
                item_id
                for item_id, prediction in zip(ids, predictions)
                if prediction is None
            ][:10]
            raise RuntimeError(
                f"test produced {invalid_count} invalid answers; "
                f"first IDs: {bad_ids}"
            )
        prediction_df = pd.DataFrame(
            {"id": ids, "answer": [str(item) for item in predictions]}
        )
        sample_path = args.data_root / args.sample_submission
        if sample_path.exists():
            sample_df = pd.read_csv(sample_path)
            require_columns(sample_df, ("id", "answer"), str(sample_path))
            submission_df = sample_df[["id"]].merge(
                prediction_df,
                on="id",
                how="left",
                validate="one_to_one",
            )
            if submission_df["answer"].isna().any():
                raise RuntimeError("some sample-submission IDs have no prediction")
        else:
            submission_df = prediction_df
        submission_path = args.output_dir / "submission_qwen35_4b.csv"
        submission_df.to_csv(submission_path, index=False, encoding="utf-8-sig")
        print(f"Saved submission: {submission_path}")

    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Total elapsed: {(time.time() - started) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
