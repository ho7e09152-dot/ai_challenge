"""Create a conservative text-only augmentation of train.csv.

Each source row is retained and receives one synthetic sibling row that:
1. points to the same image,
2. asks a semantically equivalent question,
3. contains exactly the same four answer choices in a shuffled order, and
4. updates the answer label to the new location of the correct choice.

No image is inspected and no new visual fact is introduced.
"""

from __future__ import annotations

import csv
import hashlib
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "train.csv"
OUTPUT = ROOT / "train_aug.csv"
FIELDS = ["id", "path", "question", "a", "b", "c", "d", "answer"]
LABELS = ("a", "b", "c", "d")


# Every replacement below preserves the referent or the question intent.
# Longer/specific expressions come before shorter/general expressions.
LOCATION_REWRITES = (
    ("사진 속에서", "사진에서"),
    ("사진 속에 보이는", "사진에 보이는"),
    ("사진 속에 있는", "사진에 있는"),
    ("사진 속의", "사진에 보이는"),
    ("사진 속에", "사진에"),
    ("사진에서 보이는", "사진 속"),
    ("사진에 보이는", "사진 속"),
    ("사진에 있는", "사진 속의"),
    ("사진 속", "사진에 보이는"),
)

ENDING_REWRITES = (
    ("몇 개입니까?", "몇 개인가요?"),
    ("몇 개인가요?", "몇 개입니까?"),
    ("무엇인가요?", "무엇입니까?"),
    ("무엇입니까?", "무엇인가요?"),
    ("어디인가요?", "어디입니까?"),
    ("어디입니까?", "어디인가요?"),
    ("있나요?", "있습니까?"),
    ("있습니까?", "있나요?"),
    ("보이나요?", "보입니까?"),
    ("보입니까?", "보이나요?"),
    ("만들어졌나요?", "만들어졌습니까?"),
    ("만들어졌습니까?", "만들어졌나요?"),
    ("얼마인가요?", "얼마입니까?"),
    ("얼마입니까?", "얼마인가요?"),
    ("종류인가요?", "종류입니까?"),
    ("종류입니까?", "종류인가요?"),
    ("것인가요?", "것입니까?"),
    ("것입니까?", "것인가요?"),
    ("해야 할까요?", "해야 합니까?"),
    ("해야 하나요?", "해야 합니까?"),
    ("음료입니까?", "음료인가요?"),
)


def replace_once(text: str, rewrites: tuple[tuple[str, str], ...]) -> str:
    for old, new in rewrites:
        if old in text:
            return text.replace(old, new, 1)
    return text


def paraphrase_question(question: str) -> str:
    """Apply only controlled, meaning-preserving Korean rewrites."""
    original = re.sub(r"\s+", " ", str(question)).strip()
    changed = replace_once(original, LOCATION_REWRITES)
    changed = replace_once(changed, ENDING_REWRITES)

    # Questions without one of the common photo phrases still get a harmless
    # surface-form change. '사진' and '이미지' are equivalent in this dataset.
    if changed == original and "사진" in changed:
        changed = changed.replace("사진", "이미지", 1)
    elif changed == original and "이미지" in changed:
        changed = changed.replace("이미지", "사진", 1)

    # Final fallback keeps the complete original question verbatim and adds no
    # proposition. This is preferable to inventing an unsupported visual fact.
    if changed == original:
        changed = f"다음 질문에 답해 주세요. {original}"

    return changed


def stable_rng(row: dict[str, str], index: int) -> random.Random:
    payload = f"{index}|{row['id']}|{row['question']}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return random.Random(seed)


def augment_row(row: dict[str, str], index: int) -> dict[str, str]:
    answer = row["answer"].strip().lower()
    if answer not in LABELS:
        raise ValueError(f"row {index}: invalid answer label {answer!r}")

    correct_text = row[answer]
    choice_values = [row[label] for label in LABELS]
    order = list(range(4))
    rng = stable_rng(row, index)

    # Guarantee a changed order instead of relying on chance.
    while order == [0, 1, 2, 3]:
        rng.shuffle(order)

    shuffled = [choice_values[i] for i in order]
    correct_source_index = LABELS.index(answer)
    new_answer = LABELS[order.index(correct_source_index)]

    augmented = dict(row)
    augmented["question"] = paraphrase_question(row["question"])
    for label, value in zip(LABELS, shuffled):
        augmented[label] = value
    augmented["answer"] = new_answer

    if augmented[new_answer] != correct_text:
        raise AssertionError(f"row {index}: correct answer text was not preserved")
    return augmented


def read_source() -> list[dict[str, str]]:
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(
                f"unexpected columns: {reader.fieldnames!r}; expected {FIELDS!r}"
            )
        return list(reader)


def main() -> None:
    rows = read_source()
    augmented_rows = [augment_row(row, i) for i, row in enumerate(rows)]

    # Interleave source and synthetic rows so each pair is easy to inspect.
    combined: list[dict[str, str]] = []
    for source, augmented in zip(rows, augmented_rows):
        combined.extend((source, augmented))

    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(combined)

    print(f"source rows:    {len(rows):,}")
    print(f"synthetic rows: {len(augmented_rows):,}")
    print(f"output rows:    {len(combined):,}")
    print(f"saved to:       {OUTPUT}")


if __name__ == "__main__":
    main()
