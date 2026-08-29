from pathlib import Path
import pandas as pd
import numpy as np

PROB_DIR = Path("outputs/qwen35_9b_fulltrain/probabilities")
OUT_DIR = Path("outputs/qwen35_9b_fulltrain/submissions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["a", "b", "c", "d"]
PROB_COLS = ["p_a", "p_b", "p_c", "p_d"]


def load_probs(epoch):
    path = PROB_DIR / f"probabilities_epoch_{epoch}.csv"
    df = pd.read_csv(path)

    print(path)
    print(df.columns.tolist())

    required = ["id", *PROB_COLS]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{path}에 필요한 컬럼이 없습니다: {missing}"
        )

    return df


p10 = load_probs("1.0")
p15 = load_probs("1.5")
p20 = load_probs("2.0")


def make_ensemble(name, models):
    """
    models = [
        (dataframe, weight),
        ...
    ]
    """

    base_df = models[0][0].copy()

    # 모든 probability 파일의 ID 순서가 같은지 검사
    base_ids = base_df["id"].astype(str).tolist()

    for df, _ in models[1:]:
        current_ids = df["id"].astype(str).tolist()

        if current_ids != base_ids:
            raise ValueError(
                "Probability CSV들의 id 순서가 서로 다릅니다."
            )

    total_weight = sum(weight for _, weight in models)

    if total_weight <= 0:
        raise ValueError("ensemble weight 합은 0보다 커야 합니다.")

    ensemble_probs = np.zeros(
        (len(base_df), len(LABELS)),
        dtype=np.float64
    )

    for df, weight in models:
        ensemble_probs += (
            df[PROB_COLS].to_numpy(dtype=np.float64)
            * weight
        )

    ensemble_probs /= total_weight

    pred_idx = ensemble_probs.argmax(axis=1)

    submission = pd.DataFrame({
        "id": base_df["id"],
        "answer": [LABELS[i] for i in pred_idx]
    })

    output_path = OUT_DIR / f"submission_{name}.csv"

    submission.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nSaved: {output_path}")

    print("Answer distribution:")
    print(submission["answer"].value_counts().sort_index())

    return submission


# =========================================================
# 1순위: 1.5 epoch 중심
# =========================================================

make_ensemble(
    "ensemble_1.0_1.5_2.0_w20_60_20",
    [
        (p10, 0.20),
        (p15, 0.60),
        (p20, 0.20),
    ]
)


# =========================================================
# 2순위: 1.0 + 1.5
# =========================================================

make_ensemble(
    "ensemble_1.0_1.5_w25_75",
    [
        (p10, 0.25),
        (p15, 0.75),
    ]
)


# =========================================================
# 3순위: 1.5 + 2.0
# =========================================================

make_ensemble(
    "ensemble_1.5_2.0_w75_25",
    [
        (p15, 0.75),
        (p20, 0.25),
    ]
)


# =========================================================
# 추가 후보: 세 checkpoint 동일 평균
# =========================================================

make_ensemble(
    "ensemble_1.0_1.5_2.0_equal",
    [
        (p10, 1.0),
        (p15, 1.0),
        (p20, 1.0),
    ]
)