"""Download and normalize the official MedQA-USMLE (4-option) dataset.

This repo ships a small, original `sample_dev.jsonl` (20 self-authored
questions) so the pipeline can be smoke-tested with zero downloads. That
file is NOT the official benchmark and must not be used for any reported
results. Run this script (with real internet access, e.g. in Colab) to
fetch the authentic dataset and produce the files the configs expect:

    data/dev_150.jsonl        - a fixed-size sample of the train/dev split,
                                 for prompt tuning and error analysis only
    data/official_test.jsonl - the full official test split (1273 Qs),
                                 to be run exactly once for reported numbers

Usage:
    python data/prepare_medqa.py --dev-size 150 --seed 42

The script tries a couple of known Hugging Face mirrors of the dataset,
since the original MedQA release (Jin et al., 2020) is distributed as a
Google Drive archive with no stable direct-download URL. If your
environment has a local copy of the official files instead (e.g. you
downloaded the Google Drive archive by hand), point --local-train /
--local-test at the extracted JSONL files and this script will just
normalize + re-split them.

Citation:
    Jin, D., Pan, E., Oufattole, N., Weng, W.H., Fang, H., & Szolovits, P.
    (2020). What Disease does this Patient Have? A Large-scale Open Domain
    Question Answering Dataset from Medical Exams. arXiv:2009.13081.
"""

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Known Hugging Face mirrors of MedQA-USMLE-4-options, tried in order.
# Different mirrors use different column names, hence _normalize_row below.
HF_CANDIDATES = [
    ("GBaker/MedQA-USMLE-4-options", None),
    ("bigbio/med_qa", "med_qa_en_4options_source"),
]


def _normalize_row(idx: int, row: dict, prefix: str) -> dict:
    """Map a HF-dataset row (schema varies by mirror) to this repo's schema:
    {"id", "question", "options": {"A".."D"}, "answer"}.
    """
    # GBaker/MedQA-USMLE-4-options: {"question", "answer", "options": {...}}
    # where "answer" may be the letter or the answer text.
    if "options" in row and isinstance(row["options"], dict):
        options = {k.upper(): v for k, v in row["options"].items()}
        answer = row.get("answer", row.get("answer_idx", ""))
        answer = str(answer).strip().upper()
        if answer not in options:
            # answer given as full text instead of a letter - look it up.
            for letter, text in options.items():
                if str(text).strip().lower() == str(answer).strip().lower():
                    answer = letter
                    break
        return {
            "id": f"{prefix}_{idx:05d}",
            "question": row.get("question", "").strip(),
            "options": {k: options[k] for k in ["A", "B", "C", "D"] if k in options},
            "answer": answer,
        }

    # bigbio/med_qa (med_qa_en_4options_source schema):
    # {"question", "answer_idx", "options": [{"key": "A", "value": "..."}]}
    if "choices" in row or "answer_idx" in row:
        opts_list = row.get("options") or row.get("choices") or []
        options = {}
        for opt in opts_list:
            if isinstance(opt, dict):
                key = str(opt.get("key", "")).strip().upper()
                val = opt.get("value", "")
            else:
                key, val = opt
            if key:
                options[key] = val
        answer = str(row.get("answer_idx", row.get("answer", ""))).strip().upper()
        return {
            "id": f"{prefix}_{idx:05d}",
            "question": row.get("question", "").strip(),
            "options": {k: options[k] for k in ["A", "B", "C", "D"] if k in options},
            "answer": answer,
        }

    raise ValueError(f"Unrecognized row schema, cannot normalize: {list(row.keys())}")


def _valid(row: dict) -> bool:
    return (
        set(row.get("options", {}).keys()) == {"A", "B", "C", "D"}
        and row.get("answer") in {"A", "B", "C", "D"}
        and bool(row.get("question"))
    )


def load_from_huggingface():
    """Try each known mirror in turn; return (train_rows, test_rows) of
    normalized dicts, or raise RuntimeError if none of them work.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required to fetch MedQA-USMLE. "
            "Install it with: pip install datasets"
        ) from exc

    errors = []
    for name, config in HF_CANDIDATES:
        try:
            ds = load_dataset(name, config) if config else load_dataset(name)
            train_split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
            test_split = ds["test"] if "test" in ds else train_split

            train_rows = [
                _normalize_row(i, r, "train") for i, r in enumerate(train_split)
            ]
            test_rows = [
                _normalize_row(i, r, "test") for i, r in enumerate(test_split)
            ]
            train_rows = [r for r in train_rows if _valid(r)]
            test_rows = [r for r in test_rows if _valid(r)]
            if train_rows and test_rows:
                print(f"Loaded MedQA-USMLE from Hugging Face dataset '{name}'.")
                return train_rows, test_rows
        except Exception as exc:  # noqa: BLE001 - we want to try the next mirror
            errors.append(f"{name}: {exc}")

    raise RuntimeError(
        "Could not load MedQA-USMLE from any known Hugging Face mirror. "
        "Tried:\n  " + "\n  ".join(errors) + "\n"
        "If you have the official files locally (from the Google Drive "
        "archive released by Jin et al. 2020), re-run with --local-train "
        "and --local-test pointing at the extracted train/test JSONL files."
    )


def load_from_local(train_path: str, test_path: str):
    def _load(path, prefix):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                rows.append(_normalize_row(i, raw, prefix))
        return [r for r in rows if _valid(r)]

    return _load(train_path, "train"), _load(test_path, "test")


def write_jsonl(rows: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-size", type=int, default=150,
                         help="Number of questions to sample for the dev set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-train", default=None,
                         help="Path to a local official train/dev JSONL file, "
                              "used instead of downloading from Hugging Face.")
    parser.add_argument("--local-test", default=None,
                         help="Path to a local official test JSONL file, "
                              "used instead of downloading from Hugging Face.")
    parser.add_argument("--dev-out", default=str(REPO_ROOT / "data" / "dev_150.jsonl"))
    parser.add_argument("--test-out", default=str(REPO_ROOT / "data" / "official_test.jsonl"))
    args = parser.parse_args()

    if args.local_train and args.local_test:
        train_rows, test_rows = load_from_local(args.local_train, args.local_test)
    else:
        train_rows, test_rows = load_from_huggingface()

    rng = random.Random(args.seed)
    dev_rows = rng.sample(train_rows, min(args.dev_size, len(train_rows)))

    write_jsonl(dev_rows, Path(args.dev_out))
    write_jsonl(test_rows, Path(args.test_out))

    print(
        "\nDone. Point configs/*.yaml at these files "
        "(data.dev_path / data.test_path) once you're ready to move off "
        "the bundled sample_dev.jsonl."
    )


if __name__ == "__main__":
    sys.exit(main())
