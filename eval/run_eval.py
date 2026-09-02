"""Run evaluation for one system variant, with checkpointing so a Colab
disconnect doesn't force re-running the official test set from scratch.
"""

import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_questions(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_checkpoint(output_path: Path) -> dict:
    """Resume from an existing predictions file, keyed by question id."""
    if not output_path.exists():
        return {}
    done = {}
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            done[row["id"]] = row
    return done


def run_single_question(question: dict, config: dict) -> dict:
    # TODO: wire this up to agents/orchestrator.py once the agent pipeline exists.
    raise NotImplementedError("Implement the call into your agent pipeline here.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--official", action="store_true", help="Run on the official test set (use ONCE).")
    args = parser.parse_args()

    config = load_config(args.config)
    data_path = config["data"]["test_path"] if args.official else config["data"]["dev_path"]
    questions = load_questions(data_path)

    output_path = Path(config["eval"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_checkpoint(output_path)

    checkpoint_every = config["eval"].get("checkpoint_every", 20)
    with open(output_path, "a", encoding="utf-8") as f:
        for i, q in enumerate(tqdm(questions)):
            if q["id"] in done:
                continue
            result = run_single_question(q, config)
            f.write(json.dumps(result) + "\n")
            if (i + 1) % checkpoint_every == 0:
                f.flush()

    print(f"Done. Predictions written to {output_path}")


if __name__ == "__main__":
    main()

