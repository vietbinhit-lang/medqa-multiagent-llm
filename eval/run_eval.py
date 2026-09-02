"""Run evaluation for one system variant, with checkpointing so a Colab
disconnect doesn't force re-running the official test set from scratch.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

# Make the repo root importable regardless of the working directory this
# script is invoked from (e.g. `python eval/run_eval.py` from the repo
# root, as the README's quickstart shows) - eval/ is a sibling of
# agents/, memory/, and rag/, so it is not on sys.path by default.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.orchestrator import Orchestrator
from memory.memory import LongTermMemory
from rag.retriever import Retriever


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


def build_orchestrator(config: dict) -> Orchestrator:
    """Build the Orchestrator once per run so the RAG index and long-term
    memory are shared across all questions instead of rebuilt per question.
    """
    retriever = None
    if config["components"].get("rag"):
        rag_cfg = config.get("rag", {})
        retriever = Retriever(
            corpus_path=rag_cfg.get("corpus_path", "rag/corpus/"),
            embedding_model=rag_cfg.get("embedding_model", "BAAI/bge-small-en"),
            top_k=rag_cfg.get("top_k", 5),
        )
        retriever.build_index()

    long_term_memory = None
    if config["components"].get("long_term_memory"):
        long_term_memory = LongTermMemory()

    return Orchestrator(config, retriever=retriever, long_term_memory=long_term_memory)


def run_single_question(question: dict, orchestrator: Orchestrator) -> dict:
    return orchestrator.run(question)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--official", action="store_true", help="Run on the official test set (use ONCE).")
    args = parser.parse_args()

    config = load_config(args.config)
    data_path = config["data"]["test_path"] if args.official else config["data"]["dev_path"]
    questions = load_questions(data_path)

    orchestrator = build_orchestrator(config)

    output_path = Path(config["eval"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_checkpoint(output_path)

    checkpoint_every = config["eval"].get("checkpoint_every", 20)
    with open(output_path, "a", encoding="utf-8") as f:
        for i, q in enumerate(tqdm(questions)):
            if q["id"] in done:
                continue
            result = run_single_question(q, orchestrator)
            f.write(json.dumps(result) + "\n")
            if (i + 1) % checkpoint_every == 0:
                f.flush()

    print(f"Done. Predictions written to {output_path}")


if __name__ == "__main__":
    main()
