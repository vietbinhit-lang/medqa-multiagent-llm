# MedQA Multi-Agent LLM System

Midterm project: a medical multi-agent LLM system for factual accuracy on MedQA-USMLE. The project evaluates system design (baseline vs. RAG vs. multi-agent vs. full pipeline), not clinical deployment.

## Repository structure

configs/      YAML configs per variant (V0-V4): model, temperature, seed, retrieval settings
agents/       agent implementations (reasoning agent, verifier, orchestrator)
rag/          retrieval module (index builder, retriever)
memory/       short-term and long-term memory
eval/         evaluation pipeline (run_eval.py, metrics.py)
data/         dev split only (100-150 questions) - official test set is NOT stored here
predictions/  output predictions per variant, per run
results/      leaderboard, paired comparison, error analysis tables
report/       midterm report source
notebooks/    colab_runner.ipynb - clones this repo and runs eval on Colab

## System variants

| Variant | Description |
|---|---|
| V0 | Direct LLM baseline |
| V1 | RAG-only |
| V2 | Multi-agent without memory |
| V3 | Full system (agents + RAG + memory + verifier) |
| V4 | Full system without verifier (optional) |

## Quickstart

    git clone https://github.com/vietbinhit-lang/medqa-multiagent-llm.git
    cd medqa-multiagent-llm
    pip install -r requirements.txt
    python eval/run_eval.py --config configs/v0_baseline.yaml

On Colab: open notebooks/colab_runner.ipynb and run the cells in order.

## Evaluation protocol

- Dev set (100-150 questions): debugging and tuning only.
- Official MedQA-USMLE test set (1273 questions): run once for final results, never used for tuning.
- Metrics: accuracy, invalid response rate, accuracy gain (V3-V0), ablation comparison, win/loss/tie analysis, bootstrap confidence interval, McNemar test, cost and latency.

## Local LLM, no external API

This project runs an open-weight LLM locally (via vLLM or Ollama) exposed through an OpenAI-compatible endpoint, so the agent code stays provider-agnostic. See agents/README.md for serving instructions.

## Status

Work in progress - midterm project for mon hoc cua thay Le Kim Hung.
