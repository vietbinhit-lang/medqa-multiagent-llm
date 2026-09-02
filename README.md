# MedQA Multi-Agent LLM System

Midterm project: a medical multi-agent LLM system for factual accuracy on
MedQA-USMLE. The project evaluates system design (baseline vs. RAG vs.
multi-agent vs. full pipeline), not clinical deployment.

## Repository structure

- `configs/` - YAML configs per variant (V0-V4): model, temperature, seed,
  retrieval settings
- `agents/` - agent implementations (reasoning agent, verifier, orchestrator)
- `rag/` - retrieval module (index builder, retriever, sample + real corpus prep)
- `memory/` - short-term and long-term memory
- `eval/` - evaluation pipeline (run_eval.py, metrics.py)
- `data/` - sample_dev.jsonl (bundled, original) + prepare_medqa.py for the
  real dev/test splits - official test set is NOT stored here
- `predictions/` - output predictions per variant, per run
- `results/` - leaderboard, paired comparison, error analysis tables
- `report/` - midterm report source
- `notebooks/` - colab_runner.ipynb - clones this repo and runs eval on Colab
- `streamlit_app/` - interactive Q&A demo + evaluation dashboard, deployable
  on Streamlit Community Cloud (see `streamlit_app/README.md`)

## System variants

| Variant | Description |
| --- | --- |
| V0 | Direct LLM baseline |
| V1 | RAG-only |
| V2 | Multi-agent without memory |
| V3 | Full system (agents + RAG + memory + verifier) |
| V4 | Full system without verifier (optional) |

## Quickstart

```bash
git clone https://github.com/vietbinhit-lang/medqa-multiagent-llm.git
cd medqa-multiagent-llm
pip install -r requirements.txt
python eval/run_eval.py --config configs/v0_baseline.yaml
```

On Colab: open `notebooks/colab_runner.ipynb` and run the cells in order.

By default, `configs/v0_baseline.yaml` and `configs/v3_full.yaml` point at
the bundled `data/sample_dev.jsonl` (20 original questions) and
`rag/corpus/sample/` (20 original passages), so the quickstart above runs
end-to-end with zero downloads. See `data/README.md` and `rag/README.md`
for how to swap in the real MedQA-USMLE dataset and a larger reference
corpus once you're ready for an actual evaluation.

## Interactive demo (Streamlit)

`streamlit_app/` ships a web app with two tabs: an interactive Q&A demo
(pick a variant + question, see the full agent trace) and a dashboard that
runs all 5 variants over the sample set and compares accuracy. It never
serves an LLM itself - you paste in the base URL, model name, and API key
of any OpenAI-compatible endpoint (a tunneled local vLLM/Ollama server, or
a hosted free-tier provider like Groq/OpenRouter/Together). Run it locally
with `streamlit run streamlit_app/app.py`, or deploy it straight from this
repo on [Streamlit Community Cloud](https://share.streamlit.io) - see
`streamlit_app/README.md` for details.

## Evaluation protocol

- Dev set (100-150 questions): debugging and tuning only.
- Official MedQA-USMLE test set (1273 questions): run once for final
  results, never used for tuning.
- Metrics: accuracy, invalid response rate, accuracy gain (V3-V0), ablation
  comparison, win/loss/tie analysis, bootstrap confidence interval,
  McNemar test, cost and latency.

## Local LLM, no external API

This project runs an open-weight LLM locally (via vLLM or Ollama) exposed
through an OpenAI-compatible endpoint, so the agent code stays
provider-agnostic. See `agents/README.md` for serving instructions.

## Status

- Multi-agent orchestrator (reasoning + critic + verifier agents), RAG
  retriever, and short/long-term memory are implemented and verified with
  mocked-LLM smoke tests (see `agents/README.md`).
- All 5 variant configs (V0-V4) exist under `configs/` and were verified to
  activate the correct agent/component combination.
- Bundled sample data (`data/sample_dev.jsonl`) and sample RAG corpus
  (`rag/corpus/sample/`) let every variant run end-to-end with zero
  downloads; `data/prepare_medqa.py` and `rag/prepare_corpus.py` build the
  real dataset and a larger corpus when you have internet access.
- `streamlit_app/` (interactive demo + eval dashboard) is implemented and
  verified with `streamlit.testing.v1.AppTest` across all 5 variants.
- Still to do: run V0-V4 on the official test set with a served local LLM,
  and fill in `results/` and `report/`.

Midterm project for mon hoc cua thay Le Kim Hung.
