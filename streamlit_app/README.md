# Streamlit app

A demo + evaluation dashboard for the MedQA multi-agent system, deployable
on [Streamlit Community Cloud](https://streamlit.io/cloud).

## What it does

- **Hỏi đáp trực tiếp (interactive Q&A):** pick any of the 5 variants
  (V0-V4), pick one of the 20 bundled sample questions (or type your own),
  run it, and see the final answer plus a step-by-step trace of every
  agent that fired (reasoning → critic → verifier).
- **Dashboard đánh giá (evaluation dashboard):** run one or more variants
  across all 20 sample questions and compare accuracy / invalid-response
  rate, with a bar chart and a downloadable JSON of raw predictions.

## Why you must supply your own LLM endpoint

Streamlit Community Cloud runs on CPU-only, memory-limited containers -
it cannot host vLLM or a 7B local model itself. The sidebar instead asks
for any **OpenAI-compatible** chat endpoint:

- A local vLLM or Ollama server you expose publicly (e.g. with
  [ngrok](https://ngrok.com) or [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)).
- A hosted provider with an OpenAI-compatible API and a free tier, such as
  [Groq](https://console.groq.com), [OpenRouter](https://openrouter.ai), or
  [Together AI](https://www.together.ai).

Nothing is hardcoded: base URL, model name, and API key are all entered at
runtime in the sidebar and never committed to the repo.

## Run locally

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

## Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
2. "New app" → pick this repo, branch `main`, main file path
   `streamlit_app/app.py`.
3. Deploy. Streamlit Cloud installs `streamlit_app/requirements.txt`
   automatically because it lives next to `app.py`.
4. Once the app is up, open it and enter your LLM endpoint details in the
   sidebar before running anything.

## Notes

- `rag/corpus/sample/` and `data/sample_dev.jsonl` are original,
  self-authored content bundled in the repo (see the root `data/README.md`
  and `rag/README.md`) - not the official MedQA-USMLE benchmark. The
  dashboard's accuracy numbers are for demoing the pipeline, not for
  reporting real results.
- The RAG retriever downloads the `BAAI/bge-small-en` embedding model on
  first use and caches it for the life of the app instance
  (`st.cache_resource`), so the first RAG-enabled run is slower than
  subsequent ones.
