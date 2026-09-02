# RAG corpus

## Files

| Path | Status | Purpose |
|---|---|---|
| `corpus/sample/*.txt` | **Bundled, original** | 20 short, self-authored reference passages, one per topic in `data/sample_dev.jsonl`. Lets `rag/retriever.py` be smoke-tested end-to-end with zero downloads. |
| `prepare_corpus.py` | Script | Fetches a larger corpus of Wikipedia medical articles (with real internet access) and writes them as `.txt` files. |
| `wikipedia_topics.txt` | Input list | Starter list of ~30 general medical topics used by `prepare_corpus.py`. Extend it for broader coverage. |

## How retrieval works

`rag/retriever.py`'s `Retriever` class globs every `*.txt` file under
`rag.corpus_path` (see `configs/*.yaml`), embeds each document with a
sentence-transformers model, and indexes the embeddings with a FAISS
`IndexFlatIP` index. At query time, the question is embedded the same way
and the top-`k` most similar documents are returned as context for the
reasoning agent.

Because retrieval works over whatever `.txt` files exist in
`corpus_path`, you can mix and match: point `rag.corpus_path` at
`rag/corpus/sample/` for quick local testing, at
`rag/corpus/wikipedia/` after running `prepare_corpus.py`, or at your own
folder of licensed reference material.

## Building a larger corpus

The bundled `corpus/sample/` is intentionally narrow (one short passage
per sample question) so it is not a meaningful test of retrieval quality
on unseen questions - it exists only to prove the RAG code path is
correct. For an actual evaluation, build a broader corpus with real
internet access (e.g. in Colab):

```bash
pip install wikipedia-api
python rag/prepare_corpus.py --out-dir rag/corpus/wikipedia --topics-file rag/wikipedia_topics.txt
```

Then update `rag.corpus_path` in `configs/v3_full.yaml` (and any other
variant using RAG) to point at `rag/corpus/wikipedia/`, or a folder
combining both the sample and Wikipedia passages.

If you have access to licensed medical textbooks or other reference
material, you can skip `prepare_corpus.py` entirely: just place `.txt`
files (one document per file, one topic per file works best for
retrieval granularity) in a folder and point `rag.corpus_path` at it.

## Licensing note

The bundled sample passages in `corpus/sample/` are original text written
for this project and are not copied from any textbook, exam prep
resource, or Wikipedia. Content fetched by `prepare_corpus.py` comes from
Wikipedia and is subject to Wikipedia's own licensing (CC BY-SA); check
the license of any other reference material (textbooks, licensed medical
databases) before including it in a corpus, especially if the repo or its
outputs will be shared publicly.
