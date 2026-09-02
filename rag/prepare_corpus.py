"""Build a larger RAG corpus of medical reference passages.

This repo ships a small, original `rag/corpus/sample/` (20 short passages,
one per topic in `data/sample_dev.jsonl`) so the RAG pipeline is
smoke-testable with zero downloads. That corpus is deliberately narrow -
it exists to prove the retrieval code path works, not to give the system
broad medical coverage.

For a real evaluation you want a much larger, general-purpose medical
reference corpus (e.g. Wikipedia medical articles, MedlinePlus topics, or
a licensed textbook you have permission to use). This script fetches a
list of Wikipedia articles by title and writes each one as a `.txt` file,
so you get a bigger corpus without hand-authoring passages.

Usage (with real internet access, e.g. in Colab):

    pip install wikipedia-api
    python rag/prepare_corpus.py --out-dir rag/corpus/wikipedia --topics-file rag/wikipedia_topics.txt

`--topics-file` is a plain text file, one article title per line (see
`rag/wikipedia_topics.txt` for a starter list of ~40 general medical
topics). If you have your own corpus already as a folder of .txt files
(e.g. exported from a textbook you're licensed to use), you don't need
this script at all - just point `rag.corpus_path` in the config at that
folder directly, since `rag/retriever.py` indexes every `*.txt` file
under `corpus_path` recursively.
"""

import argparse
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOPICS_FILE = REPO_ROOT / "rag" / "wikipedia_topics.txt"


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title.strip().lower()).strip("_")
    return slug or "untitled"


def fetch_wikipedia_articles(topics: list, lang: str = "en"):
    """Yield (title, text) for each topic, skipping ones that fail to fetch."""
    try:
        import wikipediaapi
    except ImportError as exc:
        raise RuntimeError(
            "The 'wikipedia-api' package is required. "
            "Install it with: pip install wikipedia-api"
        ) from exc

    wiki = wikipediaapi.Wikipedia(
        language=lang,
        user_agent="medqa-multiagent-llm-corpus-builder/1.0",
    )
    for title in topics:
        title = title.strip()
        if not title or title.startswith("#"):
            continue
        page = wiki.page(title)
        if not page.exists():
            print(f"  [skip] '{title}' not found on Wikipedia")
            continue
        yield title, page.text
        time.sleep(0.2)  # be polite to the API


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "rag" / "corpus" / "wikipedia"))
    parser.add_argument("--topics-file", default=str(DEFAULT_TOPICS_FILE))
    parser.add_argument("--lang", default="en")
    parser.add_argument("--max-chars-per-doc", type=int, default=20000,
                         help="Truncate very long articles to this many characters.")
    args = parser.parse_args()

    topics_path = Path(args.topics_file)
    if not topics_path.exists():
        print(f"Topics file not found: {topics_path}", file=sys.stderr)
        return 1
    topics = topics_path.read_text(encoding="utf-8").splitlines()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for title, text in fetch_wikipedia_articles(topics, lang=args.lang):
        text = text[: args.max_chars_per_doc]
        out_path = out_dir / f"{_slugify(title)}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"  [ok] wrote {out_path}")
        count += 1

    print(f"\nDone. Wrote {count} article(s) to {out_dir}")
    print(
        "Point rag.corpus_path in configs/*.yaml at this folder "
        "(or merge it with rag/corpus/sample/) once you're ready to move "
        "off the bundled sample corpus."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
