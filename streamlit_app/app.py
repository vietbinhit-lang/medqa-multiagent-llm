"""Streamlit demo + evaluation dashboard for the MedQA multi-agent system.

Streamlit Community Cloud has no GPU, so this app never serves an LLM
itself. Instead the sidebar lets the user point at any OpenAI-compatible
chat endpoint: a local vLLM/Ollama server exposed publicly (e.g. via
ngrok), or a hosted provider such as Groq / OpenRouter / Together that
offers an OpenAI-compatible API.

Run locally with:
    streamlit run streamlit_app/app.py
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# This file lives in streamlit_app/, a sibling of agents/, memory/, rag/,
# eval/ and configs/ at the repo root - add the repo root to sys.path so
# those packages import regardless of the working directory Streamlit was
# launched from (mirrors the same fix in eval/run_eval.py).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.orchestrator import Orchestrator  # noqa: E402
from eval.metrics import accuracy, invalid_response_rate  # noqa: E402
from memory.memory import LongTermMemory  # noqa: E402
from rag.retriever import Retriever  # noqa: E402

CONFIGS_DIR = REPO_ROOT / "configs"
SAMPLE_DATA_PATH = REPO_ROOT / "data" / "sample_dev.jsonl"

st.set_page_config(page_title="MedQA Multi-Agent LLM", page_icon="🩺", layout="wide")


# --------------------------------------------------------------------------
# Data / config loading
# --------------------------------------------------------------------------

@st.cache_data
def load_configs() -> dict:
    """Return {label: config_dict} for every configs/*.yaml, sorted by variant."""
    configs = {}
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        configs[cfg.get("variant", path.stem)] = cfg
    return configs


@st.cache_data
def load_sample_questions() -> list:
    if not SAMPLE_DATA_PATH.exists():
        return []
    with open(SAMPLE_DATA_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


VARIANT_DESCRIPTIONS = {
    "V0_direct_llm": "Baseline: 1 agent trả lời trực tiếp, không RAG, không bộ nhớ.",
    "V1_rag_only": "1 agent + truy hồi tài liệu (RAG), không multi-agent, không bộ nhớ.",
    "V2_multiagent_no_memory": "3 agent (reasoning → critic → verifier), không RAG, không bộ nhớ.",
    "V3_full_system": "Đầy đủ: RAG + 3 agent + bộ nhớ ngắn/dài hạn + verifier.",
    "V4_full_no_verifier": "Như V3 nhưng bỏ verifier, dùng luật hợp thức đơn giản thay thế.",
}


def components_badge(components: dict) -> str:
    on = [name for name, flag in components.items() if flag is True]
    return ", ".join(on) if on else "(không bật thành phần nào)"


# --------------------------------------------------------------------------
# Orchestrator construction (LLM settings come from the sidebar, not the
# YAML file, so the same bundled configs work with whatever endpoint the
# user has access to)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Đang tải mô hình embedding và build FAISS index cho RAG...")
def get_retriever(corpus_path: str, embedding_model: str, top_k: int) -> Retriever:
    retriever = Retriever(corpus_path=corpus_path, embedding_model=embedding_model, top_k=top_k)
    retriever.build_index()
    return retriever


def build_runtime_config(base_config: dict, llm_settings: dict) -> dict:
    """Overlay sidebar LLM settings onto a bundled config's model block."""
    cfg = json.loads(json.dumps(base_config))  # cheap deep copy
    cfg["model"]["base_url"] = llm_settings["base_url"]
    cfg["model"]["name"] = llm_settings["model_name"]
    cfg["model"]["temperature"] = llm_settings["temperature"]
    cfg["model"]["max_tokens"] = llm_settings["max_tokens"]
    return cfg


def build_orchestrator(cfg: dict, api_key: str) -> Orchestrator:
    import os

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    retriever = None
    if cfg["components"].get("rag"):
        rag_cfg = cfg.get("rag", {})
        retriever = get_retriever(
            corpus_path=str(REPO_ROOT / rag_cfg.get("corpus_path", "rag/corpus/sample/")),
            embedding_model=rag_cfg.get("embedding_model", "BAAI/bge-small-en"),
            top_k=rag_cfg.get("top_k", 5),
        )

    long_term_memory = LongTermMemory() if cfg["components"].get("long_term_memory") else None
    return Orchestrator(cfg, retriever=retriever, long_term_memory=long_term_memory)


# --------------------------------------------------------------------------
# Sidebar: LLM connection settings
# --------------------------------------------------------------------------

st.sidebar.title("⚙️ Kết nối LLM")
st.sidebar.caption(
    "Streamlit Community Cloud không có GPU nên app này không tự chạy "
    "vLLM/Ollama. Hãy trỏ tới một endpoint OpenAI-compatible: server cục bộ "
    "của bạn expose qua ngrok, hoặc một provider có free tier như Groq, "
    "OpenRouter, Together AI."
)

base_url = st.sidebar.text_input(
    "Base URL", value="http://localhost:8000/v1",
    help="Ví dụ: http://localhost:8000/v1 (vLLM/Ollama qua ngrok), "
         "hoặc https://api.groq.com/openai/v1",
)
model_name = st.sidebar.text_input("Tên model", value="qwen2.5-7b-instruct")
api_key = st.sidebar.text_input("API key (nếu cần)", value="", type="password")

with st.sidebar.expander("Tham số sinh văn bản"):
    temperature = st.slider("Temperature", 0.0, 1.5, 0.0, 0.1)
    max_tokens = st.slider("Max tokens", 64, 2048, 512, 64)

llm_settings = {
    "base_url": base_url,
    "model_name": model_name,
    "temperature": temperature,
    "max_tokens": max_tokens,
}

st.sidebar.divider()
st.sidebar.caption(
    "Repo: [medqa-multiagent-llm]"
    "(https://github.com/vietbinhit-lang/medqa-multiagent-llm)"
)

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

st.title("🩺 MedQA Multi-Agent LLM System")
st.caption(
    "Hệ thống multi-agent (reasoning → critic → verifier) + RAG + bộ nhớ "
    "cho câu hỏi trắc nghiệm y khoa kiểu USMLE. Dữ liệu và corpus mẫu là "
    "nội dung tự soạn, không phải bộ đề MedQA-USMLE chính thức."
)

configs = load_configs()
sample_questions = load_sample_questions()

if not configs:
    st.error("Không tìm thấy file cấu hình nào trong configs/. Kiểm tra lại repo.")
    st.stop()

tab_demo, tab_dashboard = st.tabs(["🗣️ Hỏi đáp trực tiếp", "📊 Dashboard đánh giá"])

# --------------------------------------------------------------------------
# Tab 1: interactive Q&A
# --------------------------------------------------------------------------

with tab_demo:
    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        variant_label = st.selectbox("Chọn variant", list(configs.keys()))
        base_cfg = configs[variant_label]
        st.caption(VARIANT_DESCRIPTIONS.get(variant_label, ""))
        st.code(components_badge(base_cfg["components"]), language=None)

        question_source = st.radio(
            "Nguồn câu hỏi", ["Chọn từ bộ mẫu (20 câu)", "Tự nhập câu hỏi"], horizontal=False
        )

        current_question = None
        if question_source.startswith("Chọn"):
            if not sample_questions:
                st.warning("Không tìm thấy data/sample_dev.jsonl.")
            else:
                options_map = {
                    f"{q['id']}: {q['question'][:70]}..." if len(q["question"]) > 70
                    else f"{q['id']}: {q['question']}": q
                    for q in sample_questions
                }
                picked = st.selectbox("Câu hỏi mẫu", list(options_map.keys()))
                current_question = options_map[picked]
                st.markdown(f"**Câu hỏi:** {current_question['question']}")
                for key, val in sorted(current_question["options"].items()):
                    st.markdown(f"- **{key}.** {val}")
                show_gold = st.checkbox("Hiện đáp án đúng (trước khi chạy)")
                if show_gold:
                    st.info(f"Đáp án đúng: **{current_question['answer']}**")
        else:
            q_text = st.text_area("Nội dung câu hỏi", height=120)
            c1, c2 = st.columns(2)
            opt_a = c1.text_input("Lựa chọn A")
            opt_b = c2.text_input("Lựa chọn B")
            opt_c = c1.text_input("Lựa chọn C")
            opt_d = c2.text_input("Lựa chọn D")
            if q_text and opt_a and opt_b and opt_c and opt_d:
                current_question = {
                    "id": "custom",
                    "question": q_text,
                    "options": {"A": opt_a, "B": opt_b, "C": opt_c, "D": opt_d},
                }

        run_clicked = st.button("▶️ Chạy", type="primary", disabled=current_question is None)

    with col_right:
        if run_clicked and current_question is not None:
            runtime_cfg = build_runtime_config(base_cfg, llm_settings)
            try:
                with st.spinner("Đang gọi các agent..."):
                    orchestrator = build_orchestrator(runtime_cfg, api_key)
                    started = time.time()
                    result = orchestrator.run(current_question)
                    elapsed = time.time() - started
            except Exception as exc:  # noqa: BLE001 - surface any endpoint/config error to the user
                st.error(f"Lỗi khi gọi LLM endpoint: {exc}")
            else:
                answer = result.get("answer")
                gold = current_question.get("answer")
                st.subheader(f"Đáp án cuối cùng: {answer or '(không hợp lệ)'}")
                if gold:
                    if answer == gold:
                        st.success(f"✅ Đúng (đáp án đúng: {gold})")
                    else:
                        st.error(f"❌ Sai (đáp án đúng: {gold})")
                st.caption(f"Thời gian chạy: {elapsed:.1f}s")

                st.markdown("**Giải thích cuối cùng:**")
                st.write(result.get("explanation") or "(trống)")

                trace = result.get("trace", {})
                st.divider()
                st.markdown("### Trace từng agent")
                if "proposal" in trace:
                    with st.expander("1️⃣ Reasoning agent (đề xuất ban đầu)", expanded=True):
                        st.write(f"Đáp án đề xuất: **{trace['proposal'].get('answer')}**")
                        st.text(trace["proposal"].get("raw", ""))
                if "review" in trace:
                    with st.expander("2️⃣ Critic agent (phản biện độc lập)"):
                        st.write(f"Đáp án của critic: **{trace['review'].get('answer')}**")
                        st.text(trace["review"].get("raw", ""))
                if "verifier" in trace:
                    with st.expander("3️⃣ Verifier agent (quyết định cuối)"):
                        st.write(f"Đáp án cuối: **{trace['verifier'].get('answer')}**")
                        st.text(trace["verifier"].get("raw", ""))
        elif current_question is None:
            st.info("Chọn hoặc nhập một câu hỏi ở bên trái, rồi bấm Chạy.")

# --------------------------------------------------------------------------
# Tab 2: evaluation dashboard
# --------------------------------------------------------------------------

with tab_dashboard:
    st.markdown(
        "Chạy từng variant trên toàn bộ 20 câu hỏi mẫu (`data/sample_dev.jsonl`) "
        "và so sánh accuracy. Đây là bộ mẫu tự soạn để demo pipeline, **không phải** "
        "bộ test chính thức MedQA-USMLE."
    )

    if not sample_questions:
        st.warning("Không tìm thấy data/sample_dev.jsonl.")
    else:
        selected_variants = st.multiselect(
            "Chọn variant để đánh giá", list(configs.keys()), default=list(configs.keys())
        )
        n_calls_estimate = 0
        for label in selected_variants:
            comps = configs[label]["components"]
            per_q = 1 + int(bool(comps.get("multi_agent"))) + int(bool(comps.get("verifier")))
            n_calls_estimate += per_q * len(sample_questions)
        st.caption(
            f"Ước tính khoảng **{n_calls_estimate}** lượt gọi LLM cho toàn bộ lựa chọn hiện tại "
            f"({len(sample_questions)} câu hỏi mỗi variant)."
        )

        run_eval_clicked = st.button("▶️ Chạy đánh giá", type="primary", disabled=not selected_variants)

        if run_eval_clicked:
            results_rows = []
            all_predictions = {}
            progress = st.progress(0.0, text="Đang chạy...")
            total_steps = len(selected_variants) * len(sample_questions)
            done_steps = 0

            for label in selected_variants:
                runtime_cfg = build_runtime_config(configs[label], llm_settings)
                try:
                    orchestrator = build_orchestrator(runtime_cfg, api_key)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"[{label}] Lỗi khởi tạo: {exc}")
                    continue

                preds, gold = [], []
                error = None
                for q in sample_questions:
                    try:
                        r = orchestrator.run(q)
                    except Exception as exc:  # noqa: BLE001
                        error = str(exc)
                        break
                    preds.append(r.get("answer") or "")
                    gold.append(q["answer"])
                    done_steps += 1
                    progress.progress(
                        done_steps / total_steps, text=f"Đang chạy {label}: {done_steps}/{total_steps}"
                    )

                if error:
                    st.error(f"[{label}] Dừng do lỗi khi gọi LLM: {error}")
                    continue

                all_predictions[label] = preds
                results_rows.append(
                    {
                        "variant": label,
                        "accuracy": accuracy(preds, gold),
                        "invalid_rate": invalid_response_rate(preds),
                        "n_questions": len(gold),
                    }
                )

            progress.empty()

            if results_rows:
                st.session_state["eval_results"] = results_rows
                st.session_state["eval_predictions"] = all_predictions
                st.session_state["eval_gold"] = [q["answer"] for q in sample_questions]

        if "eval_results" in st.session_state:
            df = pd.DataFrame(st.session_state["eval_results"]).sort_values(
                "accuracy", ascending=False
            )
            st.markdown("### Kết quả")
            st.dataframe(
                df.style.format({"accuracy": "{:.1%}", "invalid_rate": "{:.1%}"}),
                width="stretch",
                hide_index=True,
            )
            st.bar_chart(df.set_index("variant")["accuracy"])

            preds = st.session_state.get("eval_predictions", {})
            if "V0_direct_llm" in preds and "V3_full_system" in preds:
                gold = st.session_state["eval_gold"]
                gain = accuracy(preds["V3_full_system"], gold) - accuracy(preds["V0_direct_llm"], gold)
                st.metric("Accuracy gain (V3 − V0)", f"{gain:+.1%}")

            for label, preds_list in preds.items():
                st.download_button(
                    f"Tải dự đoán {label} (JSON)",
                    data=json.dumps(preds_list, ensure_ascii=False, indent=2),
                    file_name=f"{label}_predictions.json",
                    mime="application/json",
                )
