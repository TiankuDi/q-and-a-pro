"""Self-contained Streamlit astronomy QA app."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import streamlit as st
from fastembed import TextEmbedding


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATASET_PATH = Path(__file__).with_name("QA.csv")


def normalize(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(lengths, 1e-12)


def load_dataset(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到問答資料檔：{path}")

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        expected_columns = ["question", "answer"]
        if reader.fieldnames != expected_columns:
            raise ValueError(
                f"CSV 欄位必須剛好是 {expected_columns}，目前為 {reader.fieldnames}"
            )
        rows = [
            {"question": row["question"].strip(), "answer": row["answer"].strip()}
            for row in reader
            if row.get("question") and row.get("answer")
        ]

    if not rows:
        raise ValueError("QA.csv 沒有可供查詢的問答資料。")
    return rows


@st.cache_resource(show_spinner="正在載入問答資料與語意模型，首次啟動可能需要一些時間…")
def load_index(dataset_path: str, dataset_mtime_ns: int):
    # The modification time is part of the cache key to refresh the index when QA.csv changes.
    del dataset_mtime_ns
    rows = load_dataset(Path(dataset_path))
    embedder = TextEmbedding(model_name=MODEL_NAME)
    texts = [f"{row['question']} {row['answer']}" for row in rows]
    vectors = normalize(
        np.asarray(list(embedder.embed(texts, batch_size=32)), dtype=np.float32)
    )
    return rows, embedder, vectors


def retrieve_answer(question: str, rows, embedder, document_vectors):
    query_vector = normalize(
        np.asarray(list(embedder.embed([question])), dtype=np.float32)
    )[0]
    similarities = document_vectors @ query_vector
    best_index = int(np.argmax(similarities))
    return rows[best_index], float(similarities[best_index])


def render_app() -> None:
    st.set_page_config(page_title="宇宙天文問答", page_icon="🌌")
    st.title("🌌 宇宙天文問答")
    st.caption("輸入問題後，系統會從 QA.csv 比對最相關的問答並即時回覆。")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("例如：黑洞的事件視界代表什麼？")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            stat = DATASET_PATH.stat()
            rows, embedder, document_vectors = load_index(
                str(DATASET_PATH), stat.st_mtime_ns
            )
            with st.spinner("正在比對相關問答…"):
                match, similarity = retrieve_answer(
                    question.strip(), rows, embedder, document_vectors
                )
            response = (
                f"{match['answer']}\n\n"
                f"**最相關的資料問題：** {match['question']}\n\n"
                f"**語意相似度：** {similarity:.3f}"
            )
            st.markdown(response)
        except Exception as error:
            response = f"載入資料或查詢時發生錯誤：{error}"
            st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    render_app()
