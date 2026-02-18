import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Models
# We use the local model for embeddings to match the preparation script
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'
CHAT_MODEL = "gpt-4o-mini"

st.set_page_config(page_title="Fordham RAG", layout="wide")
st.title("Fordham RAG Assistant")

# Paths
PROJECT_ROOT = Path.cwd()
CHUNKS_PATH = PROJECT_ROOT / "chunks.csv"
EMB_PATH = PROJECT_ROOT / "chunk_embeddings.npy"
ENV_PATH = PROJECT_ROOT / ".env"

# Load env
load_dotenv(dotenv_path=ENV_PATH)

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY missing. Add it to .env in the project root.")
    st.stop()

client = OpenAI()

@st.cache_resource
def load_models():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_data
def load_index():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"chunks.csv not found at {CHUNKS_PATH}. Run prepare_rag_data.py first.")
    if not EMB_PATH.exists():
        raise FileNotFoundError(f"chunk_embeddings.npy not found at {EMB_PATH}. Run prepare_rag_data.py first.")

    df = pd.read_csv(CHUNKS_PATH)
    emb = np.load(EMB_PATH).astype(np.float32)
    # Normalize
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return df, emb

def retrieve(df, emb, model, question, top_k):
    qv = model.encode([question])[0].astype(np.float32)
    qv = qv / (np.linalg.norm(qv) + 1e-12)
    scores = emb @ qv
    idx = np.argsort(-scores)[:top_k]
    out = df.iloc[idx].copy()
    out["score"] = scores[idx]
    return out

def generate_answer(question, sources_df):
    blocks = []
    for i, row in sources_df.reset_index(drop=True).iterrows():
        url = str(row.get("url", "")).strip()
        chunk = str(row.get("chunk", "")).strip()
        blocks.append(f"Source {i+1}\nURL: {url}\n\n{chunk}")

    context = "\n\n---\n\n".join(blocks)

    prompt = f"""You are a Fordham University assistant.
Answer using ONLY the sources. If not found, say: I don't know based on the provided sources.
Cite sources like [Source 1], [Source 2].

Question: {question}

Sources:
{context}
"""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content

# App Logic
try:
    embed_model = load_models()
    df_chunks, vectors = load_index()
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()

with st.sidebar:
    st.info(f"Loaded {len(df_chunks)} chunks.")
    top_k = st.slider("Top K sources", 3, 12, 5)
    show_sources = st.checkbox("Show sources", True)

question = st.text_input("Ask a question about Fordham", placeholder="e.g., How do I apply for financial aid?")
ask = st.button("Ask")

if ask:
    if not question.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("Retrieving sources..."):
            sources = retrieve(df_chunks, vectors, embed_model, question, top_k=top_k)

        with st.spinner("Generating answer..."):
            ans = generate_answer(question, sources)

        st.subheader("Answer")
        st.write(ans)

        if show_sources:
            st.subheader("Sources")
            for i, row in sources.reset_index(drop=True).iterrows():
                st.markdown(f"**Source {i+1}** | score `{row['score']:.4f}`")
                url = row.get("url", "")
                if url:
                    st.markdown(f"[{url}]({url})")
                st.code(str(row.get("chunk", ""))[:1200])
                st.divider()
