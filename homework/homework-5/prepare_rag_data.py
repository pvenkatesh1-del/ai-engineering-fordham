import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pathlib
from tqdm import tqdm

def load_fordham_data(source_path):
    data = []
    print(f"Loading data from {source_path}...")
    path = pathlib.Path(source_path)
    files = list(path.glob('*.md'))
    # Using a subset for faster demonstration, or all if preferred
    # Files are around 9500. Let's try to process all as it's local.
    for file_path in tqdm(files, desc="Loading files"):
         try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if not lines: continue
                url = lines[0].strip()
                content = "".join(lines[1:]).strip()
                data.append({
                    "filename": file_path.name,
                    "url": url,
                    "content": content
                })
         except Exception:
            continue
    return pd.DataFrame(data)

def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    if not text: return chunks
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        step = chunk_size - overlap
        if step <= 0: step = 1
        start += step
    return chunks

def main():
    SCRIPT_DIR = pathlib.Path(__file__).parent
    REPO_ROOT = SCRIPT_DIR.parents[1]
    
    source_path = REPO_ROOT / 'data' / 'fordham-website'
    df = load_fordham_data(str(source_path))
    
    chunked_data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        content = row.get('content', '')
        text_chunks = chunk_text(content)
        for i, chunk_content in enumerate(text_chunks):
            if not chunk_content.strip(): continue
            chunked_data.append({
                "chunk": chunk_content.strip(),
                "url": row['url']
            })
    
    df_chunks = pd.DataFrame(chunked_data)
    print(f"Generated {len(df_chunks)} chunks.")
    
    print("Embedding chunks with all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df_chunks['chunk'].tolist(), show_progress_bar=True)
    
    # Save files expected by Streamlit app
    df_chunks.to_csv(SCRIPT_DIR / "chunks.csv", index=False)
    np.save(SCRIPT_DIR / "chunk_embeddings.npy", embeddings)
    print(f"Data preparation complete. Files created in {SCRIPT_DIR}")

if __name__ == "__main__":
    main()
