import json
import os
from pathlib import Path

notebooks = [
    "homework/homework-5/5.you-can-just-build-things (1).ipynb",
    "homework/homework-5/5.you-can-just-build-things_executed.ipynb"
]

# Robust retrieve function that handles both 4-arg and 5-arg styles
FIX_RETRIEVE_CODE = """def retrieve(df, emb, model_or_question, question=None, top_k=5):
    \"\"\"
    A robust retrieve function that handles both notebook and app-style calls.
    Styles:
    1. retrieve(df_chunks, vectors, question, top_k=5)
    2. retrieve(df_chunks, vectors, model, question, top_k=5)
    \"\"\"
    if question is None:
        # Style 1: question is the 3rd positional argument
        actual_question = model_or_question
        # Use the notebook's embed_query function (OpenAI API) or local model
        try:
            qv = embed_query(actual_question)
        except NameError:
            # Fallback if embed_query isn't defined
            qv = model.encode([actual_question])[0].astype(np.float32)
    else:
        # Style 2: model is 3rd, question is 4th
        model = model_or_question
        actual_question = question
        qv = model.encode([actual_question])[0].astype(np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-12)

    scores = emb @ qv
    idx = np.argsort(-scores)[:top_k]
    out = df.iloc[idx].copy()
    out["score"] = scores[idx]
    return out
"""

for nb_path in notebooks:
    if not os.path.exists(nb_path):
        print(f"Skipping {nb_path} (not found)")
        continue
    
    print(f"Fixing {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    modified = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            
            # Find the second definition of retrieve (the one with 4 args)
            if "def retrieve(df" in source and "emb:" in source:
                print(f"  - Found advanced retrieve definition, applying fix.")
                cell["source"] = [FIX_RETRIEVE_CODE]
                modified = True

    if modified:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"  - Successfully updated {nb_path}")
    else:
        print(f"  - No matching retrieve definition found in {nb_path}")
