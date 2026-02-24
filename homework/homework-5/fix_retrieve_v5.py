import json
import os

notebooks = [
    "homework/homework-5/5.you-can-just-build-things (1).ipynb",
    "homework/homework-5/5.you-can-just-build-things_executed.ipynb"
]

# Using single triple-quotes for the outer string to avoid escaping inner triple-double-quotes
UNIVERSAL_RETRIEVE = '''def retrieve(*args, **kwargs):
    """
    Universal retrieve function that handles all calling patterns:
    1. retrieve(query, top_k=5)
    2. retrieve(df, emb, query, top_k=5)
    3. retrieve(df, emb, model, query, top_k=5)
    4. retrieve(query, df, top_k=5)
    """
    import pandas as pd
    import numpy as np
    
    # 1. Inspect Positional Arguments by Type
    df = kwargs.get('df')
    emb = kwargs.get('emb')
    model = kwargs.get('model')
    query = kwargs.get('query') or kwargs.get('question')
    top_k = kwargs.get('top_k', 5)
    
    for arg in args:
        if isinstance(arg, pd.DataFrame):
            df = arg
        elif isinstance(arg, np.ndarray):
            emb = arg
        elif hasattr(arg, "encode") and not isinstance(arg, str):
            model = arg
        elif isinstance(arg, str):
            query = arg
        elif isinstance(arg, (int, float)) and not isinstance(arg, bool):
            top_k = int(arg)

    # 2. Defaults from Globals
    if df is None: df = globals().get('df_chunks')
    if emb is None: emb = globals().get('emb_matrix')
    
    if df is None: raise ValueError("df_chunks not found - please run the data processing cells.")
    if emb is None: raise ValueError("emb_matrix not found - please run the embedding cells.")
    if query is None: raise ValueError("No search query provided.")
    
    # 3. Embedding Logic
    if model is not None:
        qv = model.encode([query])[0].astype("float32")
    else:
        # Try local embed_query function first
        try:
            qv = globals().get('embed_query')(query)
        except:
            # Fallback to global 'model'
            m = globals().get('model')
            if m: qv = m.encode([query])[0].astype("float32")
            else: raise ValueError("No embedding model or embed_query() found.")

    # 4. Perform Search
    qv = qv / (np.linalg.norm(qv) + 1e-12)
    scores = emb @ qv
    idx = np.argsort(-scores)[:top_k]
    out = df.iloc[idx].copy()
    out["score"] = scores[idx]
    return out
'''

for nb_path in notebooks:
    if not os.path.exists(nb_path):
        continue
    
    print(f"Processing {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            
            # Replace ANY definition of retrieve
            if "def retrieve" in source:
                print(f"  - Replacing retrieve definition.")
                cell["source"] = [UNIVERSAL_RETRIEVE]
                modified = True

    if modified:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"  - Successfully updated {nb_path}")
