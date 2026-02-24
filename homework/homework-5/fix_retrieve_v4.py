import json
import os

notebooks = [
    "homework/homework-5/5.you-can-just-build-things (1).ipynb",
    "homework/homework-5/5.you-can-just-build-things_executed.ipynb"
]

ROBUST_RETRIEVE = """def retrieve(*args, **kwargs):
    \"\"\"
    Robust retrieve function that handles both notebook and app-style calls.
    Styles:
    1. retrieve(df_chunks, vectors, question, top_k=5)
    2. retrieve(df_chunks, vectors, model, question, top_k=5)
    \"\"\"
    import numpy as np
    import pandas as pd
    
    # Extract positional arguments
    # retrieve(df, emb, question, [top_k])
    # retrieve(df, emb, model, question, [top_k])
    
    df = args[0]
    emb = args[1]
    
    if len(args) >= 4:
        # Check if 3rd arg is a model (has 'encode' method)
        if hasattr(args[2], 'encode'):
            model = args[2]
            question = args[3]
            top_k = args[4] if len(args) > 4 else kwargs.get('top_k', 5)
        else:
            model = None
            question = args[2]
            top_k = args[3]
    elif len(args) == 3:
        # Check if 3rd arg is a model (unlikely here but for safety)
        if hasattr(args[2], 'encode'):
            # This would be an invalid 3-arg call if they wanted to search
            # But let's assume it's (df, emb, question)
            model = None
            question = args[2]
        else:
            model = None
            question = args[2]
        top_k = kwargs.get('top_k', 5)
    else:
        raise ValueError(f"Invalid number of arguments for retrieve: {len(args)}")

    # Get query vector
    if model is not None:
        qv = model.encode([question])[0].astype("float32")
        qv = qv / (np.linalg.norm(qv) + 1e-12)
    else:
        # Try global embed_query or other encoders
        try:
            # If in a cell, embed_query should be defined
            qv = embed_query(question)
        except NameError:
            # Try to find 'model' in globals
            glb_model = globals().get('model')
            if glb_model and hasattr(glb_model, 'encode'):
                qv = glb_model.encode([question])[0].astype("float32")
                qv = qv / (np.linalg.norm(qv) + 1e-12)
            else:
                raise NameError("Could not find an embedding model or embed_query function.")

    scores = emb @ qv
    idx = np.argsort(-scores)[:top_k]
    out = df.iloc[idx].copy()
    out["score"] = scores[idx]
    return out
"""

USER_GENERATION_CODE = """# 7. Synthetic Question Generation (Evaluation)
import pandas as pd
import litellm
import asyncio
import random
import textwrap
from pydantic import BaseModel, Field

# Ensure litellm is installed
# !uv pip install litellm

class SyntheticQuestion(BaseModel):
    chain_of_thought: str = Field(description="Step-by-step reasoning about what makes a good question for this document")
    question: str = Field(description="A natural, specific question that can be answered using the document")
    answer: str = Field(description="The answer to the question")

constraints = [
    "The question should be answerable in one word or a short phrase",
    "The question should require synthesizing multiple facts from the document",
    "Frame the question as something a prospective student might ask",
    "Ask about a specific group, deadline, or requirement mentioned in the document",
]

async def generate_question(doc_id: str, content: str) -> dict:
    \"\"\"Generate a synthetic question for a single chunk using an LLM.\"\"\"
    constraint = random.choice(constraints)
    try:
        # Make sure Gemini is set in your environment variables!
        response = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": textwrap.dedent(f\"\"\"
                    I will give you a document chunk from the Fordham University website. Please generate a question that can be answered using the following document.
                    
                    Text: {content}
                    
                    Rules:
                    - Your question should be natural and specific and concise
                    - Your question should not assume that someone is reading the document, but rather that they are asking a general question about Fordham
                    - Your question must be answerable using the document that I gave you
                    - {constraint}
                    - Do not reference "the document" or "the webpage" in your question
                    \"\"\"
                    ),
                }
            ],
            response_format=SyntheticQuestion,
        )
        
        result = SyntheticQuestion.model_validate_json(response.choices[0].message.content)
        return {"doc_id": doc_id, "question": result.question, "answer": result.answer}
    except Exception as e:
        print(f"Error generating question: {e}")
        return None

# Sample 40 chunks to evaluate on
if 'df_chunks' in globals():
    sample_docs = df_chunks.dropna(subset=['content']).sample(n=40, random_state=42)
    
    # Generate all questions concurrently
    print(f"Generating {len(sample_docs)} synthetic questions...")
    tasks = [generate_question(row["chunk_id"], row["content"]) for _, row in sample_docs.iterrows()]
    
    # Run the async loop
    synthetic_results = await asyncio.gather(*tasks)
    
    # Filter out failed generations
    synthetic_results = [r for r in synthetic_results if r is not None]
    
    synthetic_df = pd.DataFrame(synthetic_results)
    print(f"Generated {len(synthetic_df)} synthetic questions\\n")
    
    # Show some examples
    for _, row in synthetic_df.head(3).iterrows():
        doc = df_chunks[df_chunks["chunk_id"] == row["doc_id"]].iloc[0]
        print(f"Q: {row['question']}")
        print(f"   Source Chunk: {doc['content'][:80]}...")
        print(f"   Answer: {row['answer']}")
        print()
else:
    print("Please run the earlier cells to create df_chunks first!")
"""

EVALUATION_LOOP_CODE = """# 8. Retrieval Evaluation on Synthetic Questions
if 'synthetic_df' in globals() and 'df_chunks' in globals() and 'emb_matrix' in globals():
    print("Evaluating retrieval performance on 40 synthetic questions...")
    results = []
    for _, row in synthetic_df.iterrows():
        actual_id = row['doc_id']
        question = row['question']
        
        # Use our robust retrieve function
        # This will now work regardless of the calling pattern!
        retrieved_df = retrieve(df_chunks, emb_matrix, question, top_k=5)
        
        retrieved_ids = retrieved_df['chunk_id'].tolist()
        success = actual_id in retrieved_ids
        results.append(success)

    accuracy = sum(results) / len(results)
    print("="*40)
    print(f"Retrieval Accuracy (Top-5): {accuracy:.2%}")
    print("="*40)
else:
    print("Missing required data (synthetic_df, df_chunks, or emb_matrix). Please run the previous cells.")
"""

for nb_path in notebooks:
    if not os.path.exists(nb_path):
        continue
    
    print(f"Processing {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    new_cells = []
    inserted_generation = False
    
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            
            # Replace ALL definitions of retrieve
            if "def retrieve" in source:
                print(f"  - Replacing retrieve definition.")
                cell["source"] = [ROBUST_RETRIEVE]
            
            # Remove old generation/eval blocks if they were partially added or exist
            if "synthetic_df" in source and "40" in source and "generate_question" in source:
                print(f"  - Found existing generation cell, replacing with updated version.")
                cell["source"] = [USER_GENERATION_CODE]
                inserted_generation = True

        new_cells.append(cell)
    
    # If not found, append at the end of the retrieval section or end of notebook
    if not inserted_generation:
        print(f"  - Appending generation and evaluation cells.")
        gen_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [USER_GENERATION_CODE]
        }
        eval_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [EVALUATION_LOOP_CODE]
        }
        new_cells.append(gen_cell)
        new_cells.append(eval_cell)
    else:
        # Ensure evaluation cell is added after generation if it wasn't there
        if not any("accuracy = sum(results)" in "".join(c["source"]) for c in new_cells if c["cell_type"] == "code"):
             print(f"  - Adding evaluation cell after generation.")
             eval_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [EVALUATION_LOOP_CODE]
             }
             new_cells.append(eval_cell)

    nb["cells"] = new_cells
    
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"  - Successfully updated {nb_path}")
