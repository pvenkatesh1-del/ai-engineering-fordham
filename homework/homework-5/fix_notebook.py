import json
import os

notebook_path = "5.you-can-just-build-things (1).ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        
        # 1. Define test_queries if not present
        if "for q in test_queries:" in source and "test_queries =" not in source:
            cell["source"] = ["test_queries = [\n", 
                             "    \"What programs does the Gabelli School of Business offer?\",\n",
                             "    \"How do I apply for financial aid?\",\n",
                             "    \"Where is Fordham's campus?\"\n",
                             "]\n"] + cell["source"]
        
        # 2. Fix OpenAI SDK calls
        if "client.responses.create" in source:
            source = source.replace("client.responses.create", "client.chat.completions.create")
            source = source.replace("input=prompt", "messages=[{\"role\": \"user\", \"content\": prompt}]")
            source = source.replace("max_output_tokens=400", "max_tokens=400")
            cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")]
            if cell["source"][-1] == "\n": cell["source"].pop()

        if "response.output_text" in source:
            cell["source"] = [line.replace("response.output_text", "response.choices[0].message.content") for line in cell["source"]]

        # 3. Fix __file__ in streamlit section
        if "__file__" in source:
            cell["source"] = [line.replace("Path(__file__)", "Path.cwd()") for line in cell["source"]]
            cell["source"] = [line.replace("parents[1]", "parents[0]") for line in cell["source"]] # Adjust parents

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
