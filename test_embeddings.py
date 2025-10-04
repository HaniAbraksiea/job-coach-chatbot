# test_embeddings.py
import pandas as pd
from embeddings import create_embeddings, get_embedding

# Testdata
data = {
    "title": ["Python-utvecklare", "Data Scientist", "Frontend Developer"],
    "description": [
        "Vi söker en Python-utvecklare med erfarenhet av webbutveckling.",
        "Data Scientist som kan ML och analys av stora datamängder.",
        "Frontend Developer med React/JS erfarenhet."
    ]
}

df = pd.DataFrame(data)

print("=== Test med Hugging Face embeddings ===")
df_hf = create_embeddings(df.copy())
print(df_hf[['title', 'embedding']].head())

# Testa embedding för en användarfråga
query = "Jag kan Python och vill jobba med data"
query_vec = get_embedding(query)
print("\nEmbedding för användarfrågan:", query_vec[:5], "...")  # visa första 5 värden


# python test_embeddings.py
