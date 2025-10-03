# retriever.py
import numpy as np
import pandas as pd
from embeddings import create_embeddings

def cosine_similarity(a, b):
    """Beräkna cosinuslikhet mellan två vektorer."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_jobs(query, use_dummy=True, top_k=3):
    """
    Tar en användarfråga och returnerar de mest relevanta jobben.
    """
    # Dummy-dataset (du kan byta till riktiga jobbannonser senare)
    data = {
        "title": ["Python-utvecklare", "Data Scientist", "Frontend Developer"],
        "description": [
            "Vi söker en Python-utvecklare med erfarenhet av webbutveckling.",
            "Data Scientist som kan ML och analys av stora datamängder.",
            "Frontend Developer med React/JS erfarenhet."
        ]
    }
    df = pd.DataFrame(data)

    # Skapa embeddings (dummy eller riktiga)
    df = create_embeddings(df, use_dummy=use_dummy)

    # Embedding för query
    if use_dummy:
        query_emb = np.random.rand(1536)
    else:
        from openai import OpenAI
        import os
        from dotenv import load_dotenv
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        query_emb = client.embeddings.create(
            model="text-embedding-3-small", input=query
        ).data[0].embedding

    # Beräkna likhet
    df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity(x, query_emb))

    # Sortera och returnera toppresultat
    top_results = df.sort_values("similarity", ascending=False).head(top_k)
    return list(zip(top_results["title"], top_results["description"]))


if __name__ == "__main__":
    query = "Jag gillar maskininlärning och stora datamängder."
    results = search_jobs(query, use_dummy=True)
    print("Query:", query)
    print("Mest relevanta jobb:")
    for title, desc in results:
        print("-", title, ":", desc)



# python retriever.py
