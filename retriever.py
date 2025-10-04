# retriever.py
import numpy as np
import pandas as pd
from embeddings import create_embeddings, EMB_DIM, get_embedding

def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def search_jobs(query, use_dummy=True, top_k=3):
    """
    Enkel retriever som returnerar top_k jobb (title, description).
    Just nu använder intern dummy-dataset; senare ersätter du dataset med JobTech API.
    """
    # ---- dummy dataset ----
    data = {
        "title": ["Python-utvecklare", "Data Scientist", "Frontend Developer", "Backend Engineer", "ML Engineer"],
        "company": ["Tech AB", "DataCorp", "WebDevX", "BackendCo", "AI Solutions"],
        "city": ["Stockholm", "Göteborg", "Malmö", "Stockholm", "Uppsala"],
        "description": [
            "Vi söker en Python-utvecklare med erfarenhet av webbutveckling.",
            "Data Scientist som kan ML och analys av stora datamängder.",
            "Frontend Developer med React/JS erfarenhet.",
            "Backend Engineer med Python och Django.",
            "Machine Learning Engineer som jobbar med modellutveckling."
        ]
    }
    df = pd.DataFrame(data)
    # skapa embeddings för jobben
    df = create_embeddings(df, use_dummy=use_dummy)

    # skapa query-embedding
    if use_dummy:
        query_emb = np.random.rand(EMB_DIM).astype(np.float32)
    else:
        query_emb = get_embedding(query)

    # beräkna likhet
    df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity(x, query_emb))

    # sortera och returnera topp-k som lista av tuples
    top = df.sort_values("similarity", ascending=False).head(top_k)
    return list(zip(top["title"], top["company"], top["city"], top["description"], top["similarity"]))



# python retriever.py
