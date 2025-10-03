# fetch_jobs.py
import pandas as pd
from embeddings import create_embeddings
import faiss
import numpy as np

def get_jobs(query, limit=5, use_dummy=True):
    # Exempeldata (du kan ersätta med JobTech API senare)
    data = {
        "title": ["Python-utvecklare", "Data Scientist", "Frontend Developer"],
        "company": ["Tech AB", "DataCorp", "WebDevX"],
        "city": ["Stockholm", "Göteborg", "Malmö"],
        "description": [
            "Vi söker en Python-utvecklare med erfarenhet av webbutveckling.",
            "Data Scientist som kan ML och analys av stora datamängder.",
            "Frontend Developer med React/JS erfarenhet."
        ]
    }
    df = pd.DataFrame(data)

    # Skapa embeddings (dummy nu)
    df = create_embeddings(df, use_dummy=use_dummy)

    # Bygg FAISS index
    dimension = len(df['embedding'].iloc[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.vstack(df['embedding'].values))

    # Skapa embedding för frågan (dummy nu)
    if use_dummy:
        query_vec = np.random.rand(dimension)
    else:
        from embeddings import get_embedding
        query_vec = get_embedding(query)

    # Sök topp-N
    D, I = index.search(np.array([query_vec]), k=limit)
    return df.iloc[I[0]]


# python fetch_jobs.py
