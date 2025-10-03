# test_embeddings.py
import pandas as pd
from embeddings import create_embeddings

# Exempeldata för test
data = {
    "title": ["Python-utvecklare", "Data Scientist", "Frontend Developer"],
    "description": [
        "Vi söker en Python-utvecklare med erfarenhet av webbutveckling.",
        "Data Scientist som kan ML och analys av stora datamängder.",
        "Frontend Developer med React/JS erfarenhet."
    ]
}

df = pd.DataFrame(data)

# Skapa embeddings med dummy
df = create_embeddings(df, use_dummy=True)

# Visa resultat
print(df[['title', 'embedding']])

# python test_embeddings.py
