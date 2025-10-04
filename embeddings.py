# embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

# --- Initiera Hugging Face model ---
hf_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def create_embeddings(df):
    """
    Skapar embeddings för alla rader i df['description'] med Hugging Face.
    """
    if 'description' not in df.columns:
        raise ValueError("DataFrame saknar kolumnen 'description'")
    
    # Gör NaN till tom sträng för säkerhets skull
    df['description'] = df['description'].fillna("")

    df['embedding'] = df['description'].apply(lambda x: hf_model.encode(x))
    return df

def get_embedding(text):
    """
    Skapar embedding för en textsträng med Hugging Face.
    """
    return hf_model.encode(text)



