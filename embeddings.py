# embeddings.py
import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import pandas as pd

# Läser .env
load_dotenv()

# Initiera OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-3-small"):
    """
    Skapar embedding för en textsträng med OpenAI.
    """
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

def create_embeddings(df, use_dummy=True):
    """
    Skapar embeddings för df['description'].
    use_dummy=True → lokalt/test utan OpenAI
    use_dummy=False → riktiga OpenAI embeddings
    """
    if use_dummy:
        # dummy embeddings: random vektorer 1536 dimensioner
        df['embedding'] = df['description'].apply(lambda x: np.random.rand(1536))
    else:
        df['embedding'] = df['description'].apply(lambda x: get_embedding(x))
    return df



