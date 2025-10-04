# app.py
import streamlit as st
import numpy as np
from fetch_jobs import get_jobs
from embeddings import create_embeddings, get_embedding

st.title("Jobbcoach Chatbot (RAG) — Prototype")

# --- Användarinmatning ---
user_input = st.text_input("Skriv din fråga om jobb (t.ex. 'Python'):")

# Slider för antal annonser
num_jobs = st.slider("Hur många annonser vill du hämta?", 5, 50, 10)

# --- Hämta initiala annonser för att skapa stad-dropdown ---
if user_input.strip():
    initial_df = get_jobs(query=user_input, limit=100)  # hämta max 100 annonser för dropdown
    cities = sorted(initial_df['city'].dropna().unique())
    cities.insert(0, "Alla städer")
    selected_city = st.selectbox("Välj stad:", cities)

# --- Sök-knapp ---
if st.button("Sök") and user_input.strip():
    st.write(f"🔍 Söker relevanta jobb för: {user_input} i {selected_city}")

    # Hämta annonser
    df = get_jobs(query=user_input, limit=num_jobs*2)  # hämta fler för att filtrering på stad inte ska ta bort resultat

    if df.empty:
        st.error("Inga jobbannonser hittades.")
    else:
        # Filtrera på vald stad
        if selected_city != "Alla städer":
            df = df[df["city"] == selected_city]

        # Begränsa till max antal annonser
        df = df.head(num_jobs)

        if df.empty:
            st.warning("Inga annonser hittades för den staden.")
        else:
            # Skapa embeddings
            df = create_embeddings(df)

            # Embedding för användarfrågan
            query_vec = get_embedding(user_input)

            # Cosine similarity
            def cosine_similarity(vec1, vec2):
                return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

            df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(x, query_vec))
            df_sorted = df.sort_values(by='similarity', ascending=False)

            # Visa resultat
            st.subheader("Mest relevanta jobb:")
            for _, row in df_sorted.iterrows():
                st.markdown(f"**{row['title']}** — {row['company']} ({row['city']})")
                st.write(row['description'][:200] + "…")
                ad_url = row['url'] if 'url' in row and row['url'] else f"https://jobsearch.api.jobtechdev.se/ad/{row.get('adId','')}"
                if ad_url:
                    st.markdown(f"[📄 Läs mer här]({ad_url})")
                st.write("---")



# streamlit run app.py
