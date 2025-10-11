# app.py
import streamlit as st
import numpy as np
import pandas as pd
import re
import json
from collections import Counter
from fetch_jobs import get_jobs
from embeddings import create_embeddings, get_embedding

st.set_page_config(page_title="Jobbcoach Chatbot (RAG)", layout="wide")

# --- Ladda kompetenser lokalt ---
def load_skills():
    try:
        with open("skills.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            skills = set()

            # Hämta svenska etiketter ur olika JSON-format
            if "data" in data and "concepts" in data["data"]:
                for item in data["data"]["concepts"]:
                    if "preferred_label" in item:
                        skills.add(item["preferred_label"].strip().lower())
            elif "concepts" in data:
                for item in data["concepts"]:
                    if "preferred_label" in item:
                        skills.add(item["preferred_label"].strip().lower())
            else:
                for k, v in data.items():
                    if isinstance(v, list):
                        for s in v:
                            if isinstance(s, dict) and "preferred_label" in s:
                                skills.add(s["preferred_label"].strip().lower())

            return list(skills)
    except Exception as e:
        print(f"Misslyckades med att läsa skills.json: {e}")
        return []

skills_list = load_skills()
common_words = {"och", "att", "för", "man", "vi", "är", "på", "en", "i", "ett", "med", "av", "till", "the", "you", "we", "our", "job"}

# --- Session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df_sorted" not in st.session_state:
    st.session_state.df_sorted = pd.DataFrame()
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_initialized" not in st.session_state:
    st.session_state.chat_initialized = False

# --- Chatfunktioner ---
def reset_chat():
    st.session_state.chat_history = []
    st.session_state.chat_initialized = False

def initialize_chat():
    if not st.session_state.chat_initialized:
        st.session_state.chat_history.append({
            "role": "bot",
            "content": (
                "Hej! Jag är din jobbcoach-chatbot. Jag kan svara på frågor och ge information om de annonser du har sökt.\n\n"
                "Skriv en fråga eller välj en av knapparna nedan:"
            )
        })
        st.session_state.chat_initialized = True

# --- Extrahera kompetenser ---
def extract_skills_from_text(texts):
    found_skills = []
    clean_text = " ".join(texts).lower()

    # Ta bara med hela ord
    for skill in skills_list:
        skill = skill.strip().lower()
        if len(skill) < 3 or skill in common_words:
            continue
        # Leta bara efter hela ord
        if re.search(rf"\b{re.escape(skill)}\b", clean_text):
            found_skills.append(skill)

    # Räkna förekomster
    counter = Counter(found_skills)
    common_skills = [s for s, _ in counter.most_common(10)]

    return common_skills if common_skills else ["Ingen specifik kompetens hittades"]

# --- Layout ---
col1, col2 = st.columns([0.75, 0.25])

with col1:
    st.title("Jobbcoach Chatbot (RAG) — Prototype")

    user_input = st.text_input("Skriv din fråga om jobb (t.ex. 'Python Stockholm'):")
    num_jobs = st.slider("Hur många annonser vill du hämta?", 5, 50, 10)

    # --- Sök-knapp ---
    if st.button("Sök") and user_input.strip():
        st.write(f"🔍 Söker relevanta jobb för: '{user_input}'")
        reset_chat()

        df = get_jobs(query=user_input, limit=num_jobs * 2)

        if df.empty:
            st.error("Inga jobbannonser hittades.")
        else:
            df = df.head(num_jobs)
            df = create_embeddings(df)
            query_vec = get_embedding(user_input)

            def cosine_similarity(vec1, vec2):
                return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

            df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity(x, query_vec))
            df_sorted = df.sort_values(by="similarity", ascending=False)
            st.session_state.df_sorted = df_sorted

    # --- Stad-dropdown ---
    if not st.session_state.df_sorted.empty:
        cities = sorted(st.session_state.df_sorted["city"].dropna().unique())
        cities.insert(0, "Alla städer")
        selected_city = st.selectbox("Välj stad (gäller bara annonserna du har sökt):", cities)
        df_display = st.session_state.df_sorted
        if selected_city != "Alla städer":
            df_display = df_display[df_display["city"] == selected_city]
    else:
        df_display = pd.DataFrame()

    # --- Chatbot-svar + annonser ---
    if not df_display.empty:
        df_sorted = df_display
        count = len(df_sorted)
        city_text = f" i {selected_city}" if selected_city != "Alla städer" else ""
        examples = [
            f"{row['title']} på {row['company']} ({row['city']})"
            for _, row in df_sorted.head(3).iterrows()
        ]

        st.subheader("🤖 Chatbot-svar:")
        st.write(
            f"Jag hittade {count} relevanta jobb för '{user_input}'{city_text}. "
            f"Här är några exempel: {'; '.join(examples)}. "
            "Vill du se fler detaljer om något av dessa jobb, klicka på länken under varje annons."
        )

        st.subheader("📋 Mest relevanta jobb:")
        for _, row in df_sorted.iterrows():
            st.markdown(f"**{row['title']}** — {row['company']} ({row['city']})")
            st.write(row["description"][:200] + "…")
            ad_url = row.get("url", f"https://jobsearch.api.jobtechdev.se/ad/{row.get('adId','')}")
            if ad_url:
                st.markdown(f"[📄 Läs mer här]({ad_url})")
            st.write("---")

with col2:
    st.subheader("💬 Chatbot")
    st.caption("Frågor gäller annonserna du har sökt.")

    if st.button("Öppna/stäng chatten"):
        st.session_state.chat_open = not st.session_state.chat_open
        if st.session_state.chat_open:
            initialize_chat()

    if st.session_state.chat_open:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**Du:** {msg['content']}")
            else:
                st.markdown(f"**Bot:** {msg['content']}")

        st.write("### Välj en fråga:")

        df_sorted = st.session_state.df_sorted

        # --- Fördefinierade frågor ---
        if st.button("🏙️ Vilken stad har flest jobb?") and not df_sorted.empty:
            top_city = df_sorted["city"].value_counts().idxmax()
            st.session_state.chat_history.append({"role": "user", "content": "Vilken stad har flest jobb?"})
            st.session_state.chat_history.append({"role": "bot", "content": f"🏙️ Flest jobb finns i: {top_city}"})
            st.rerun()

        if st.button("📋 Visa tre exempeljobb!") and not df_sorted.empty:
            examples = [
                f"{row['title']} på {row['company']} ({row['city']})"
                for _, row in df_sorted.head(3).iterrows()
            ]
            st.session_state.chat_history.append({"role": "user", "content": "Visa tre exempeljobb!"})
            st.session_state.chat_history.append({"role": "bot", "content": "📋 Här är tre exempeljobb: " + "; ".join(examples)})
            st.rerun()

        if st.button("🛠️ Vilka kompetenser behövs?") and not df_sorted.empty:
            descriptions = df_sorted["description"].tolist()
            skills_found = extract_skills_from_text(descriptions)
            st.session_state.chat_history.append({"role": "user", "content": "Vilka kompetenser behövs?"})
            st.session_state.chat_history.append(
                {"role": "bot", "content": f"🛠️ Några vanliga kompetenser: {', '.join(skills_found)}"}
            )
            st.rerun()

        # --- Fritext ---
        chat_input = st.text_input("✍️ Eller skriv egen fråga:")
        if st.button("Skicka") and chat_input.strip():
            user_q = chat_input.lower()
            st.session_state.chat_history.append({"role": "user", "content": chat_input})

            answer = None
            df_sorted = st.session_state.df_sorted

            if not df_sorted.empty:
                # --- Antal jobb i viss stad ---
                city_match = None
                for city in df_sorted["city"].dropna().unique():
                    if city.lower() in user_q:
                        city_match = city
                        break
                if city_match:
                    count = len(df_sorted[df_sorted["city"].str.lower() == city_match.lower()])
                    answer = f"🤖 Det finns {count} jobb i {city_match} enligt de annonser vi har hämtat."

                # --- Distansjobb ---
                elif any(word in user_q for word in ["distans", "remote", "hemifrån", "fjärrarbete"]):
                    remote_jobs = df_sorted[
                        df_sorted["description"].str.contains("distans|remote|hemifrån|fjärr", case=False, na=False)
                    ]
                    count = len(remote_jobs)
                    if count > 0:
                        examples = [
                            f"{row['title']} på {row['company']} ({row['city']})"
                            for _, row in remote_jobs.head(3).iterrows()
                        ]
                        answer = f"🤖 Jag hittade {count} distansjobb. Exempel: {'; '.join(examples)}."
                    else:
                        answer = "🤖 Jag hittade tyvärr inga distansjobb."

                # --- Kompetensfrågor i fritext ---
                elif "kompetens" in user_q or "skills" in user_q:
                    descriptions = df_sorted["description"].tolist()
                    skills_found = extract_skills_from_text(descriptions)
                    answer = f"🛠️ Några vanliga kompetenser: {', '.join(skills_found)}"

                # --- Annars fallback ---
                if not answer:
                    answer = (
                        "Åhnej, detta har jag inte lärt mig än 🙁️ "
                        "Kan jag kanske hjälpa till med något annat istället?"
                    )

            else:
                answer = "🤖 Jag har ingen annonsdata att analysera just nu. Sök efter jobb först."

            st.session_state.chat_history.append({"role": "bot", "content": answer})
            st.rerun()

        # --- Rensa ---
        if st.button("🧹 Rensa chatten"):
            reset_chat()
            initialize_chat()
            st.rerun()



# streamlit run app.py
