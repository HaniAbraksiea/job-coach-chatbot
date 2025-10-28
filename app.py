# app.py
import streamlit as st
import numpy as np
import pandas as pd
import re
from collections import Counter
from fetch_jobs import get_jobs
from embeddings import create_embeddings, get_embedding
from load_taxonomy import load_taxonomy

st.set_page_config(page_title="💬 Jobbcoach Chatbot", layout="wide")

taxonomy, taxonomy_skill_set = load_taxonomy()

def normalize_text(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^\wåäö\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def find_best_occupation_from_query_or_titles(query, df):
    """
    Försök matcha en yrkesetikett från taxonomin:
    1) kolla query text
    2) om inget: titta på titlar i annonsdata och räkna träffar per occupation
    Returnerar occupation-lower eller None
    """
    q = normalize_text(query)
    # 1) matcha direkt i query
    for entry in taxonomy:
        occ = entry["occupation"]
        if occ and occ in q:
            return occ

    # 2) matcha genom titlar i df (räkna)
    if df is None or df.empty:
        return None
    title_texts = " ".join([normalize_text(str(t)) for t in df["title"].fillna("")])
    occ_counts = Counter()
    for entry in taxonomy:
        occ = entry["occupation"]
        if occ and occ in title_texts:
            occ_counts[occ] += title_texts.count(occ)
    if occ_counts:
        # return top occ
        return occ_counts.most_common(1)[0][0]
    return None

def extract_skills_present_in_descriptions(skills_candidates, descriptions, top_k=7):
    """
    Givet en lista skills_candidates (lowercase phrases) och annonsbeskrivningar,
    returnera de skills som faktiskt förekommer i texten, sorterade efter frekvens.
    """
    if not skills_candidates:
        return []
    joined = " ".join([normalize_text(s) for s in descriptions if isinstance(s, str)])
    found = []
    for sk in skills_candidates:
        sk_norm = normalize_text(sk)
        if len(sk_norm) < 2:
            continue
        if re.search(rf"\b{re.escape(sk_norm)}\b", joined):
            found.append(sk_norm)
    if not found:
        return []
    c = Counter(found)
    return [s for s,_ in c.most_common(top_k)]

def get_skills_for_user_query(query, df, top_k=7):
    """
    Huvudfunktion för kompetenssvar:
    - försök koppla frågan till ett yrke
    - om yrke hittas: hämta skills från taxonomy för det yrket
      - returnera de av dessa skills som faktiskt syns i annonsbeskrivningar (upp till top_k)
      - om inga av dem syns, returnera top_k skills från taxonomy för detta yrke (som generella tips)
    - om inget yrke hittas: försök hitta vanliga skills i beskrivningarna genom att matcha hela taxonomy_skill_set
    """
    occ = find_best_occupation_from_query_or_titles(query, df)
    descriptions = df["description"].fillna("").tolist() if df is not None else []
    if occ:
        # hitta entry
        for entry in taxonomy:
            if entry["occupation"] == occ:
                skills_for_occ = entry["skills"]
                # prefer those present in descriptions
                present = extract_skills_present_in_descriptions(skills_for_occ, descriptions, top_k=top_k)
                if present:
                    return present
                # otherwise return first top_k skills from taxonomy for this occupation
                return skills_for_occ[:top_k] if skills_for_occ else ["Ingen specifik kompetens hittades"]
    # fallback: sök efter taxonomy-skill-ord som förekommer i descriptions
    joined = " ".join([normalize_text(s) for s in descriptions])
    hits = []
    for sk in sorted(taxonomy_skill_set, key=lambda x: -len(x)):  # längre först
        sk_norm = normalize_text(sk)
        if re.search(rf"\b{re.escape(sk_norm)}\b", joined):
            hits.append(sk_norm)
        if len(hits) >= top_k:
            break
    return hits[:top_k] if hits else ["Ingen specifik kompetens hittades"]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df_sorted" not in st.session_state:
    st.session_state.df_sorted = pd.DataFrame()
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_initialized" not in st.session_state:
    st.session_state.chat_initialized = False

def reset_chat():
    st.session_state.chat_history = []
    st.session_state.chat_initialized = False

def initialize_chat():
    if not st.session_state.chat_initialized:
        st.session_state.chat_history.append({
            "role": "bot",
            "content": (
                "👋 Hej där! Jag är din jobbcoach-chatbot 🤖✨\n\n"
                "Jag hjälper dig att utforska de jobbannonser du har sökt 💼.\n"
                "Du kan fråga mig om 🌍 distansjobb, 🗣️ språkkrav, 🎓 utbildning, 🚗 körkort, 🕒 anställningstyp, 📄 antal jobb i en stad eller 🛠️ kompetenser.\n\n"
                "Skriv en fråga eller välj en av knapparna nedan ⬇️"
            )
        })
        st.session_state.chat_initialized = True

col1, col2 = st.columns([0.75, 0.25])

with col1:
    st.title("💬 Jobbcoach Chatbot 🤖")

    user_input = st.text_input("👩‍💼 Jobbcoach Chatbot: Vad vill du jobba med? Du kan söka efter jobbtitel, ort eller företag (t.ex. 'Data Scientist i Stockholm'):")
    num_jobs = st.slider("📊 Hur många annonser vill du hämta?", 5, 50, 10)

    if st.button("🔍 Sök") and user_input.strip():
        st.write(f"🔍 Söker relevanta jobb för: '{user_input}'")
        reset_chat()
        df = get_jobs(query=user_input, limit=num_jobs * 2)
        if df.empty:
            st.error("🚫 Inga jobbannonser hittades.")
        else:
            df = df.head(num_jobs)
            try:
                df = create_embeddings(df)
                qvec = get_embedding(user_input)
                def cosine_similarity(a,b): return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
                df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity(x, qvec))
                df_sorted = df.sort_values(by="similarity", ascending=False)
            except Exception:
                df_sorted = df.copy()
            st.session_state.df_sorted = df_sorted

    if not st.session_state.df_sorted.empty:
        df_sorted = st.session_state.df_sorted
        cities = sorted(df_sorted["city"].fillna("Ingen ort").unique())
        cities.insert(0, "Alla städer 🌆")
        selected_city = st.selectbox("📍 Välj stad (gäller bara annonserna du har sökt):", cities)
        df_display = df_sorted
        if selected_city != "Alla städer 🌆":
            df_display = df_display[df_display["city"] == selected_city]

        count = len(df_display)
        city_text = f" i {selected_city}" if selected_city != "Alla städer 🌆" else ""
        examples = [f"{r['title']} på {r['company']} ({r['city']})" for _, r in df_display.head(3).iterrows()]
        st.subheader("🤖 Chatbot-svar:")
        st.write(
            f"💬 Jag hittade {count} relevanta jobb för '{user_input}'{city_text}. "
            f"Här är några exempel: {'; '.join(examples)}.\n\n"
            f"👉 **Vill du se fler detaljer om något av dessa jobb?** Klicka på länken under varje annons."
            )

        st.subheader("📋 Mest relevanta jobb:")
        for _, row in df_display.iterrows():
            st.markdown(f"**💼 {row['title']}** — {row['company']} ({row['city']})")
            st.write(row.get("description","")[:200] + "…")
            ad_url = row.get("url", f"https://jobsearch.api.jobtechdev.se/ad/{row.get('adId','')}")
            if ad_url:
                st.markdown(f"[📄 Läs mer här]({ad_url})")
            st.write("---")

with col2:
    st.subheader("💬 Chatbot")
    st.caption("💡 Frågor gäller annonserna du har sökt.")
    if st.button("🪄 Öppna/stäng chatten"):
        st.session_state.chat_open = not st.session_state.chat_open
        if st.session_state.chat_open:
            initialize_chat()

    if st.session_state.chat_open:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"🧍‍♂️ **Du:** {msg['content']}")
            else:
                st.markdown(f"🤖 **Bot:** {msg['content']}")

        df_sorted = st.session_state.df_sorted

        st.write("### ⚡ Välj en fråga:")

        if st.button("🏙️ Vilken stad har flest jobb?") and df_sorted is not None and not df_sorted.empty:
            counts = df_sorted["city"].fillna("Ingen ort").value_counts()
            if counts.empty:
                answer = "🤔 Jag hittar inga annonser att analysera."
            else:
                max_count = counts.max()
                top = counts[counts == max_count].index.tolist()
                if len(top) == 1:
                    answer = f"🏙️ Flest jobb finns i: {top[0]} ({max_count} annonser)."
                else:
                    answer = f"🏙️ Flera städer delar förstaplatsen ({max_count} annonser): " + ", ".join(top)
            st.session_state.chat_history.append({"role":"user","content":"Vilken stad har flest jobb?"})
            st.session_state.chat_history.append({"role":"bot","content":answer})
            st.rerun()

        if st.button("📋 Visa tre exempeljobb!") and df_sorted is not None and not df_sorted.empty:
            examples = [f"{r['title']} på {r['company']} ({r['city']})" for _, r in df_sorted.head(3).iterrows()]
            answer = "📋 Här är tre exempeljobb: " + "; ".join(examples)
            st.session_state.chat_history.append({"role":"user","content":"Visa tre exempeljobb!"})
            st.session_state.chat_history.append({"role":"bot","content":answer})
            st.rerun()

        if st.button("🌍 Vilka jobb kan vara på distans?") and df_sorted is not None and not df_sorted.empty:
            rem = df_sorted[
                df_sorted["description"].str.contains(r"\b(distans|remote|hemifrån|fjärr)\b", case=False, na=False)
                & ~df_sorted["description"].str.contains(r"distansutbildning|distanskurs", case=False, na=False)
            ]
            cnt = len(rem)
            if cnt:
                examples = [f"{r['title']} på {r['company']} ({r['city']})" for _, r in rem.head(3).iterrows()]
                answer = f"🌍 Jag hittade {cnt} distansjobb. Exempel: {'; '.join(examples)}."
            else:
                answer = "🤔 Inga distansjobb hittades i de sökta annonserna."
            st.session_state.chat_history.append({"role":"user","content":"Hur många distansjobb finns?"})
            st.session_state.chat_history.append({"role":"bot","content":answer})
            st.rerun()

        chat_input = st.text_input("✍️ Eller skriv egen fråga:")
        if st.button("🚀 Skicka") and chat_input.strip():
            q = chat_input.strip()
            q_low = q.lower()
            st.session_state.chat_history.append({"role":"user","content":q})
            answer = None

            if df_sorted is None or df_sorted.empty:
                answer = "🤖 Jag har ingen annonsdata att analysera just nu. Sök efter jobb först."
            else:
                # antal jobb i stad
                matched_city = None
                for city in df_sorted["city"].fillna("").unique():
                    if city and city.lower() in q_low:
                        matched_city = city
                        break
                if matched_city is not None:
                    cnt = int((df_sorted["city"].fillna("") == matched_city).sum())
                    if cnt == 0:
                        answer = f"📄 Det finns inga jobb i {matched_city} i de sökta annonserna."
                    else:
                        examples = [f"{r['title']} på {r['company']} ({r['city']})" for _, r in df_sorted[df_sorted["city"]==matched_city].head(3).iterrows()]
                        answer = f"📄 Det finns {cnt} jobb i {matched_city}. Exempel: {'; '.join(examples)}."

                # distans / hybrid / plats (exempel + count)
                elif any(tok in q_low for tok in ["distans", "remote", "hemifrån", "fjärr"]):
                    rem = df_sorted[
                        df_sorted["description"].str.contains(r"\b(distans|remote|hemifrån|fjärr)\b", case=False, na=False)
                        & ~df_sorted["description"].str.contains(r"distansutbildning|distanskurs", case=False, na=False)
                    ]
                    cnt = len(rem)
                    if cnt:
                        examples = [f"{r['title']} på {r['company']} ({r['city']})" for _, r in rem.head(3).iterrows()]
                        answer = f"🌍 Jag hittade {cnt} distansjobb. Exempel: {'; '.join(examples)}."
                    else:
                        answer = "🌍 Jag hittade tyvärr inga distansjobb."

                elif "hybrid" in q_low or "både" in q_low:
                    hybrid = df_sorted[df_sorted["description"].str.contains(r"\bhybrid\b", case=False, na=False)]
                    cnt = len(hybrid)
                    if cnt:
                        examples = [f"{r['title']} på {r['company']} ({r['city']})" for _, r in hybrid.head(3).iterrows()]
                        answer = f"💻 Jag hittade {cnt} hybridjobb. Exempel: {'; '.join(examples)}."
                    else:
                        answer = "💻 Inga hybridjobb hittades i de sökta annonserna."

                elif any(tok in q_low for tok in ["plats", "kontor", "på plats"]):
                    onsite = df_sorted[~df_sorted["description"].str.contains(r"\b(distans|remote|hemifrån|fjärr)\b", case=False, na=False)]
                    cnt = len(onsite)
                    if cnt:
                        examples = [f"{r['title']} på {r['company']} ({r['city']})" for _, r in onsite.head(3).iterrows()]
                        answer = f"🏢 Jag hittade {cnt} jobb på plats. Exempel: {'; '.join(examples)}."
                    else:
                        answer = "🏢 Inga platsjobb hittades i de sökta annonserna."

                # anställningstyp
                elif any(tok in q_low for tok in ["heltid","deltid","vikariat","tillsvidare","timanställning","tidsbegränsad"]):
                    types = [w for w in ["heltid","deltid","vikariat","tillsvidare","timanställning","tidsbegränsad"] if w in q_low]
                    desc = df_sorted["description"].fillna("").str.lower()
                    hits = df_sorted[desc.str.contains("|".join(types), na=False)]
                    cnt = len(hits)
                    if cnt:
                        examples = [f"{r['title']} på {r['company']} ({r['city']})" for _, r in hits.head(3).iterrows()]
                        answer = f"🕒 Jag hittade {cnt} jobb som matchar ({', '.join(types)}). Exempel: {'; '.join(examples)}."
                    else:
                        answer = f"🕒 Inga jobb matchar ({', '.join(types)}) i de sökta annonserna."

                # utbildning gymnasie / universitet
                elif any(tok in q_low for tok in ["gymnasie","gymnasiet","gymnasieutbildning"]):
                    hits = df_sorted[df_sorted["description"].str.contains(r"\bgymnasie\b", case=False, na=False)]
                    cnt = len(hits)
                    if cnt:
                        examples = [f"{r['title']} på {r['company']} ({r['city']})" for _, r in hits.head(3).iterrows()]
                        answer = f"📘 {cnt} jobb nämner gymnasieutbildning. Exempel: {'; '.join(examples)}."
                    else:
                        answer = "📘 Inga jobb nämner gymnasieutbildning i de sökta annonserna."

                elif any(tok in q_low for tok in ["universitet","högskola","högre utbildning"]):
                    hits = df_sorted[df_sorted["description"].str.contains(r"\b(universitet|högskola|högre utbildning)\b", case=False, na=False)]
                    cnt = len(hits)
                    if cnt:
                        examples = [f"{r['title']} på {r['company']} ({r['city']})" for _, r in hits.head(3).iterrows()]
                        answer = f"🎓 {cnt} jobb nämner universitet eller högre utbildning. Exempel: {'; '.join(examples)}."
                    else:
                        answer = "🎓 Inga jobb nämner universitet eller högre utbildning i de sökta annonserna."

                # körkort
                elif "körkort" in q_low:
                    hits = df_sorted[df_sorted["description"].str.contains(r"\bkörkort\b", case=False, na=False)]
                    cnt = len(hits)
                    if cnt:
                        examples = [f"{r['title']} på {r['company']} ({r['city']})" for _, r in hits.head(3).iterrows()]
                        answer = f"🚗 {cnt} jobb kräver körkort. Exempel: {'; '.join(examples)}."
                    else:
                        answer = "🚗 Inga jobb kräver körkort i de sökta annonserna."

                # språk
                elif "engelska" in q_low:
                    hits = df_sorted[df_sorted["description"].str.contains(r"\bengelska\b", case=False, na=False)]
                    cnt = len(hits)
                    if cnt:
                        examples = [f"{r['title']} på {r['company']} ({r['city']})" for _, r in hits.head(3).iterrows()]
                        answer = f"🗣️ {cnt} jobb nämner engelska. Exempel: {'; '.join(examples)}."
                    else:
                        answer = "🗣️ Inga jobb nämner engelska i de sökta annonserna."
                elif "svenska" in q_low:
                    hits = df_sorted[df_sorted["description"].str.contains(r"\bsvenska\b", case=False, na=False)]
                    cnt = len(hits)
                    if cnt:
                        examples = [f"{r['title']} på {r['company']} ({r['city']})" for _, r in hits.head(3).iterrows()]
                        answer = f"🗣️ {cnt} jobb nämner svenska. Exempel: {'; '.join(examples)}."
                    else:
                        answer = "🗣️ Inga jobb nämner svenska i de sökta annonserna."

                # kompetenser
                elif any(tok in q_low for tok in ["kompetens","kompetenser","skills","behövs","krävs"]):
                    skills = get_skills_for_user_query(q, df_sorted, top_k=7)
                    answer = "🛠️ Några vanliga kompetenser inom detta område: " + ", ".join(skills)

                # fallback
                if not answer:
                    answer = "Åhnej, detta har jag inte lärt mig än 🙁️ Kan jag kanske hjälpa till med något annat istället?"

            st.session_state.chat_history.append({"role":"bot","content":answer})
            st.rerun()

        if st.button("🧹 Rensa chatten"):
            reset_chat()
            initialize_chat()
            st.rerun()

# streamlit run app.py
