# app.py
import streamlit as st
from retriever import search_jobs  # vår funktion för RAG-sökning

st.title("Jobbcoach Chatbot (RAG) — Prototype")

user_input = st.text_input("Skriv din fråga om jobb (t.ex. 'Jag kan Python'): ")

if st.button("Sök"):
    st.write("🔍 Söker relevanta jobb för:", user_input)

    results = search_jobs(user_input, use_dummy=True)  # dummy nu
    st.subheader("Mest relevanta jobb:")

    for title, desc in results:
        st.markdown(f"**{title}**")
        st.write(desc[:200] + "…")


#streamlit run app.py
