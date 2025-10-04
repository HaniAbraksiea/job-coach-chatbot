# fetch_jobs.py
import pandas as pd
import requests

def get_jobs(query=None, limit=5):
    """
    Hämtar riktiga jobbannonser från JobTech API.
    """

    url = "https://jobsearch.api.jobtechdev.se/search"
    params = {
        "q": query if query else "",
        "limit": limit
    }

    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        hits = resp.json().get("hits", [])
    except Exception as e:
        print("⚠️ Fel vid API-anrop:", e)
        return pd.DataFrame([])

    data = {
        "title": [hit.get("headline", "Ingen titel") for hit in hits],
        "company": [hit.get("employer", {}).get("name", "Ingen arbetsgivare") for hit in hits],
        "city": [hit.get("workplace_address", {}).get("municipality", "Ingen ort") for hit in hits],
        "description": [hit.get("description", {}).get("text", "Ingen beskrivning") for hit in hits],
        "url": [hit.get("webpage_url", "#") for hit in hits]
    }

    df = pd.DataFrame(data)
    df['city'] = df['city'].fillna("Ingen ort")
    return df



# python fetch_jobs.py
