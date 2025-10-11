import json

def load_skills():
    """Läser in och returnerar en lista med kompetenser från skills.json"""
    try:
        with open("skills.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        skills = []
        for item in data.get("data", {}).get("concepts", []):
            label = item.get("preferred_label")
            if label:
                skills.append(label.lower())  # gör små bokstäver för enklare matchning

        print(f"Laddade {len(skills)} kompetenser.")
        return skills

    except Exception as e:
        print(f"Misslyckades med att läsa skills.json: {e}")
        return []

if __name__ == "__main__":
    skills = load_skills()
    print(skills[:20])  # visar de första 20 som test


# python load_skills.py
