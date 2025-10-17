import json

def load_taxonomy(path="ssyk-level-4-groups-with-related-skills.json"):
    """
    Läser in SSYK-taxonomi (JobTech-format) och returnerar:
      - taxonomy: lista av dicts [{occupation, skills[]}, ...]
      - taxonomy_skill_set: set() av alla kompetenser (för sökning i text)
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        taxonomy = []
        taxonomy_skill_set = set()

        # Data kan vara antingen {"data": {"concepts": [...]}} eller {"concepts": [...]}
        concepts = []
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], dict) and "concepts" in data["data"]:
                concepts = data["data"]["concepts"]
            elif "concepts" in data:
                concepts = data["concepts"]
            else:
                # ibland en lista av poster i "data"
                if isinstance(data.get("data"), list):
                    concepts = data["data"]
        elif isinstance(data, list):
            concepts = data

        for entry in concepts:
            occ_label = entry.get("preferred_label") or entry.get("label") or ""
            if not occ_label:
                continue
            occ_label = occ_label.strip().lower()

            related = entry.get("related", [])
            skills = []
            for rel in related:
                if not isinstance(rel, dict):
                    continue
                label = rel.get("preferred_label") or rel.get("label")
                if isinstance(label, dict):
                    label = label.get("sv") or label.get("en")
                if isinstance(label, str):
                    skill_label = label.strip().lower()
                    if len(skill_label) > 2:
                        skills.append(skill_label)
                        taxonomy_skill_set.add(skill_label)

            taxonomy.append({
                "occupation": occ_label,
                "skills": list(set(skills))
            })

        print(f"✅ Laddade {len(taxonomy)} yrken med kompetenser ({len(taxonomy_skill_set)} unika skills) från {path}")
        return taxonomy, taxonomy_skill_set

    except Exception as e:
        print(f"❌ Misslyckades med att läsa {path}: {e}")
        return [], set()


if __name__ == "__main__":
    taxonomy, taxonomy_skill_set = load_taxonomy()
    print(json.dumps(taxonomy[:2], ensure_ascii=False, indent=2))
    print(f"\nTotalt {len(taxonomy)} yrken, {len(taxonomy_skill_set)} unika kompetenser.")


# python load_taxonomy.py

