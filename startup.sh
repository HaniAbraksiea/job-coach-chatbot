#!/bin/bash
echo "🔹 Starting Streamlit app inside Azure..."
cd /home/site/wwwroot

# Installera beroenden (lokalt i användarens pip-cache)
pip install --no-cache-dir --user -r requirements.txt

# Kör appen
~/.local/bin/streamlit run app.py --server.port 8000 --server.address 0.0.0.0
