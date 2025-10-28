#!/bin/bash
echo "🔹 Custom startup for Azure - installing dependencies and running Streamlit..."
cd /home/site/wwwroot

# Installera beroenden i användarens katalog (Azure tillåter bara skrivning där)
pip install --no-cache-dir --user -r requirements.txt

# Kör Streamlit från rätt plats
~/.local/bin/streamlit run app.py --server.port 8000 --server.address 0.0.0.0
