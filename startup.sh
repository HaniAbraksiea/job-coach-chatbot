#!/bin/bash
cd /home/site/wwwroot
echo "Starting Streamlit app..."
# Installera beroenden (för säkerhets skull)
pip install -r requirements.txt
# Starta appen på port 8000 (Azure kräver denna)
streamlit run app.py --server.port 8000 --server.address 0.0.0.0
