#!/bin/bash
# Activate venv and install dependencies
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
