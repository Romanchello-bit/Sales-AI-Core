import json
import os
import pandas as pd
from graph_module import Graph
from algorithms import bellman_ford_list
from leads_manager import get_analytics
from database import (
    init_db, get_all_scenarios_with_stats, get_scenario, get_phrase_analytics_for_scenario
)
import colosseum
import evolution
import experiments
import requests
from bs4 import BeautifulSoup

# Non-Streamlit helper functions that don't depend on Streamlit
def analyze_full_context(model, user_input, current_node, chat_history):
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-4:]])
    prompt = f"""
    ROLE: World-Class Sales Psychologist. CONTEXT: Current Step: "{current_node}", User said: "{user_input}"
    TASK: Determine Intent (MOVE, STAY, EXIT) and Archetype.
    OUTPUT JSON: {{"archetype": "...", "intent": "...", "reasoning": "..."}}
    """
    try:
        response = model.generate_content(prompt)
        return json.loads(response.text.replace("```json", "").replace("```", "").strip())
    except Exception:
        return {"archetype": "UNKNOWN", "intent": "STAY", "reasoning": "Fallback safety"}

def run_streamlit_app():
    import streamlit as st
    import graphviz
    import google.generativeai as genai
    from datetime import datetime
    import time
    import random
    import matplotlib.pyplot as plt

    st.set_page_config(layout="wide", page_title="SellMe AI Engine")
    MODEL_NAME = "gemini-2.5-flash"

    # --- SESSION STATE INIT ---
    # ... (All session state initializations)

    @st.cache_resource
    def configure_genai(api_key):
        try:
            genai.configure(api_key=api_key)
            return True
        except Exception as e:
            st.error(f"Failed to configure API Key: {e}")
            return False

    @st.cache_resource
    def get_model():
        print("Initializing Generative Model...")
        return genai.GenerativeModel(MODEL_NAME)

    # ... (All other helper functions like load_graph_data, draw_graph, etc. go here)

    init_db()
    st.sidebar.title("🛠️ SellMe Control")
    mode = st.sidebar.radio("Mode", ["🤖 Sales Bot CRM", "⚔️ Evolution Hub", "🧪 Math Lab"], index=1)

    api_key = st.sidebar.text_input("Google API Key", type="password", help="Required for all modes.")
    if not api_key:
        st.warning("Please enter your Google API Key to proceed."); st.stop()
    if not configure_genai(api_key):
        st.stop()

    model = get_model()
    
    # ... (The full logic for all three modes goes here)

if __name__ == "__main__":
    run_streamlit_app()
