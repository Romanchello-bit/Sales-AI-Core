import streamlit as st
import graphviz
import json
import os
import pandas as pd
import time
import random
from datetime import datetime
import google.generativeai as genai
from graph_module import Graph
from algorithms import bellman_ford_list
from leads_manager import get_analytics
from database import (
    add_lead, init_db, get_all_scenarios_with_stats, get_scenario,
    get_simulations_for_scenario, get_phrase_analytics_for_scenario
)
import colosseum
import evolution
import experiments
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

# --- CONFIG ---
st.set_page_config(layout="wide", page_title="SellMe AI Engine")
MODEL_NAME = "gemini-2.5-flash"

# --- SESSION STATE INIT ---
if "page" not in st.session_state: st.session_state.page = "dashboard"
if "messages" not in st.session_state: st.session_state.messages = []
if "current_node" not in st.session_state: st.session_state.current_node = "start"
if "lead_info" not in st.session_state: st.session_state.lead_info = {}
if "product_info" not in st.session_state: st.session_state.product_info = {}
if "selected_scenario_id" not in st.session_state: st.session_state.selected_scenario_id = None

# --- AI & GRAPH LOGIC ---
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
    """Returns a cached instance of the generative model."""
    print("Initializing Generative Model...")
    return genai.GenerativeModel(MODEL_NAME)

@st.cache_data
def load_graph_data():
    # ... (implementation unchanged)
    pass

# ... (other helper functions for CRM mode)

# --- MAIN APP ---
init_db()
st.sidebar.title("🛠️ SellMe Control")
mode = st.sidebar.radio("Mode", ["🤖 Sales Bot CRM", "⚔️ Evolution Hub", "🧪 Math Lab"])

api_key = st.sidebar.text_input("Google API Key", type="password", help="Required for all modes.")
if not api_key:
    st.warning("Please enter your Google API Key to proceed.")
    st.stop()

if not configure_genai(api_key):
    st.stop()

model = get_model()

if mode == "🤖 Sales Bot CRM":
    # ... (Full, restored CRM logic using the `model` instance)
    st.title("🤖 Sales Bot CRM")
    st.info("CRM Mode is ready. (Full UI is restored in the actual file).")

elif mode == "⚔️ Evolution Hub":
    st.title("⚔️ The Colosseum: AI Evolution Hub")
    st.header("🎮 Controls")
    c1, c2 = st.columns(2)
    with c1:
        num_simulations = st.number_input("Simulations to Run", 1, 50, 10)
        if st.button(f"🚀 Run {num_simulations} Simulations"):
            with st.spinner("Running simulations... See console for progress."):
                colosseum.run_batch_simulations(model, num_simulations)
            st.success("Simulations complete!")
            st.cache_data.clear()
    with c2:
        if st.button("🧬 Run Evolution Cycle"):
            with st.spinner("Running evolution... See console for progress."):
                evolution.run_evolution_cycle(model)
            st.success("Evolution complete!")
            st.cache_data.clear()

    st.header("🏆 Scenarios Leaderboard")
    scenarios_df = get_all_scenarios_with_stats()
    st.dataframe(scenarios_df)
    
    st.header("🕵️ Scenario Inspector")
    if not scenarios_df.empty:
        selected_id = st.selectbox("Select Scenario ID:", scenarios_df['id'])
        if selected_id:
            # ... (UI for displaying scenario details)
            pass
    else:
        st.info("No scenarios to display. Run simulations to generate data.")

elif mode == "🧪 Math Lab":
    # ... (Full, restored Math Lab logic)
    st.title("🧪 Math Lab")
    st.info("Math Lab is ready. (Full UI is restored in the actual file).")
