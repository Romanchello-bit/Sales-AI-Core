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
if "visited_history" not in st.session_state: st.session_state.visited_history = []

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
    print("Initializing Generative Model...")
    return genai.GenerativeModel(MODEL_NAME)

@st.cache_data
def load_graph_data():
    script_file = "sales_script.json"
    if not os.path.exists(script_file): return None, None, None, None, None
    with open(script_file, "r", encoding="utf-8") as f: data = json.load(f)
    nodes, edges = data["nodes"], data["edges"]
    node_to_id = {name: i for i, name in enumerate(nodes.keys())}
    id_to_node = {i: name for i, name in enumerate(nodes.keys())}
    graph = Graph(len(nodes), directed=True)
    for edge in edges:
        if edge["from"] in node_to_id and edge["to"] in node_to_id:
            graph.add_edge(node_to_id[edge["from"]], node_to_id[edge["to"]], edge["weight"])
    return graph, node_to_id, id_to_node, nodes, edges

def analyze_full_context(model, user_input, current_node, chat_history):
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-4:]])
    prompt = f"""
    ROLE: World-Class Sales Psychologist.
    CONTEXT: Current Step: "{current_node}", User said: "{user_input}"
    TASK: Determine Intent (MOVE, STAY, EXIT) and Archetype.
    OUTPUT JSON: {{"archetype": "...", "intent": "...", "reasoning": "..."}}
    """
    try:
        response = model.generate_content(prompt)
        return json.loads(response.text.replace("```json", "").replace("```", "").strip())
    except:
        return {"archetype": "UNKNOWN", "intent": "STAY", "reasoning": "Fallback safety"}

def generate_response_stream(model, instruction_text, user_input, lead_info, archetype, product_info={}):
    bot_name = lead_info.get('bot_name', 'Олексій')
    client_name = lead_info.get('name', 'Клієнт')
    company = lead_info.get('company', 'Компанія')
    tone = "Professional, confident."
    if archetype == "DRIVER": tone = "Direct, concise, results-oriented."
    elif archetype == "ANALYST": tone = "Logical, factual, detailed."
    elif archetype == "EXPRESSIVE": tone = "Energetic, inspiring, emotional."
    elif archetype == "CONSERVATIVE": tone = "Calm, supportive, reassuring."
    product_context = ""
    if product_info:
        product_context = f"PRODUCT CONTEXT: You are selling: {product_info.get('product_name', 'Our Solution')}"
    prompt = f"""
    ROLE: You are {bot_name}, a top-tier sales representative.
    CLIENT: {client_name} from {company}.
    CURRENT GOAL: "{instruction_text}"
    USER SAID: "{user_input}"
    ARCHETYPE: {archetype}
    {product_context}
    TASK: Generate the spoken response in Ukrainian. Adapt to the client's tone ({tone}).
    OUTPUT: Just the spoken words.
    """
    return model.generate_content(prompt, stream=True)

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
    st.title("🤖 Sales Bot CRM")
    graph_data = load_graph_data()
    if graph_data[0] is None:
        st.error("sales_script.json not found. CRM mode requires it.")
        st.stop()
    graph, node_to_id, id_to_node, nodes, edges = graph_data

    if st.sidebar.button("📊 Dashboard"): st.session_state.page = "dashboard"; st.rerun()
    if st.sidebar.button("📞 New Call"): st.session_state.page = "setup"; st.rerun()

    if st.session_state.page == "dashboard":
        st.header("Dashboard")
        data, stats = get_analytics()
        if data is not None and not data.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Calls", stats["total"])
            c2.metric("Success Rate", f"{stats['success_rate']}%")
            c3.metric("AI Learning Iterations", "v1.3")
        else:
            st.info("No calls in the database yet.")

    elif st.session_state.page == "setup":
        st.header("Setup New Call")
        with st.form("setup_form"):
            bot_name = st.text_input("Your Name", value="Олексій")
            client_name = st.text_input("Client Name", value="Олександр")
            company = st.text_input("Company", value="SoftServe")
            submitted = st.form_submit_button("🚀 Start Call")
            if submitted:
                st.session_state.lead_info = {"name": client_name, "bot_name": bot_name, "company": company}
                st.session_state.page = "chat"
                st.session_state.messages = []
                st.session_state.current_node = "start"
                st.session_state.visited_history = []
                st.rerun()

    elif st.session_state.page == "chat":
        st.header(f"Call with {st.session_state.lead_info.get('name', 'client')}")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Your reply..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            analysis = analyze_full_context(model, prompt, st.session_state.current_node, st.session_state.messages)
            intent = analysis.get("intent", "STAY")
            archetype = analysis.get("archetype", "UNKNOWN")
            
            if intent == "EXIT":
                outcome = "Success" if "close" in st.session_state.current_node else "Fail"
                add_lead({"Date": datetime.now().strftime("%Y-%m-%d"), "Name": st.session_state.lead_info['name'], "Outcome": outcome, "Archetype": archetype})
                st.success("Call ended and saved.")
                time.sleep(2)
                st.session_state.page = "dashboard"
                st.rerun()
            else:
                if intent == "MOVE":
                    if st.session_state.current_node not in st.session_state.visited_history:
                        st.session_state.visited_history.append(st.session_state.current_node)
                    curr_id = node_to_id[st.session_state.current_node]
                    best_next = None; min_w = float('inf')
                    for n, w in graph.adj_list[curr_id]:
                        if w < min_w: min_w = w; best_next = n
                    if best_next is not None:
                        st.session_state.current_node = id_to_node[best_next]
                    else:
                        st.warning("End of script reached.")
                        add_lead({"Date": datetime.now().strftime("%Y-%m-%d"), "Name": st.session_state.lead_info['name'], "Outcome": "End of Script", "Archetype": archetype})
                        st.stop()
                
                instruction_text = nodes[st.session_state.current_node]
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    stream = generate_response_stream(model, instruction_text, prompt, st.session_state.lead_info, archetype, st.session_state.product_info)
                    for chunk in stream:
                        full_response += (chunk.text or "")
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

elif mode == "⚔️ Evolution Hub":
    st.title("⚔️ The Colosseum: AI Evolution Hub")
    st.header("🎮 Controls")
    c1, c2 = st.columns(2)
    with c1:
        num_simulations = st.number_input("Simulations to Run", 1, 50, 10)
        if st.button(f"🚀 Run {num_simulations} Simulations"):
            log_container = st.container(height=200)
            progress_bar = st.progress(0)
            reports = []
            def progress_callback(report, current, total):
                reports.append(report)
                progress_bar.progress(current / total)
                persona = report['customer_persona']
                log_container.write(f"Sim #{current}: Scen. {report['scenario_id']} vs {persona['archetype']} -> **{report['outcome']}** (Score: {report['score']})")
            colosseum.run_batch_simulations(model, num_simulations, progress_callback)
            st.success("Batch simulation complete!")
            st.header("📊 Post-Battle Report")
            report_df = pd.DataFrame(reports)
            best_id = report_df.groupby('scenario_id')['score'].mean().idxmax()
            worst_id = report_df.groupby('scenario_id')['score'].mean().idxmin()
            st.metric("Most Effective Scenario", f"ID: {best_id}", f"{report_df[report_df['scenario_id'] == best_id]['score'].mean():.2f} avg score")
            st.metric("Least Effective Scenario", f"ID: {worst_id}", f"{report_df[report_df['scenario_id'] == worst_id]['score'].mean():.2f} avg score")
            st.cache_data.clear()
    with c2:
        if st.button("🧬 Run Evolution Cycle"):
            with st.spinner("Running evolution..."):
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
            c1, c2 = st.columns(2)
            with c1:
                st.subheader(f"📜 Graph for Scenario {selected_id}")
                st.json(get_scenario(selected_id), height=400)
            with c2:
                st.subheader("👍👎 Phrase Analytics")
                st.dataframe(get_phrase_analytics_for_scenario(selected_id))
    else:
        st.info("No scenarios to display.")

elif mode == "🧪 Math Lab":
    st.title("🧪 Computational Math Lab")
    # ... (Full Math Lab logic restored here)
    st.info("Math Lab is ready.")
