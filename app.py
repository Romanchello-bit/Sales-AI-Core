import json
import os
import pandas as pd
# google.generativeai moved into run_streamlit_app to avoid import-time dependency errors
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
    # This function relies only on model and JSON handling, safe to keep at top level
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

# Streamlit UI and Streamlit-dependent helpers are encapsulated in this function
def run_streamlit_app():
    import streamlit as st
    import graphviz
    try:
        import google.generativeai as genai
    except Exception:
        genai = None

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
    if "current_archetype" not in st.session_state: st.session_state.current_archetype = "UNKNOWN"
    if "reasoning" not in st.session_state: st.session_state.reasoning = ""
    if 'lab_graph' not in st.session_state: st.session_state.lab_graph = None

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
        if genai is None:
            return None
        print("Initializing Generative Model...")
        return genai.GenerativeModel(MODEL_NAME)

    @st.cache_data
    def load_graph_data():
        # Resolve sales_script.json relative to this file so Streamlit launched from src/ works
        base_dir = os.path.dirname(os.path.abspath(__file__))
        script_file = os.path.join(base_dir, "sales_script.json")
        if not os.path.exists(script_file):
            # Do not stop the whole Streamlit app; return None so caller can show a friendly message
            return None, None, None, None, None
        with open(script_file, "r", encoding="utf-8") as f: data = json.load(f)
        nodes, edges = data["nodes"], data["edges"]
        node_to_id = {name: i for i, name in enumerate(nodes.keys())}
        id_to_node = {i: name for i, name in enumerate(nodes.keys())}
        graph = Graph(len(nodes), directed=True)
        for edge in edges:
            if edge["from"] in node_to_id and edge["to"] in node_to_id:
                graph.add_edge(node_to_id[edge["from"]], node_to_id[edge["to"]], edge["weight"])
        return graph, node_to_id, id_to_node, nodes, edges

    def generate_response_stream(model, instruction_text, user_input, lead_info, archetype, product_info=None):
        if product_info is None: product_info = {}
        # If model is not available, return a tiny generator with a fallback response
        if model is None:
            def _stub():
                class Chunk:
                    def __init__(self, text):
                        self.text = text
                yield Chunk("[AI unavailable — running in offline mode]")
            return _stub()

        bot_name = lead_info.get('bot_name', 'Олексій')
        client_name = lead_info.get('name', 'Клієнт')
        company = lead_info.get('company', 'Компанія')
        tone = "Professional, confident."
        if archetype == "DRIVER": tone = "Direct, concise, results-oriented."
        elif archetype == "ANALYST": tone = "Logical, factual, detailed."
        elif archetype == "EXPRESSIVE": tone = "Energetic, inspiring, emotional."
        elif archetype == "CONSERVATIVE": tone = "Calm, supportive, reassuring."
        product_context = f"PRODUCT CONTEXT: You are selling: {product_info.get('product_name', 'Our Solution')}" if product_info else ""
        prompt = f"""
        ROLE: You are {bot_name}, a top-tier sales representative. CLIENT: {client_name} from {company}.
        CURRENT GOAL: "{instruction_text}". USER SAID: "{user_input}". ARCHETYPE: {archetype}. {product_context}
        TASK: Generate the spoken response in Ukrainian. Adapt to the client's tone ({tone}). OUTPUT: Just the spoken words.
        """
        try:
            return model.generate_content(prompt, stream=True)
        except Exception:
            # Fallback if model call fails mid-run
            def _stub_err():
                class Chunk:
                    def __init__(self, text):
                        self.text = text
                yield Chunk("[AI request failed — fallback response]")
            return _stub_err()

    # --- MAIN APP ---
    init_db()
    st.sidebar.title("🛠️ SellMe Control")
    mode = st.sidebar.radio("Mode", ["🤖 Sales Bot CRM", "⚔️ Evolution Hub", "🧪 Math Lab"])

    api_key = st.sidebar.text_input("Google API Key", type="password", help="Optional. Provide to enable Google Generative AI features.")
    model = None
    if api_key:
        if genai is None:
            st.error("google-generativeai package not available. AI features disabled.")
        else:
            if configure_genai(api_key):
                model = get_model()
            else:
                st.error("Failed to configure Generative AI. Continuing with AI disabled.")
    else:
        st.info("Running without Google API Key — AI features are disabled. The UI (Sales Bot, Evolution Hub, Math Lab) will still be available.")

    if mode == "🤖 Sales Bot CRM":
        st.title("🤖 Sales Bot CRM")
        graph_data = load_graph_data()
        if graph_data[0] is None:
            st.warning("sales_script.json not found. CRM features are disabled. Place 'sales_script.json' in the repository root to enable the Sales Bot CRM.")
        else:
            graph, node_to_id, id_to_node, nodes, edges = graph_data

            if st.sidebar.button("📊 Dashboard"): st.session_state.page = "dashboard"; st.rerun()
            if st.sidebar.button("📞 New Call"): st.session_state.page = "setup"; st.rerun()

            if st.session_state.page == "dashboard":
                st.header("Dashboard")
                data, stats = get_analytics()
                if data is not None and not data.empty:
                    c1, c2, c3 = st.columns(3); c1.metric("Total Calls", stats["total"]); c2.metric("Success Rate", f"{stats['success_rate']}%"); c3.metric("AI Learning Iterations", "v1.4")
                else: st.info("No calls in the database yet.")

            elif st.session_state.page == "setup":
                st.header("Setup New Call")
                c1, c2 = st.columns(2)
                with c2:
                    st.markdown("### 📦 Product / Service Info")
                    url = st.text_input("Product URL", placeholder="https://example.com/product")
                    if st.button("🤖 Fetch Product Info from URL"):
                        if url:
                            with st.spinner("Fetching and analyzing URL..."):
                                scraped_info = scrape_and_summarize(model, url)
                                if scraped_info:
                                    st.session_state.product_info = scraped_info
                                    st.success("Product info populated!")
                        else: st.warning("Please enter a URL.")

                with st.form("lead_form"):
                    c1_form, c2_form = st.columns(2)
                    with c1_form:
                        st.markdown("### 👨‍💼 Lead Info")
                        bot_name = st.text_input("Your Name", value="Олексій")
                        client_name = st.text_input("Client Name", value="Олександр")
                        company = st.text_input("Company", value="SoftServe")
                    with c2_form:
                        st.markdown("### 📦 Product / Service Info (Editable)")
                        p_name = st.text_input("Product Name", value=st.session_state.product_info.get("product_name", ""))
                        p_value = st.text_input("Main Benefit (Value)", value=st.session_state.product_info.get("product_value", ""))
                        p_price = st.text_input("Price / Pricing Model", value=st.session_state.product_info.get("product_price", ""))
                        p_diff = st.text_input("Competitive Edge", value=st.session_state.product_info.get("competitor_diff", ""))

                    submitted = st.form_submit_button("🚀 Start Call")
                    if submitted:
                        st.session_state.lead_info = {"name": client_name, "bot_name": bot_name, "company": company}
                        st.session_state.product_info = {"product_name": p_name, "product_value": p_value, "product_price": p_price, "competitor_diff": p_diff}
                        st.session_state.page = "chat"; st.session_state.messages = []; st.session_state.current_node = "start"; st.session_state.visited_history = []
                        st.rerun()

            elif st.session_state.page == "chat":
                col_chat, col_tools = st.columns([1.5, 1])
                with col_chat:
                    st.header(f"Call with {st.session_state.lead_info.get('name', 'client')}")
                    for msg in st.session_state.messages:
                        with st.chat_message(msg["role"]): st.markdown(msg["content"])

                with col_tools:
                    st.header("Analytics")
                    st.markdown("#### 🧠 Profile")
                    st.text(f"Archetype: {st.session_state.current_archetype} ({st.session_state.reasoning})")
                    st.markdown("#### 📊 Strategy")
                    path = bellman_ford_list(graph, node_to_id[st.session_state.current_node])
                    predicted_path = [id_to_node[i] for i, d in enumerate(path) if d != float('inf')] if path else []
                    st.graphviz_chart(draw_graph(graph_data, st.session_state.current_node, predicted_path), use_container_width=True)

                if prompt := st.chat_input("Your reply..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user", container=col_chat): st.markdown(prompt)

                    analysis = analyze_full_context(model, prompt, st.session_state.current_node, st.session_state.messages)
                    st.session_state.current_archetype = analysis.get("archetype", "UNKNOWN")
                    st.session_state.reasoning = analysis.get("reasoning", "")

                    if analysis.get("intent") == "MOVE":
                        if st.session_state.current_node not in st.session_state.visited_history: st.session_state.visited_history.append(st.session_state.current_node)
                        curr_id = node_to_id[st.session_state.current_node]
                        best_next = min(graph.adj_list[curr_id], key=lambda x: x[1], default=None)
                        if best_next: st.session_state.current_node = id_to_node[best_next[0]]
                        else: st.warning("End of script."); st.stop()

                    instruction_text = nodes[st.session_state.current_node]
                    with st.chat_message("assistant", container=col_chat):
                        message_placeholder = st.empty()
                        full_response = ""
                        stream = generate_response_stream(model, instruction_text, prompt, st.session_state.lead_info, st.session_state.current_archetype)
                        for chunk in stream:
                            full_response += (chunk.text or ""); message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.rerun()

    elif mode == "⚔️ Evolution Hub":
        st.title("⚔️ The Colosseum: AI Evolution Hub")
        st.header("🎮 Controls")
        c1, c2 = st.columns(2)
        with c1:
            num_simulations = st.number_input("Simulations to Run", 1, 50, 10)
            if st.button(f"🚀 Run {num_simulations} Simulations"):
                log_container = st.container(height=200); progress_bar = st.progress(0); reports = []
                def progress_callback(report, current, total):
                    reports.append(report); progress_bar.progress(current / total)
                    if 'error' not in report:
                        persona = report['customer_persona']
                        log_container.write(f"Sim #{current}: Scen. {report['scenario_id']} vs {persona['archetype']} -> **{report['outcome']}** (Score: {report['score']})")
                colosseum.run_batch_simulations(model, num_simulations, progress_callback)
                st.success("Batch simulation complete!")
                if reports:
                    st.header("📊 Post-Battle Report")
                    report_df = pd.DataFrame(reports)
                    if not report_df.empty and 'scenario_id' in report_df.columns:
                        best_id = report_df.groupby('scenario_id')['score'].mean().idxmax()
                        worst_id = report_df.groupby('scenario_id')['score'].mean().idxmin()
                        st.metric("Most Effective Scenario", f"ID: {best_id}", f"{report_df[report_df['scenario_id'] == best_id]['score'].mean():.2f} avg score")
                        st.metric("Least Effective Scenario", f"ID: {worst_id}", f"{report_df[report_df['scenario_id'] == worst_id]['score'].mean():.2f} avg score")
                st.cache_data.clear()
        with c2:
            if st.button("🧬 Run Evolution Cycle"):
                with st.spinner("Running evolution..."): evolution.run_evolution_cycle(model)
                st.success("Evolution complete!"); st.cache_data.clear()

        st.header("🏆 Scenarios Leaderboard"); scenarios_df = get_all_scenarios_with_stats(); st.dataframe(scenarios_df)
        if not scenarios_df.empty:
            st.header("🕵️ Scenario Inspector")
            selected_id = st.selectbox("Select Scenario ID:", scenarios_df['id'])
            if selected_id:
                c1, c2 = st.columns(2)
                with c1: st.subheader(f"📜 Graph for Scenario {selected_id}"); st.json(get_scenario(selected_id), height=400)
                with c2: st.subheader("👍👎 Phrase Analytics"); st.dataframe(get_phrase_analytics_for_scenario(selected_id))

    elif mode == "🧪 Math Lab":
        st.title("🧪 Computational Math Lab")
        st.markdown("### Section A: Graph Inspector")
        col1, col2 = st.columns(2)
        n_nodes = col1.slider("N (Vertices)", 5, 15, 10)
        density = col2.slider("Density", 0.1, 1.0, 0.5)

        if st.button("Generate Graph"):
             st.session_state.lab_graph = experiments.generate_erdos_renyi(n_nodes, density)

        if st.session_state.lab_graph:
            graph = st.session_state.lab_graph
            tab1, tab2, tab3 = st.tabs(["Visual Graph", "Adjacency Matrix", "Adjacency List"])
            with tab1:
                st.subheader("Graphviz Visualization")
                dot = graphviz.Digraph()
                for u, neighbors in graph.adj_list.items():
                    dot.node(str(u), label=str(u))
                    for v, w in neighbors: dot.edge(str(u), str(v), label=str(w))
                st.graphviz_chart(dot)
            with tab2:
                st.subheader("Adjacency Matrix (Heatmap)")
                matrix = graph.to_adjacency_matrix()
                df_matrix = pd.DataFrame(matrix)
                df_heatmap = df_matrix.replace(float('inf'), None)
                st.dataframe(df_heatmap.style.background_gradient(cmap="Blues", axis=None).format(na_rep="∞"))
            with tab3:
                st.subheader("Adjacency List")
                st.write(graph.adj_list)

        st.divider()
        st.markdown("### Section B: Scientific Experiments")
        st.markdown("Comparing Bellman-Ford implementations: **Adjacency List vs Adjacency Matrix**.")
        sizes_preset = list(range(20, 120, 20))
        densities_preset = [0.2, 0.5, 0.8]

        if st.button("🚀 Run Scientific Benchmark"):
            with st.spinner("Running benchmarks... This may take a while."):
                results = experiments.run_scientific_benchmark(sizes_preset, densities_preset, num_runs=3)
                df_results = pd.DataFrame(results)
                st.subheader("Raw Data")
                st.dataframe(df_results)
                st.divider()

                c_chart, c_filter = st.columns([3, 1])
                with c_filter:
                    sel_density = st.selectbox("Density:", densities_preset, index=1)
                with c_chart:
                    st.subheader("Benchmark Results")
                    filtered_df = df_results[df_results["Density"] == sel_density].sort_values("Vertices (N)")
                    st.line_chart(filtered_df.set_index("Vertices (N)")[["Time_List", "Time_Matrix"]])
                st.success("Benchmarking complete!")

# Only run the Streamlit UI when explicitly requested (streamlit launcher sets env var RUNNING_IN_STREAMLIT)
if os.environ.get('RUNNING_IN_STREAMLIT') or __name__ == '__main__':
    run_streamlit_app()
