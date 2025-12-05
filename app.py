import streamlit as st
import graphviz
import json
import os
import pandas as pd
import time
from datetime import datetime
import google.generativeai as genai
from graph_module import Graph
from algorithms import bellman_ford_list
from leads_manager import get_analytics
import experiments

# --- CONFIG ---
st.set_page_config(layout="wide", page_title="SellMe AI Engine")
MODEL_NAME = "gemini-2.5-flash"
LEADS_FILE = "leads_database.csv"

# --- SESSION STATE INIT ---
if "page" not in st.session_state: st.session_state.page = "dashboard"
if "messages" not in st.session_state: st.session_state.messages = []
if "current_node" not in st.session_state: st.session_state.current_node = "start"
if "lead_info" not in st.session_state: st.session_state.lead_info = {}
if "product_info" not in st.session_state: st.session_state.product_info = {} # NEW: Product Context
if "visited_history" not in st.session_state: st.session_state.visited_history = []
if "current_archetype" not in st.session_state: st.session_state.current_archetype = "UNKNOWN"
if "reasoning" not in st.session_state: st.session_state.reasoning = ""
if "current_sentiment" not in st.session_state: st.session_state.current_sentiment = 0.0
if "checklist" not in st.session_state:
    st.session_state.checklist = {
        "Identify Customer": False,
        "Determine Objectives": False,
        "Outline Advantages": False,
        "Keep it Brief": True,
        "Experiment/Revise": False
    }

# --- DATA MANAGER ---
def init_db():
    if not os.path.exists(LEADS_FILE):
        df = pd.DataFrame(columns=[
            "Date", "Name", "Company", "Type", "Context", 
            "Pain Point", "Budget", "Outcome", "Summary"
        ])
        df.to_csv(LEADS_FILE, index=False)

def save_lead_to_db(lead_info, chat_history, outcome):
    init_db()
    model = genai.GenerativeModel(MODEL_NAME)
    chat_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history])
    
    prompt = f"""
    Analyze this sales conversation:
    {chat_text}
    
    Extract these fields in JSON format:
    - pain_point: What is the client's main problem?
    - budget: Did they mention money/price sensitivity?
    - summary: 1 sentence summary of the call.
    """
    try:
        response = model.generate_content(prompt)
        ai_data = response.text
    except:
        ai_data = "AI Extraction Failed"

    new_row = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Name": lead_info.get("name"),
        "Company": lead_info.get("company"),
        "Type": lead_info.get("type"),
        "Context": lead_info.get("context"),
        "Pain Point": "AI Analysis Pending",
        "Budget": "Unknown",
        "Outcome": outcome,
        "Summary": f"Call with {len(chat_history)} messages. {outcome}"
    }
    
    df = pd.read_csv(LEADS_FILE)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(LEADS_FILE, index=False)

# --- AI & GRAPH LOGIC ---
def configure_genai(api_key):
    try:
        genai.configure(api_key=api_key)
        return True
    except: return False

def load_graph_data():
    script_file = "sales_script_learned.json" if os.path.exists("sales_script_learned.json") else "sales_script.json"
    with open(script_file, "r", encoding="utf-8") as f: data = json.load(f)
    nodes = data["nodes"]
    edges = data["edges"]
    node_to_id = {name: i for i, name in enumerate(nodes.keys())}
    id_to_node = {i: name for i, name in enumerate(nodes.keys())}
    graph = Graph(len(nodes), directed=True)
    for edge in edges:
        if edge["from"] in node_to_id and edge["to"] in node_to_id:
            graph.add_edge(node_to_id[edge["from"]], node_to_id[edge["to"]], edge["weight"])
    return graph, node_to_id, id_to_node, nodes, edges

def get_predicted_path(graph, start_id, target_id, id_to_node, node_to_id):
    visited_ids = [node_to_id[n] for n in st.session_state.get('visited_history', []) if n in node_to_id]
    client_type = st.session_state.lead_info.get('type', 'B2B')
    
    dist = bellman_ford_list(graph, start_id, visited_nodes=visited_ids, client_type=client_type)
    if dist[target_id] == float('inf'): return []
    path = [target_id]
    curr = target_id
    attempts = 0
    while curr != start_id and attempts < 200:
        found = False
        attempts += 1
        for u in range(graph.num_vertices):
            for v, w in graph.adj_list[u]:
                if v == curr and dist[v] == dist[u] + w:
                    path.append(u); curr = u; found = True; break
            if found: break
        if not found: break
    return [id_to_node[i] for i in reversed(path)]

def analyze_full_context(model, user_input, current_node, chat_history):
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-4:]])
    
    prompt = f"""
    ROLE: World-Class Sales Psychologist.
    
    CONTEXT:
    Current Step: "{current_node}"
    User said: "{user_input}"
    
    TASK: Determine Intent (MOVE, STAY, EXIT).
    
    CRITICAL RULES FOR INTENT:
    1. **EXIT** triggers ONLY if user is HOSTILE or EXPLICITLY ends the call.
    2. **STAY** (Objection Handling) triggers for ANY resistance.
    3. **MOVE** triggers only if user agrees or answers a question positively.
    
    OUTPUT JSON format:
    {{
        "archetype": "DRIVER" | "ANALYST" | "EXPRESSIVE" | "CONSERVATIVE",
        "intent": "MOVE" | "STAY" | "EXIT",
        "reasoning": "Why?"
    }}
    """
    try:
        response = model.generate_content(prompt)
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except:
        return {"archetype": "UNKNOWN", "intent": "STAY", "reasoning": "Fallback safety"}

def generate_response(model, instruction_text, user_input, intent, lead_info, archetype, product_info={}):
    """
    Generates a generic or product-specific response.
    """
    bot_name = lead_info.get('bot_name', 'Олексій')
    client_name = lead_info.get('name', 'Клієнт')
    company = lead_info.get('company', 'Компанія')
    context = lead_info.get('context', 'Cold')
    
    tone = "Professional, confident."
    if archetype == "DRIVER": tone = "Direct, concise, results-oriented (Time is money)."
    elif archetype == "ANALYST": tone = "Logical, factual, detailed."
    elif archetype == "EXPRESSIVE": tone = "Energetic, inspiring, emotional."
    elif archetype == "CONSERVATIVE": tone = "Calm, supportive, reassuring."
    
    length_instruction = "Keep it concise."
    if "Cold" in context: length_instruction = "Extremely short and punchy (Elevator Pitch)."
    
    # NEW: Product Context Injection
    product_context = ""
    if product_info:
        product_context = f"""
        PRODUCT CONTEXT:
        You are selling: {product_info.get('product_name', 'Our Solution')}
        Value Proposition: {product_info.get('product_value', 'High Value')}
        Pricing: {product_info.get('product_price', 'Custom Pricing')}
        Competitive Edge: {product_info.get('competitor_diff', 'Best in class')}
        
        CRITICAL INSTRUCTION:
        Whenever the script graph says "Pitch", "Price", or "Objection", use the PRODUCT CONTEXT above. Do NOT invent fake features.
        """

    prompt = f"""
    ROLE: You are {bot_name}, a top-tier sales representative at SellMe AI.
    CLIENT: {client_name} from {company}.
    CURRENT GOAL (INSTRUCTION): "{instruction_text}"
    USER SAID: "{user_input}"
    INTENT DETECTED: {intent}
    ARCHETYPE: {archetype}
    
    {product_context}
    
    TASK: Generate the spoken response in Ukrainian.
    
    CRITICAL RULES:
    1. DO NOT output the instruction itself. ACT IT OUT.
    2. Adapt to the client's tone ({tone}).
    3. {length_instruction}
    4. If INTENT is 'STAY' (Objection): Acknowledge the objection, reframe it, and steer back to the goal.
    5. If INTENT is 'MOVE': Validate the user's answer and transition smoothly to the goal.
    
    OUTPUT: Just the spoken words. No "Option 1", no quotes.
    """
    
    try:
        return model.generate_content(prompt).text.strip()
    except Exception as e:
        return f"[System Error: {e}]"

def generate_greeting(model, start_instruction, lead_info, product_info={}):
    bot_name = lead_info.get('bot_name', 'Manager')
    client_name = lead_info.get('name', 'Client')
    context = lead_info.get('context', 'Cold')
    
    # NEW: Product Context Injection
    product_context = ""
    if product_info:
        product_context = f"""
        PRODUCT CONTEXT:
        You are selling: {product_info.get('product_name', 'Our Solution')}
        """

    prompt = f"""
    ROLE: Sales Rep {bot_name}.
    CLIENT: {client_name}.
    CONTEXT: {context} call.
    INSTRUCTION: "{start_instruction}"
    
    {product_context}
    
    TASK: Generate the opening line.
    - If Cold Call: Be brief, aggressive (pattern interrupt).
    - If Warm Call: Be welcoming, reference the application.
    - Language: Ukrainian.
    """
    try:
        return model.generate_content(prompt).text.strip()
    except:
        return f"Алло, {client_name}? Це {bot_name}."


def train_brain():
    df, _ = get_analytics()
    if df is None or df.empty or "Transcript" not in df.columns:
        return "Недостатньо даних для навчання."

    graph, node_to_id, id_to_node, nodes, edges = load_graph_data()
    success_bonuses = {}
    
    for index, row in df.iterrows():
        is_success = (row["Outcome"] == "Success")
        transcript = str(row["Transcript"])
        for node_name, node_text in nodes.items():
            snippet = node_text[:20] 
            if snippet in transcript:
                if is_success:
                    success_bonuses[node_name] = success_bonuses.get(node_name, 0) + 1
                else:
                    success_bonuses[node_name] = success_bonuses.get(node_name, 0) - 1

    new_edges = []
    changes_log = []
    for edge in edges:
        u_name, v_name = edge["from"], edge["to"]
        old_weight = edge["weight"]
        new_weight = old_weight
        score = success_bonuses.get(v_name, 0)
        
        if score > 0: new_weight *= 0.9
        elif score < 0: new_weight *= 1.1
            
        new_weight = max(1, min(new_weight, 100))
        new_edges.append({"from": u_name, "to": v_name, "weight": round(new_weight, 2)})
        
        if old_weight != new_weight:
            changes_log.append(f"{u_name}->{v_name}: {old_weight} -> {new_weight}")

    learned_data = {"nodes": nodes, "edges": new_edges}
    with open("sales_script_learned.json", "w", encoding="utf-8") as f:
        json.dump(learned_data, f, ensure_ascii=False, indent=2)
        
    return f"Brain Updated! {len(changes_log)} weights adjusted based on {len(df)} calls."

# --- UI COMPONENTS ---
def draw_graph(graph_data, current_node, predicted_path):
    nodes = graph_data[3]
    edges = graph_data[4]
    dot = graphviz.Digraph()
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.3', ranksep='0.4', bgcolor='transparent')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='11', width='2.5', height='0.5', margin='0.1')
    dot.attr('edge', fontname='Arial', fontsize='9', arrowsize='0.6')

    for n in nodes:
        fill = '#F7F9F9'; color = '#BDC3C7'; pen = '1'; font = '#424949'
        if n == current_node: 
            fill = '#FF4B4B'; color = '#922B21'; pen = '2'; font = 'white'
        elif n in predicted_path: 
            fill = '#FEF9E7'; color = '#F1C40F'; pen = '1'; font = 'black'
        dot.node(n, label=n, fillcolor=fill, color=color, penwidth=pen, fontcolor=font)
        
    for e in edges:
        color = '#D5D8DC'; pen = '1'
        if e["from"] in predicted_path and e["to"] in predicted_path:
             try:
                 if predicted_path.index(e["to"]) == predicted_path.index(e["from"]) + 1:
                    color = '#F1C40F'; pen = '2.5'
             except: pass
        dot.edge(e["from"], e["to"], color=color, penwidth=pen)
    return dot

# --- MAIN APP ---
st.sidebar.title("🛠️ SellMe Control")
mode = st.sidebar.radio("Mode", ["🤖 Sales Bot CRM", "🧪 Math Lab"])

if mode == "🤖 Sales Bot CRM":
    api_key = None
    try:
        if "GOOGLE_API_KEY" in st.secrets: api_key = st.secrets["GOOGLE_API_KEY"]
    except: pass

    if not api_key: api_key = st.sidebar.text_input("Google API Key", type="password")

    if st.sidebar.button("📊 Dashboard"): st.session_state.page = "dashboard"; st.rerun()
    if st.sidebar.button("📞 New Call"): st.session_state.page = "setup"; st.rerun()

    if not api_key:
        st.warning("🔑 Please enter API Key to start.")
        st.stop()

    configure_genai(api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    graph_data = load_graph_data()
    graph, nodes = graph_data[0], graph_data[3]
    node_to_id, id_to_node = graph_data[1], graph_data[2]

    # --- DASHBOARD ---
    if st.session_state.page == "dashboard":
        st.title("📊 CRM & Analytics Hub")
        if st.button("🧠 Train AI on History (RL)"):
            with st.spinner("Analyzing patterns... Updating weights..."): msg = train_brain()
            st.success(msg)
        
        data, stats = get_analytics()
        if data is not None and not data.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Calls", stats["total"])
            c2.metric("Success Rate", f"{stats['success_rate']}%")
            c3.metric("AI Learning Iterations", "v1.2")
            st.divider()
            
            st.subheader("🕵️ Call Inspector")
            options = data.apply(lambda x: f"{x['Date']} | {x['Name']} ({x['Outcome']})", axis=1).tolist()
            selected_option = st.selectbox("Select a call to review:", options)
            if selected_option:
                selected_row = data.iloc[options.index(selected_option)]
                with st.expander("📝 Full Transcript & Insights", expanded=True):
                    st.markdown(f"**Client:** {selected_row['Name']} ({selected_row['Type']})")
                    st.markdown(f"**Result:** {selected_row['Outcome']}")
                    st.text_area("Transcript", str(selected_row.get("Transcript", "No transcript available")), height=300)
                    if "AI Insights" in selected_row and selected_row["AI Insights"]:
                        st.info(f"💡 **AI Insight:** {selected_row['AI Insights']}")
                    else: st.warning("No insights generated for this call.")
        else: st.info("Database is empty. Make some calls!")

    # --- SETUP WITH PRODUCT INFO ---
    elif st.session_state.page == "setup":
        st.title("👤 Налаштування Дзвінка")
        with st.form("lead_form"):
            c1, c2 = st.columns(2)
            # Lead Info
            c1.markdown("### 👨‍💼 Lead Info")
            bot_name = c1.text_input("Ваше ім'я (Менеджера)", "Олексій")
            name = c1.text_input("Ім'я Клієнта", "Олександр")
            company = c1.text_input("Компанія", "SoftServe")
            type_ = c1.selectbox("Тип бізнесу", ["B2B", "B2C"])
            context = c1.selectbox("Контекст", ["Холодний дзвінок", "Теплий лід", "Повторний дзвінок"])

            # NEW: Product Info
            c2.markdown("### 📦 Product / Service Info")
            p_name = c2.text_input("Product Name", "AI Sales Engine")
            p_value = c2.text_input("Main Benefit (Value)", "Increases sales by 300%")
            p_price = c2.text_input("Price / Pricing Model", "$100/month")
            p_diff = c2.text_input("Competitive Edge (Why us?)", "Learns from every call")
            
            if c1.checkbox("🔍 Перевірити в базі"):
                try:
                    from leads_manager import connect_to_gsheet
                    sheet = connect_to_gsheet()
                    if sheet:
                        records = sheet.get_all_records()
                        found = [r for r in records if str(r['Name']).lower() == name.lower()]
                        if found:
                            last = found[-1]
                            st.info(f"📜 Contact Found: {last['Date']}")
                            context = "Повторний дзвінок" 
                        else: st.success("✨ New Client")
                except: st.error("Database unavailable.")
            
            submitted = st.form_submit_button("🚀 Start Call")
            if submitted:
                st.session_state.lead_info = {
                    "bot_name": bot_name, "name": name, 
                    "company": company, "type": type_, "context": context
                }
                # Store Product Info
                st.session_state.product_info = {
                    "product_name": p_name,
                    "product_value": p_value,
                    "product_price": p_price,
                    "competitor_diff": p_diff
                }
                st.session_state.messages = []
                st.session_state.current_node = "start"
                st.session_state.checklist = {k:False for k in st.session_state.checklist}
                st.session_state.page = "chat"
                st.session_state.visited_history = []
                st.rerun()

    # --- CHAT ---
    elif st.session_state.page == "chat":
        st.markdown(f"### Call with {st.session_state.lead_info['name']}")
        col_chat, col_tools = st.columns([1.5, 1])
        
        with col_tools:
            st.markdown("#### 🎯 Objectives")
            if "qualification" in st.session_state.current_node: st.session_state.checklist["Identify Customer"] = True
            if "pain" in st.session_state.current_node: st.session_state.checklist["Determine Objectives"] = True
            if "pitch" in st.session_state.current_node: st.session_state.checklist["Outline Advantages"] = True
            for goal, done in st.session_state.checklist.items(): st.write(f"{'✅' if done else '⬜'} {goal}")
            
            st.markdown("#### 🧠 Profile")
            current_archetype = st.session_state.get("current_archetype", "Analyzing...")
            cols = st.columns(4)
            def op(t): return "1.0" if current_archetype == t else "0.3"
            cols[0].markdown(f"<div style='opacity:{op('DRIVER')};text-align:center'>🔴<br>Boss</div>", unsafe_allow_html=True)
            cols[1].markdown(f"<div style='opacity:{op('ANALYST')};text-align:center'>🔵<br>Analyst</div>", unsafe_allow_html=True)
            cols[2].markdown(f"<div style='opacity:{op('EXPRESSIVE')};text-align:center'>🟡<br>Fan</div>", unsafe_allow_html=True)
            cols[3].markdown(f"<div style='opacity:{op('CONSERVATIVE')};text-align:center'>🟢<br>Safe</div>", unsafe_allow_html=True)
            if st.session_state.reasoning: st.caption(f"🤖 {st.session_state.reasoning}")
                
            st.markdown("#### 📊 Strategy")
            curr_id = node_to_id[st.session_state.current_node]
            target_id = node_to_id["close_standard"]
            path = get_predicted_path(graph, curr_id, target_id, id_to_node, node_to_id)
            st.graphviz_chart(draw_graph(graph_data, st.session_state.current_node, path), use_container_width=True)
            
            with st.expander("🧮 Bellman-Ford Logs"):
                visited_ids = [node_to_id[n] for n in st.session_state.get('visited_history', []) if n in node_to_id]
                client_type = st.session_state.lead_info.get('type', 'B2B')
                current_sentiment = st.session_state.get("current_sentiment", 0.0)
                raw_dist = bellman_ford_list(graph, curr_id, visited_nodes=visited_ids, client_type=client_type, sentiment_score=current_sentiment)
                
                debug_data = []
                target_path_set = set(path)
                for i, d in enumerate(raw_dist):
                    node_name = id_to_node[i]
                    status = "⬜"
                    if node_name == st.session_state.current_node: status = "📍 Start"
                    elif node_name in target_path_set: status = "✨ Path"
                    elif d == float('inf'): status = "🚫 Unreachable"
                    debug_data.append({"Node": node_name, "Cost": "∞" if d==float('inf') else round(d,2), "Status": status})
                
                df_log = pd.DataFrame(debug_data)
                df_log["sort"] = df_log["Cost"].apply(lambda x: 9999 if x=="∞" else float(x))
                st.dataframe(df_log.sort_values("sort").drop(columns=["sort"]), hide_index=True)

        with col_chat:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]): st.write(msg["content"])
                
            if not st.session_state.messages:
                with st.spinner("AI warming up..."):
                    # Pass Product Info
                    greeting = generate_greeting(model, nodes["start"], st.session_state.lead_info, st.session_state.product_info)
                st.session_state.messages.append({"role": "assistant", "content": greeting})
                st.rerun()

            if user_input := st.chat_input("Reply..."):
                st.session_state.messages.append({"role": "user", "content": user_input})
                current_text = nodes[st.session_state.current_node]
                analysis = analyze_full_context(model, user_input, st.session_state.current_node, st.session_state.messages)
                intent, archetype = analysis.get("intent", "STAY"), analysis.get("archetype", "UNKNOWN")
                st.session_state.current_archetype = archetype
                st.session_state.reasoning = analysis.get("reasoning", "")
                
                if "EXIT" in intent:
                    outcome = "Success" if "close" in st.session_state.current_node else "Fail"
                    save_lead_to_db(st.session_state.lead_info, st.session_state.messages, outcome)
                    st.success("Call Saved!")
                    st.session_state.page = "dashboard"; st.rerun()
                elif "STAY" in intent:
                    # Pass Product Info
                    resp = generate_response(model, current_text, user_input, "STAY", st.session_state.lead_info, archetype, st.session_state.product_info)
                else: # MOVE
                     if st.session_state.current_node not in st.session_state.visited_history:
                        st.session_state.visited_history.append(st.session_state.current_node)
                     curr_id = node_to_id[st.session_state.current_node]
                     best_next = None; min_w = float('inf')
                     for n, w in graph.adj_list[curr_id]:
                        if w < min_w: min_w = w; best_next = n
                     if best_next is not None:
                        st.session_state.current_node = id_to_node[best_next]
                        new_text = nodes[st.session_state.current_node]
                        # Pass Product Info
                        resp = generate_response(model, new_text, user_input, "MOVE", st.session_state.lead_info, archetype, st.session_state.product_info)
                     else:
                        resp = "Call finished."
                        save_lead_to_db(st.session_state.lead_info, st.session_state.messages, "End of Script")

                st.session_state.messages.append({"role": "assistant", "content": resp})
                st.rerun()

elif mode == "🧪 Math Lab":
    st.title("🧪 Computational Math Lab")
    st.markdown("### Section A: Graph Inspector")
    col1, col2 = st.columns(2)
    n_nodes = col1.slider("N (Vertices)", 5, 15, 10)
    density = col2.slider("Density", 0.1, 1.0, 0.5)
    
    if st.button("Generate Graph"):
         graph = experiments.generate_erdos_renyi(n_nodes, density)
         st.session_state.lab_graph = graph
    
    if 'lab_graph' in st.session_state:
        graph = st.session_state.lab_graph
        tab1, tab2, tab3 = st.tabs(["Visual Graph", "Adjacency Matrix", "Adjacency List"])
        
        with tab1:
            st.subheader("Graphviz Visualization")
            is_connected, unreachable = graph.check_connectivity(0)
            if is_connected: st.success("✅ Graph is Fully Connected (from Node 0)")
            else: st.error(f"⚠️ Warning: Unreachable nodes: {unreachable}")
            
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
            st.dataframe(df_heatmap.style.background_gradient(cmap="Blues", axis=None).format(formatter=lambda x: f"{x:.0f}" if pd.notnull(x) else "∞"))
            
        with tab3:
            st.subheader("Adjacency List")
            st.write(graph.adj_list)
            
    st.divider()
    st.markdown("### Section B: Scientific Experiments")
    st.markdown("Comparing Bellman-Ford implementations: **Adjacency List vs Adjacency Matrix**.")
    sizes_preset = list(range(20, 220, 20)) 
    densities_preset = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    if st.button("🚀 Run Scientific Benchmark"):
        with st.spinner("Running benchmarks..."):
            results = experiments.run_scientific_benchmark(sizes_preset, densities_preset, num_runs=20)
            df_results = pd.DataFrame(results)
            st.subheader("Raw Data")
            st.dataframe(df_results)
            st.divider()
            
            c_chart, c_filter = st.columns([3, 1])
            with c_filter:
                sel_density = st.selectbox("Density:", densities_preset, index=2)
                st.info(f"**Analysis:** List O(E) vs Matrix O(V^3).")
            with c_chart:
                filtered_df = df_results[df_results["Density"] == sel_density].sort_values("Vertices (N)")
                st.line_chart(filtered_df.set_index("Vertices (N)")[["Time_List", "Time_Matrix"]])
            st.success("Benchmarking complete!")
