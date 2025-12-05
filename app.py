import streamlit as st
import graphviz
import json
import os
import pandas as pd
from datetime import datetime
import google.generativeai as genai
from graph_module import Graph
from algorithms import bellman_ford_list

# --- CONFIG ---
st.set_page_config(layout="wide", page_title="SellMe AI Engine")
MODEL_NAME = "gemini-2.5-flash"
LEADS_FILE = "leads_database.csv"

# --- SESSION STATE INIT ---
if "page" not in st.session_state: st.session_state.page = "dashboard"
if "messages" not in st.session_state: st.session_state.messages = []
if "current_node" not in st.session_state: st.session_state.current_node = "start"
if "lead_info" not in st.session_state: st.session_state.lead_info = {}
if "visited_history" not in st.session_state: st.session_state.visited_history = []  # Track visited nodes
if "current_archetype" not in st.session_state: st.session_state.current_archetype = "UNKNOWN"
if "reasoning" not in st.session_state: st.session_state.reasoning = ""
if "current_sentiment" not in st.session_state: st.session_state.current_sentiment = 0.0
# Checklist status based on your screenshot
if "checklist" not in st.session_state:
    st.session_state.checklist = {
        "Identify Customer": False,
        "Determine Objectives": False,
        "Outline Advantages": False,
        "Keep it Brief": True, # Always try to be brief
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
    # Ask AI to extract structured data from chat
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
        # Simple parsing (in production use structured output)
        ai_data = response.text
    except:
        ai_data = "AI Extraction Failed"

    new_row = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Name": lead_info.get("name"),
        "Company": lead_info.get("company"),
        "Type": lead_info.get("type"),
        "Context": lead_info.get("context"),
        "Pain Point": "AI Analysis Pending", # Placeholder for simplicity
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
    # Get visited node IDs and client type for smart pathfinding
    visited_ids = [node_to_id[n] for n in st.session_state.get('visited_history', []) if n in node_to_id]
    client_type = st.session_state.lead_info.get('type', 'B2B')
    
    # Use enhanced Bellman-Ford with penalties
    dist = bellman_ford_list(graph, start_id, visited_nodes=visited_ids, client_type=client_type)
    if dist[target_id] == float('inf'): return []
    path = [target_id]
    curr = target_id
    while curr != start_id:
        found = False
        for u in range(graph.num_vertices):
            for v, w in graph.adj_list[u]:
                if v == curr and dist[v] == dist[u] + w:
                    path.append(u); curr = u; found = True; break
            if found: break
        if not found: break
    return [id_to_node[i] for i in reversed(path)]

def analyze_full_context(model, user_input, current_node, chat_history):
    """
    Аналізує Інтент, Емоції та ПСИХОТИП клієнта.
    """
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-4:]]) # Беремо останні 4 фрази для контексту
    
    prompt = f"""
    ROLE: Behavioral Psychologist & Sales Expert.
    
    CONTEXT:
    Current Step: "{current_node}"
    Recent Chat History:
    {history_text}
    User just said: "{user_input}"
    
    TASK 1: Detect User Archetype (Pattern). Choose ONE:
    - DRIVER (Direct, impatient, results-oriented)
    - ANALYST (Detail-oriented, asks 'how', skeptical)
    - EXPRESSIVE (Emotional, enthusiastic, visionary)
    - CONSERVATIVE (Risk-averse, slow, likes stability)
    
    TASK 2: Analyze Intent (MOVE, STAY, EXIT).
    
    OUTPUT JSON format:
    {{
        "archetype": "DRIVER" | "ANALYST" | "EXPRESSIVE" | "CONSERVATIVE",
        "intent": "MOVE" | "STAY" | "EXIT",
        "reasoning": "Why you chose this archetype (1 short sentence)"
    }}
    """
    try:
        response = model.generate_content(prompt)
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except:
        return {"archetype": "UNKNOWN", "intent": "STAY", "reasoning": "Error"}

def generate_response(model, context, user_input, intent, lead_info, archetype):
    # Визначаємо стиль спілкування залежно від патерну
    style_instruction = ""
    
    if archetype == "DRIVER":
        style_instruction = "STYLE: Ultra-short, confident. Focus on ROI and speed. No fluff. Be direct."
    elif archetype == "ANALYST":
        style_instruction = "STYLE: Logical, detailed. Use facts, numbers, and technical terms. Prove your point."
    elif archetype == "EXPRESSIVE":
        style_instruction = "STYLE: Energetic, inspiring. Use metaphors, exclamation marks. Focus on the 'Future Success'."
    elif archetype == "CONSERVATIVE":
        style_instruction = "STYLE: Calm, supportive, safe. Emphasize low risk, support, and ease of use. Don't push."
    else:
        style_instruction = "STYLE: Professional and polite."

    if intent == "STAY":
        prompt = f"""
        ROLE: Chameleon Sales Rep.
        ARCHETYPE DETECTED: {archetype} -> {style_instruction}
        
        SITUATION: Step "{context}". Client Objected: "{user_input}".
        TASK: Handle objection strictly matching the detected STYLE.
        CONSTRAINT: Speak naturally in Ukrainian. Output ONLY the response.
        """
    else:
        prompt = f"""
        ROLE: Chameleon Sales Rep.
        ARCHETYPE DETECTED: {archetype} -> {style_instruction}
        
        GOAL: Transition to "{context}". User said: "{user_input}".
        TASK: Bridge to the next step using the detected STYLE.
        CONSTRAINT: Speak naturally in Ukrainian. Output ONLY the response.
        """
        
    try:
        return model.generate_content(prompt).text.strip()
    except: return "..."

def generate_greeting(model, start_node_text, lead_info):
    """Генерує ПЕРШЕ повідомлення"""
    
    # Використовуємо .get(), щоб уникнути помилок, якщо ключа немає
    bot_name = lead_info.get('bot_name', 'Олексій')
    client_name = lead_info.get('name', 'Клієнт')  # <--- ТУТ ТЕЖ МАЄ БУТИ 'name'
    company = lead_info.get('company', 'Компанія')
    
    prompt = f"""
    ROLE: Professional Sales Rep named {bot_name}.
    CLIENT: {client_name} from {company}.
    TYPE: {lead_info.get('type')} ({lead_info.get('context')}).
    
    GOAL: Start conversation based on instruction: "{start_node_text}".
    
    INSTRUCTIONS:
    - Always state your name ({bot_name}) and company (SellMe AI).
    - If B2B: Be formal.
    - If B2C: Be friendly.
    - Language: Ukrainian.
    
    OUTPUT: Just the spoken greeting.
    """
    try:
        return model.generate_content(prompt).text.strip()
    except:
        return f"Доброго дня, це {bot_name} з SellMe. Маєте хвилинку?"


# --- UI COMPONENTS ---
def draw_graph(graph_data, current_node, predicted_path):
    nodes = graph_data[3]
    edges = graph_data[4]
    
    dot = graphviz.Digraph()
    
    # --- НАЛАШТУВАННЯ ГЕОМЕТРІЇ (Compact Mode) ---
    dot.attr(
        rankdir='TB',        # Зверху вниз
        splines='ortho',     # Ламані лінії (прямі кути)
        nodesep='0.3',       # Мінімальний відступ збоку
        ranksep='0.4',       # Мінімальний відступ знизу (робить граф коротшим)
        bgcolor='transparent' # Прозорий фон, щоб зливався з темою
    )
    
    # --- СТИЛЬ БЛОКІВ (Wide & Slim) ---
    # shape='note' виглядає як документ, або 'box' для суворості
    # fixedsize='false' дозволяє блоку розтягуватись під текст, але ми задаємо мінімальну ширину
    dot.attr('node', 
             shape='box', 
             style='rounded,filled', 
             fontname='Arial', 
             fontsize='11', 
             width='2.5',      # Робимо їх широкими
             height='0.5',     # Робимо їх низькими
             margin='0.1'      # Менше полів всередині блоку
    )
    
    # --- СТИЛЬ ЛІНІЙ ---
    dot.attr('edge', fontname='Arial', fontsize='9', arrowsize='0.6')

    for n in nodes:
        # Базовий стиль (Світло-сірий, непомітний)
        fill = '#F7F9F9'; color = '#BDC3C7'; pen = '1'; font = '#424949'
        
        # Поточний крок (Червоний акцент)
        if n == current_node: 
            fill = '#FF4B4B'; color = '#922B21'; pen = '2'; font = 'white'
            
        # Золотий шлях (Жовтий підсвіт)
        elif n in predicted_path: 
            fill = '#FEF9E7'; color = '#F1C40F'; pen = '1'; font = 'black'
            
        # Малюємо вузол
        dot.node(n, label=n, fillcolor=fill, color=color, penwidth=pen, fontcolor=font)
        
    for e in edges:
        color = '#D5D8DC'; pen = '1' # Дуже світлі лінії за замовчуванням
        
        # Підсвітка шляху
        if e["from"] in predicted_path and e["to"] in predicted_path:
             try:
                 # Перевіряємо послідовність
                 if predicted_path.index(e["to"]) == predicted_path.index(e["from"]) + 1:
                    color = '#F1C40F'; pen = '2.5' # Жирна золота лінія
             except: pass
             
        dot.edge(e["from"], e["to"], color=color, penwidth=pen)
        
    return dot

# --- MAIN APP ---
st.sidebar.title("🛠️ SellMe Control")

# --- API KEY SETUP (Robust) ---
api_key = None
try:
    # Try to get key from secrets (Cloud)
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
except:
    # If secrets file is missing (Local run), just ignore and pass
    pass

# Fallback to manual input if no key found yet
if not api_key:
    api_key = st.sidebar.text_input("Google API Key", type="password")

if st.sidebar.button("📊 Dashboard"): st.session_state.page = "dashboard"; st.rerun()
if st.sidebar.button("📞 New Call"): st.session_state.page = "setup"; st.rerun()

if not api_key:
    st.warning("🔑 Please enter API Key to start.")
    st.stop()

configure_genai(api_key)
model = genai.GenerativeModel(MODEL_NAME)
graph_data = load_graph_data()
graph, node_to_id, id_to_node, nodes, edges = graph_data

# --- PAGE: DASHBOARD ---
if st.session_state.page == "dashboard":
    st.title("📊 CRM Analytics")
    init_db()
    df = pd.read_csv(LEADS_FILE)
    if not df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Calls", len(df))
        c2.metric("B2B Leads", len(df[df['Type']=='B2B']))
        c3.metric("Success", len(df[df['Outcome']=='Success']))
        st.dataframe(df)
    else: st.info("Database empty.")

# --- PAGE: SETUP ---
elif st.session_state.page == "setup":
    st.title("👤 Налаштування Дзвінка")
    
    with st.form("lead_form"):
        st.markdown("### 👨‍💼 Хто дзвонить?")
        bot_name = st.text_input("Ваше ім'я (Менеджера)", "Олексій")
        
        st.markdown("### 📞 Кому дзвонимо?")
        c1, c2 = st.columns(2)
        name = c1.text_input("Ім'я Клієнта", "Олександр")
        company = c2.text_input("Компанія (для B2B)", "SoftServe")
        
        type_ = c1.selectbox("Тип бізнесу", ["B2B", "B2C"])
        context = c2.selectbox("Контекст", ["Холодний дзвінок", "Теплий лід (заявка)", "Повторний дзвінок"])
        
        submitted = st.form_submit_button("🚀 Почати розмову")
        
        if submitted:
            # Зберігаємо все, включаючи ім'я бота
            st.session_state.lead_info = {
                "bot_name": bot_name,
                "name": name,         # <--- ВИПРАВИЛИ НА "name"
                "company": company, 
                "type": type_, 
                "context": context
            }
            st.session_state.messages = []
            st.session_state.current_node = "start"
            st.session_state.checklist = {k:False for k in st.session_state.checklist}
            st.session_state.page = "chat"
            st.session_state.visited_history = []
            st.rerun()

# --- PAGE: CHAT ---
elif st.session_state.page == "chat":
    st.markdown(f"### Call with {st.session_state.lead_info['name']}")
    
    col_chat, col_tools = st.columns([1.5, 1])
    
    with col_tools:
        st.markdown("#### 🎯 Call Objectives")
        # Logic to auto-update checklist based on node
        if "qualification" in st.session_state.current_node: st.session_state.checklist["Identify Customer"] = True
        if "pain" in st.session_state.current_node or "shame" in st.session_state.current_node: st.session_state.checklist["Determine Objectives"] = True
        if "pitch" in st.session_state.current_node: st.session_state.checklist["Outline Advantages"] = True
        
        for goal, done in st.session_state.checklist.items():
            icon = "✅" if done else "⬜"
            st.write(f"{icon} {goal}")
        
        # Display Client Profile (Real-time)
        st.markdown("#### 🧠 Client Profile (Real-time)")
        
        # Get current archetype from session
        current_archetype = st.session_state.get("current_archetype", "Analyzing...")
        
        # Visual cards for archetypes
        cols = st.columns(4)
        
        # Styles for highlighting
        def get_opacity(target): return "1.0" if current_archetype == target else "0.3"
        
        cols[0].markdown(f"<div style='opacity:{get_opacity('DRIVER')}; font-size:20px; text-align:center'>🔴<br>Boss</div>", unsafe_allow_html=True)
        cols[1].markdown(f"<div style='opacity:{get_opacity('ANALYST')}; font-size:20px; text-align:center'>🔵<br>Analyst</div>", unsafe_allow_html=True)
        cols[2].markdown(f"<div style='opacity:{get_opacity('EXPRESSIVE')}; font-size:20px; text-align:center'>🟡<br>Fan</div>", unsafe_allow_html=True)
        cols[3].markdown(f"<div style='opacity:{get_opacity('CONSERVATIVE')}; font-size:20px; text-align:center'>🟢<br>Safe</div>", unsafe_allow_html=True)
        
        if st.session_state.reasoning:
            st.caption(f"🤖 AI Insight: {st.session_state.reasoning}")
            
        st.markdown("#### 📊 AI Strategy")
        curr_id = node_to_id[st.session_state.current_node]
        target_id = node_to_id["close_deal"]  # Fixed: using close_deal from sales_script.json
        path = get_predicted_path(graph, curr_id, target_id, id_to_node, node_to_id)
        st.graphviz_chart(
            draw_graph(graph_data, st.session_state.current_node, path),
            use_container_width=True  # Розтягує граф на всю ширину колонки
        )
        
        # --- BELLMAN-FORD ALGORITHM LOGS ---
        st.markdown("---")
        with st.expander("🧮 Алгоритм Беллмана-Форда (Live Logs)", expanded=False):
            st.markdown("""
            **Математика прийняття рішень:**
            Алгоритм шукає шлях $P$, де сума ваг $W$ є мінімальною:
            $$ D[v] = \\min(D[v], D[u] + W(u, v)) $$
            """)
            
            # 1. Calculate real data
            visited_ids = [node_to_id[n] for n in st.session_state.get('visited_history', []) if n in node_to_id]
            client_type = st.session_state.lead_info.get('type', 'B2B')
            current_sentiment = st.session_state.get("current_sentiment", 0.0)
            
            # Call algorithm to get distance array
            raw_dist = bellman_ford_list(
                graph, 
                curr_id, 
                visited_nodes=visited_ids, 
                client_type=client_type, 
                sentiment_score=current_sentiment
            )
            
            # 2. Build beautiful table for humans
            debug_data = []
            target_path_set = set(path)  # Path we already found for graph
            
            for i, d in enumerate(raw_dist):
                node_name = id_to_node[i]
                
                # Format infinity
                cost_display = "∞" if d == float('inf') else round(d, 2)
                
                # Node status
                status = "⬜"
                if node_name == st.session_state.current_node: status = "📍 Start"
                elif node_name in target_path_set: status = "✨ Path"
                elif d == float('inf'): status = "🚫 Unreachable"
                
                debug_data.append({
                    "Node": node_name,
                    "Cost (Weight)": cost_display,
                    "Status": status
                })
            
            # Convert to DataFrame
            df_log = pd.DataFrame(debug_data)
            
            # Sort: path first, then cheap, then expensive
            df_log["sort_key"] = df_log["Cost (Weight)"].apply(lambda x: 9999 if x == "∞" else float(x))
            df_log = df_log.sort_values(by="sort_key").drop(columns=["sort_key"])
            
            # Display
            st.dataframe(
                df_log, 
                use_container_width=True,
                hide_index=True
            )
            
            # 3. Explanation "Why?"
            st.info(f"""
            **Фактори впливу:**
            - 🎭 **Емоція:** {current_sentiment} (впливає на вартість агресивних кроків)
            - 🏢 **Тип:** {client_type} (змінює пріоритет швидкості)
            - 🔄 **Повтори:** Вузли, де ми вже були, мають штраф x50.
            """)

    with col_chat:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.write(msg["content"])
            
        # --- ГЕНЕРАЦІЯ ПЕРШОГО ПОВІДОМЛЕННЯ ---
        if not st.session_state.messages:
            with st.spinner("AI готується до дзвінка..."):
                start_instruction = nodes["start"]
                # Викликаємо AI для генерації живого привітання
                # lead_info keys might differ if coming from very old session, but setup ensures keys exist.
                # Just in case, defaults from setup form are used.
                greeting = generate_greeting(model, start_instruction, st.session_state.lead_info)
                
            st.session_state.messages.append({"role": "assistant", "content": greeting})
            st.rerun()

        if user_input := st.chat_input("Reply..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Logic - Analyze with full context including archetype detection
            current_text = nodes[st.session_state.current_node]
            analysis = analyze_full_context(model, user_input, st.session_state.current_node, st.session_state.messages)
            intent = analysis.get("intent", "STAY")
            archetype = analysis.get("archetype", "UNKNOWN")
            reasoning = analysis.get("reasoning", "")
            
            # Store archetype and reasoning for display
            st.session_state.current_archetype = archetype
            st.session_state.reasoning = reasoning
            
            if "EXIT" in intent:
                outcome = "Success" if "close" in st.session_state.current_node else "Fail"
                save_lead_to_db(st.session_state.lead_info, st.session_state.messages, outcome)
                st.success("Call Saved!")
                st.session_state.page = "dashboard"; st.rerun()
            
            elif "STAY" in intent:
                resp = generate_response(model, current_text, user_input, "STAY", st.session_state.lead_info, archetype)
            
            else: # MOVE
                # Track current node in visited history before moving
                if st.session_state.current_node not in st.session_state.visited_history:
                    st.session_state.visited_history.append(st.session_state.current_node)
                
                curr_id = node_to_id[st.session_state.current_node]
                best_next = None; min_w = float('inf')
                for n, w in graph.adj_list[curr_id]:
                    if w < min_w: min_w = w; best_next = n
                
                if best_next is not None:
                    st.session_state.current_node = id_to_node[best_next]
                    new_text = nodes[st.session_state.current_node]
                    resp = generate_response(model, new_text, user_input, "MOVE", st.session_state.lead_info, archetype)
                else:
                    resp = "Call finished."
                    save_lead_to_db(st.session_state.lead_info, st.session_state.messages, "End of Script")

            st.session_state.messages.append({"role": "assistant", "content": resp})
            st.rerun()
