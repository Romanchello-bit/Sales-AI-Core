import json
import os
import sys
import google.generativeai as genai
from graph_module import Graph
from algorithms import bellman_ford_list

# --- ФІКС КОДУВАННЯ ДЛЯ WINDOWS ---
sys.stdout.reconfigure(encoding='utf-8')

# --- ТВОЯ МОДЕЛЬ ---
MODEL_NAME = "gemini-2.5-flash"

def configure_genai():
    """Налаштування API ключа"""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("--- Налаштування API ---")
        api_key = input("Встав свій Google API Key (права кнопка миші -> Paste): ").strip()
    
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        print(f"Error configuring Gemini: {e}")
        return False

def load_data():
    """Завантажує скрипт і будує граф"""
    script_file = "sales_script_learned.json" if os.path.exists("sales_script_learned.json") else "sales_script.json"
    
    with open(script_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    nodes = data["nodes"]
    edges = data["edges"]
    
    node_to_id = {name: i for i, name in enumerate(nodes.keys())}
    id_to_node = {i: name for i, name in enumerate(nodes.keys())}
    
    graph = Graph(len(nodes), directed=True)
    
    for edge in edges:
        u, v = edge["from"], edge["to"]
        if u in node_to_id and v in node_to_id:
            graph.add_edge(node_to_id[u], node_to_id[v], edge["weight"])
            
    return graph, node_to_id, id_to_node, nodes

def analyze_intent(model, user_input, current_step_text):
    """
    Визначає наміри клієнта (MOVE або STAY).
    """
    prompt = f"""
    ROLE: Sales Logic Engine.
    CONTEXT:
    Bot said: "{current_step_text}"
    User said: "{user_input}"
    
    TASK: Determine if the user allows moving forward.
    RULES:
    - Agreement / Positive answer / Neutral info -> RETURN 'MOVE'
    - Objection / Question / Confusion / Anger -> RETURN 'STAY'
    - "Stop" / "Bye" -> RETURN 'EXIT'
    
    OUTPUT: Just one word: MOVE, STAY, or EXIT.
    """
    try:
        # Використовуємо саме твою модель
        response = model.generate_content(prompt)
        decision = response.text.strip().upper()
        
        if "EXIT" in decision: return "EXIT"
        if "MOVE" in decision: return "MOVE"
        return "STAY"
    except Exception as e:
        print(f"[API Error on Intent: {e}]")
        return "STAY"

def generate_smart_response(model, context, user_input, intent):
    """Генерує відповідь"""
    if intent == "STAY":
        prompt = f"""
        Role: Polite Sales Assistant.
        Context: You are at step "{context}". User objected: "{user_input}".
        Task: Handle the objection politely. Do NOT move to the next step.
        Language: Ukrainian.
        """
    else:
        prompt = f"""
        Role: Sales Assistant.
        Goal: Transition to: "{context}". User said: "{user_input}".
        Task: Create a natural bridge phrase.
        Language: Ukrainian.
        """
        
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[API Error on Generation: {e}]"

def main():
    if not configure_genai(): return
    
    print(f"\n[INFO] Connecting to model: {MODEL_NAME}...")
    try:
        model = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        print(f"[CRITICAL ERROR] Could not initialize {MODEL_NAME}: {e}")
        return

    try:
        graph, node_to_id, id_to_node, node_texts = load_data()
    except FileNotFoundError:
        print("Помилка: Не знайдено sales_script.json")
        return
    
    current_node = "start"
    
    print("\n" + "="*50)
    print(f"🚀 SellMe Smart Bot ({MODEL_NAME})")
    print("="*50 + "\n")
    
    print(f"Bot: {node_texts[current_node]}")
    
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input: continue
        
        # 1. АНАЛІЗ
        print(f"... ({MODEL_NAME} думає) ...")
        intent = analyze_intent(model, user_input, node_texts[current_node])
        
        if intent == "EXIT":
            print("Bot: До побачення!")
            break
            
        elif intent == "STAY":
            print(f"   [🛑 Логіка: Заперечення -> Стоїмо на '{current_node}']")
            response = generate_smart_response(model, node_texts[current_node], user_input, "STAY")
            print(f"Bot: {response}")
            
        elif intent == "MOVE":
            print(f"   [✅ Логіка: Згода -> Рухаємось далі]")
            
            curr_id = node_to_id[current_node]
            best_next = None
            min_w = float('inf')
            
            for neighbor, weight in graph.adj_list[curr_id]:
                if weight < min_w:
                    min_w = weight
                    best_next = neighbor
            
            if best_next is None:
                print("Bot: Дякую за розмову!")
                break
                
            current_node = id_to_node[best_next]
            response = generate_smart_response(model, node_texts[current_node], user_input, "MOVE")
            print(f"Bot: {response}")
            
            if current_node in ["close_deal", "exit_bad"]:
                break

if __name__ == "__main__":
    main()