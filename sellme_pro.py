import json
import os
import google.generativeai as genai
from graph_module import Graph
from algorithms import bellman_ford_list


def get_sentiment(user_text, model):
    """
    Analyze user sentiment using Gemini.
    
    Args:
        user_text: The user's message
        model: Gemini model instance
        
    Returns:
        Float score from -1 (Angry) to +1 (Happy)
    """
    prompt = f"""Analyze the sentiment of this message and return ONLY a number between -1 and +1.
-1 = Very angry, hostile, frustrated
-0.5 = Slightly negative, uncertain
0 = Neutral
+0.5 = Slightly positive, interested
+1 = Very happy, enthusiastic, eager

Message: "{user_text}"

Return only the number, nothing else:"""
    
    try:
        response = model.generate_content(prompt)
        sentiment_text = response.text.strip()
        sentiment_score = float(sentiment_text)
        return max(-1.0, min(1.0, sentiment_score))
    except Exception as e:
        print(f"[WARNING] Sentiment analysis failed: {e}. Defaulting to neutral (0.0)")
        return 0.0


def update_weights(graph, str_to_int, original_edges, sentiment_score):
    """
    Dynamically update graph edge weights based on user sentiment.
    """
    close_deal_id = str_to_int.get('close_deal')
    discount_offer_id = str_to_int.get('discount_offer')
    exit_bad_id = str_to_int.get('exit_bad')
    pitch_crm_id = str_to_int.get('pitch_crm')
    pitch_no_crm_id = str_to_int.get('pitch_no_crm')
    
    # Reset graph to original weights before applying sentiment
    graph.adj_list = [[] for _ in range(graph.num_vertices)]
    for edge in original_edges:
        graph.add_edge(str_to_int[edge['from']], str_to_int[edge['to']], edge['weight'])

    for from_id in range(graph.num_vertices):
        for i, (to_id, original_weight) in enumerate(graph.adj_list[from_id]):
            adjusted_weight = original_weight
            if sentiment_score < -0.3:
                if to_id in [close_deal_id, pitch_crm_id, pitch_no_crm_id]:
                    adjusted_weight *= 2.0
                elif to_id in [discount_offer_id, exit_bad_id]:
                    adjusted_weight *= 0.5
            elif sentiment_score > 0.3:
                if to_id == close_deal_id:
                    adjusted_weight *= 0.3
                elif to_id in [pitch_crm_id, pitch_no_crm_id]:
                    adjusted_weight *= 0.7
            
            graph.adj_list[from_id][i] = (to_id, adjusted_weight)


def main():
    print("=" * 60)
    print("SellMe PRO - Dynamic Sentiment-Based Sales AI")
    print("=" * 60)
    api_key = input("\nEnter your Gemini API Key: ").strip()
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    print("\n[INFO] Gemini configured successfully!")
    print("[INFO] Loading sales script...\n")
    
    with open('sales_script.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    nodes_data = data['nodes']
    edges_data = data['edges']
    
    str_to_int = {name: i for i, name in enumerate(nodes_data.keys())}
    int_to_str = {i: name for i, name in enumerate(nodes_data.keys())}
    
    num_nodes = len(nodes_data)
    graph = Graph(num_nodes, directed=True)
    for edge in edges_data:
        graph.add_edge(str_to_int[edge['from']], str_to_int[edge['to']], edge['weight'])
    
    print("[INFO] Sales graph built successfully!")
    print(f"[INFO] Nodes: {num_nodes}, Edges: {len(edges_data)}")
    print("[INFO] Sentiment-based dynamic weighting enabled!\n")
    print("=" * 60, "\nStarting Sales Conversation\n(Type 'quit' to exit)\n", "=" * 60)
    
    current_step = "start"
    conversation_count = 0
    max_steps = 20
    
    while current_step not in ["close_deal", "exit_bad"] and conversation_count < max_steps:
        current_id = str_to_int[current_step]
        
        print(f"\n[CURRENT STEP: {current_step}]")
        user_input = input("\nYou (Client): ").strip()
        
        if user_input.lower() == 'quit':
            print("\n[INFO] Exiting demo. Goodbye!")
            break
        
        print("\n[AI is analyzing sentiment...]")
        sentiment_score = get_sentiment(user_input, model)
        sentiment_label = "NEUTRAL"
        if sentiment_score < -0.3: sentiment_label = "NEGATIVE"
        elif sentiment_score > 0.3: sentiment_label = "POSITIVE"
        print(f">>> Detected Sentiment: {sentiment_score:.2f} [{sentiment_label}]")
        
        if abs(sentiment_score) > 0.3:
            print(">>> Strategy Changed! Adjusting conversation path...")
            update_weights(graph, str_to_int, edges_data, sentiment_score)
        
        # --- OPTIMIZED PATHFINDING ---
        # Run Bellman-Ford from every node to the destination (close_deal)
        # This is inefficient but required if weights change dynamically.
        # A better approach for dynamic graphs is D* Lite, but Bellman-Ford is what we have.
        # We calculate all-pairs shortest paths to the destination.
        close_deal_id = str_to_int['close_deal']
        dist_to_target = {i: float('inf') for i in range(num_nodes)}
        
        # This is still not optimal, but better than calling BF in a loop
        # For a truly optimal solution, one would reverse the graph edges and run BF once from the target.
        for i in range(num_nodes):
            distances = bellman_ford_list(graph, i)
            if distances:
                dist_to_target[i] = distances[close_deal_id]

        best_next_id = None
        min_total_dist = float('inf')
        
        for neighbor_id, weight in graph.adj_list[current_id]:
            total_dist = weight + dist_to_target.get(neighbor_id, float('inf'))
            if total_dist < min_total_dist:
                min_total_dist = total_dist
                best_next_id = neighbor_id

        if best_next_id is None:
            print(f"[ERROR] No path found from '{current_step}' to 'close_deal'")
            break
        
        next_step_name = int_to_str[best_next_id]
        script_text = nodes_data[next_step_name]
        
        print(f"[NEXT TARGET: {next_step_name}]")
        
        prompt = f"""You are a professional sales representative for SellMe, an AI sales assistant platform.
Your goal is to move the conversation toward this step: '{next_step_name}'.
The sales script for this step says: '{script_text}'.
The client just said: '{user_input}'.
Client sentiment: {sentiment_score:.2f} ({sentiment_label})
Generate a natural, conversational response in Ukrainian that:
1. Acknowledges what the client said and their emotional state
2. Smoothly guides toward the script message
3. Adjusts tone based on sentiment (softer if negative, enthusiastic if positive)
4. Sounds human and friendly, not robotic
5. Keep it brief (1-2 sentences max)
Response:"""
        
        print("\n[AI is generating response...]")
        try:
            response = model.generate_content(prompt)
            print(f"\nSellMe AI: {response.text.strip()}")
        except Exception as e:
            print(f"\n[ERROR] Gemini API error: {e}\n[FALLBACK] Using script: {script_text}")
        
        current_step = next_step_name
        conversation_count += 1
    
    print("\n" + "=" * 60)
    if current_step == "close_deal":
        print(f"[SUCCESS] Deal closed! Final message: {nodes_data[current_step]}")
    elif current_step == "exit_bad":
        print(f"[EXIT] Client not interested. Final message: {nodes_data[current_step]}")
    else:
        print(f"[INFO] Conversation ended at step: {current_step}")
    print("=" * 60)

if __name__ == "__main__":
    main()
