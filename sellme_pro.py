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
        # Extract number from response
        sentiment_score = float(sentiment_text)
        # Clamp to [-1, 1] range
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        return sentiment_score
    except Exception as e:
        print(f"[WARNING] Sentiment analysis failed: {e}. Defaulting to neutral (0.0)")
        return 0.0


def update_weights(graph, str_to_int, original_edges, sentiment_score):
    """
    Dynamically update graph edge weights based on user sentiment.
    
    Args:
        graph: The Graph object
        str_to_int: Mapping from string node names to integer IDs
        original_edges: Original edge data from JSON
        sentiment_score: Float from -1 to +1
    """
    # Strategy mapping
    close_deal_id = str_to_int.get('close_deal')
    discount_offer_id = str_to_int.get('discount_offer')
    exit_bad_id = str_to_int.get('exit_bad')
    pitch_crm_id = str_to_int.get('pitch_crm')
    pitch_no_crm_id = str_to_int.get('pitch_no_crm')
    
    # Rebuild graph with adjusted weights
    for edge in original_edges:
        from_id = str_to_int[edge['from']]
        to_id = str_to_int[edge['to']]
        original_weight = edge['weight']
        adjusted_weight = original_weight
        
        # Negative sentiment (< -0.3): Customer is unhappy/frustrated
        if sentiment_score < -0.3:
            # INCREASE weights for aggressive moves (hard selling is bad now)
            if to_id == close_deal_id or to_id in [pitch_crm_id, pitch_no_crm_id]:
                adjusted_weight = original_weight * 2.0  # Make these paths less attractive
            
            # DECREASE weights for relationship-saving moves
            if to_id == discount_offer_id or to_id == exit_bad_id:
                adjusted_weight = original_weight * 0.5  # Make these paths more attractive
        
        # Positive sentiment (> 0.3): Customer is happy/interested
        elif sentiment_score > 0.3:
            # DECREASE weights for close_deal (strike while iron is hot!)
            if to_id == close_deal_id:
                adjusted_weight = original_weight * 0.3  # Make closing much more attractive
            
            # Also boost pitch effectiveness when customer is positive
            if to_id in [pitch_crm_id, pitch_no_crm_id]:
                adjusted_weight = original_weight * 0.7  # Make pitches more attractive
        
        # Update the graph (we need to rebuild adjacency structures)
        # Since Graph doesn't have update_edge, we'll handle this in the main loop
        graph.adj_matrix[from_id][to_id] = adjusted_weight
        
        # Update adjacency list
        for i, (neighbor, _) in enumerate(graph.adj_list[from_id]):
            if neighbor == to_id:
                graph.adj_list[from_id][i] = (neighbor, adjusted_weight)
                break


def main():
    # Configuration: Get API Key from user
    print("=" * 60)
    print("SellMe PRO - Dynamic Sentiment-Based Sales AI")
    print("=" * 60)
    api_key = input("\nEnter your Gemini API Key: ").strip()
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    print("\n[INFO] Gemini configured successfully!")
    print("[INFO] Loading sales script...\n")
    
    # Load sales_script.json
    with open('sales_script.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    nodes_data = data['nodes']
    edges_data = data['edges']
    
    # Create mappings: string IDs <-> integer IDs
    str_to_int = {}
    int_to_str = {}
    
    for idx, node_name in enumerate(nodes_data.keys()):
        str_to_int[node_name] = idx
        int_to_str[idx] = node_name
    
    # Build Graph object
    num_nodes = len(nodes_data)
    graph = Graph(num_nodes, directed=True)
    
    # Add edges with original weights
    for edge in edges_data:
        from_node = str_to_int[edge['from']]
        to_node = str_to_int[edge['to']]
        weight = edge['weight']
        graph.add_edge(from_node, to_node, weight)
    
    print("[INFO] Sales graph built successfully!")
    print(f"[INFO] Nodes: {num_nodes}, Edges: {len(edges_data)}")
    print("[INFO] Sentiment-based dynamic weighting enabled!\n")
    print("=" * 60)
    print("Starting Sales Conversation")
    print("=" * 60)
    print("(Type 'quit' to exit)\n")
    
    # Start conversation
    current_step = "start"
    conversation_count = 0
    max_steps = 20  # Safety limit
    
    while current_step not in ["close_deal", "exit_bad"] and conversation_count < max_steps:
        # Get current node ID
        current_id = str_to_int[current_step]
        
        # Get user input first
        print(f"\n[CURRENT STEP: {current_step}]")
        user_input = input("\nYou (Client): ").strip()
        
        if user_input.lower() == 'quit':
            print("\n[INFO] Exiting demo. Goodbye!")
            break
        
        # === SENTIMENT ANALYSIS ===
        print("\n[AI is analyzing sentiment...]")
        sentiment_score = get_sentiment(user_input, model)
        
        # Determine sentiment category
        if sentiment_score < -0.3:
            sentiment_label = "NEGATIVE"
        elif sentiment_score > 0.3:
            sentiment_label = "POSITIVE"
        else:
            sentiment_label = "NEUTRAL"
        
        print(f">>> Detected Sentiment: {sentiment_score:.2f} [{sentiment_label}]")
        
        # === DYNAMIC WEIGHT UPDATE ===
        if abs(sentiment_score) > 0.3:
            print(">>> Strategy Changed! Adjusting conversation path...")
            update_weights(graph, str_to_int, edges_data, sentiment_score)
        
        # Calculate best path with updated weights
        distances = bellman_ford_list(graph, current_id)
        
        if distances is None:
            print("[ERROR] Negative cycle detected in sales graph!")
            break
        
        # Get close_deal node ID
        close_deal_id = str_to_int['close_deal']
        
        # Find the best immediate next step
        adj_list = graph.get_list()
        neighbors = adj_list[current_id]
        
        if not neighbors:
            print(f"[ERROR] No path forward from '{current_step}'")
            break
        
        # Pick the neighbor with shortest total distance to close_deal
        best_next_id = None
        best_total_distance = float('inf')
        
        for neighbor_id, edge_weight in neighbors:
            # Run Bellman-Ford from this neighbor to find distance to close_deal
            neighbor_distances = bellman_ford_list(graph, neighbor_id)
            if neighbor_distances and neighbor_distances[close_deal_id] != float('inf'):
                total_distance = edge_weight + neighbor_distances[close_deal_id]
                if total_distance < best_total_distance:
                    best_total_distance = total_distance
                    best_next_id = neighbor_id
        
        if best_next_id is None:
            print(f"[ERROR] No path found from '{current_step}' to 'close_deal'")
            break
        
        # Get next step name and script text
        next_step_name = int_to_str[best_next_id]
        script_text = nodes_data[next_step_name]
        
        print(f"[NEXT TARGET: {next_step_name}]")
        
        # Create prompt for Gemini
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
        
        # Get Gemini's response
        print("\n[AI is generating response...]")
        try:
            response = model.generate_content(prompt)
            ai_response = response.text.strip()
            
            print(f"\nSellMe AI: {ai_response}")
            
        except Exception as e:
            print(f"\n[ERROR] Gemini API error: {e}")
            print(f"[FALLBACK] Using script: {script_text}")
        
        # Move to next step
        current_step = next_step_name
        conversation_count += 1
    
    # End of conversation
    print("\n" + "=" * 60)
    if current_step == "close_deal":
        print("[SUCCESS] Deal closed!")
        print(f"Final message: {nodes_data[current_step]}")
    elif current_step == "exit_bad":
        print("[EXIT] Client not interested.")
        print(f"Final message: {nodes_data[current_step]}")
    else:
        print(f"[INFO] Conversation ended at step: {current_step}")
    print("=" * 60)


if __name__ == "__main__":
    main()
