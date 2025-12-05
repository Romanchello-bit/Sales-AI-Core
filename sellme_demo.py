import json
import os
import google.generativeai as genai
from graph_module import Graph
from algorithms import bellman_ford_list


def main():
    # Configuration: Get API Key from user
    print("=" * 60)
    print("SellMe AI Sales Demo - Powered by Gemini")
    print("=" * 60)
    api_key = input("\nEnter your Gemini API Key: ").strip()
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
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
    
    # Add edges with weights
    for edge in edges_data:
        from_node = str_to_int[edge['from']]
        to_node = str_to_int[edge['to']]
        weight = edge['weight']
        graph.add_edge(from_node, to_node, weight)
    
    print("[INFO] Sales graph built successfully!")
    print(f"[INFO] Nodes: {num_nodes}, Edges: {len(edges_data)}\n")
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
        
        # Calculate best path to close_deal using Bellman-Ford
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
        
        # Get user input (simulating client)
        print(f"\n[CURRENT STEP: {current_step}]")
        print(f"[NEXT TARGET: {next_step_name}]")
        user_input = input("\nYou (Client): ").strip()
        
        if user_input.lower() == 'quit':
            print("\n[INFO] Exiting demo. Goodbye!")
            break
        
        # Create prompt for Gemini
        prompt = f"""You are a professional sales representative for SellMe, an AI sales assistant platform.

Your goal is to move the conversation toward this step: '{next_step_name}'.
The sales script for this step says: '{script_text}'.
The client just said: '{user_input}'.

Generate a natural, conversational response in Ukrainian that:
1. Acknowledges what the client said
2. Smoothly guides toward the script message
3. Sounds human and friendly, not robotic
4. Keep it brief (1-2 sentences max)

Response:"""
        
        # Get Gemini's response
        print("\n[AI is thinking...]")
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
        print("[SUCCESS] Deal closed! 🎉")
        print(f"Final message: {nodes_data[current_step]}")
    elif current_step == "exit_bad":
        print("[EXIT] Client not interested.")
        print(f"Final message: {nodes_data[current_step]}")
    else:
        print(f"[INFO] Conversation ended at step: {current_step}")
    print("=" * 60)


if __name__ == "__main__":
    main()
