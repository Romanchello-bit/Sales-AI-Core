import google.generativeai as genai
import json
import random
import time
from graph_module import Graph
from algorithms import bellman_ford_list
import database

# --- CONFIGURATION ---
MODEL_NAME = "gemini-2.5-flash"
# API_KEY is now configured globally from app.py

# --- CORE FUNCTIONS ---

def generate_initial_population(count=5):
    """Generates a diverse starting population of sales scenarios."""
    print(f"Generating {count} initial scenarios...")
    model = genai.GenerativeModel(MODEL_NAME)
    
    prompt = f"""
    You are a world-class sales strategy expert. Create {count} diverse sales script scenarios in JSON graph format.
    Each JSON object must have "nodes" and "edges".
    "nodes" is a dictionary of step_name: "prompt_for_sales_agent".
    "edges" is a list of objects with "from", "to", and "weight".

    Ensure the following nodes always exist: "start", "close_standard", "exit_bad".

    Create diverse strategies:
    1. **Standard B2B:** A classic, balanced approach.
    2. **Aggressive Closer:** A script that tries to close the deal very quickly.
    3. **Relationship Builder:** A script focused on empathy and asking questions.
    4. **Data-Driven Analyst:** A script that heavily qualifies the lead with many questions.
    5. **Short & Sweet:** An extremely concise script for busy clients.

    Return a JSON array where each element is a complete graph object.
    """
    
    try:
        response = model.generate_content(prompt)
        scenarios = json.loads(response.text.replace("```json", "").replace("```", "").strip())
        
        for scenario_json in scenarios:
            database.add_scenario(scenario_json)
        print(f"Successfully generated and saved {len(scenarios)} scenarios.")
    except Exception as e:
        print(f"Error generating initial population: {e}")

def generate_customer_persona():
    """Generates a random customer persona for simulation."""
    archetypes = ["DRIVER", "ANALYST", "EXPRESSIVE", "CONSERVATIVE"]
    pain_points = ["current software is too slow", "paying too much for a similar service", "lacks key features", "bad customer support"]
    budgets = ["low", "medium", "high"]
    
    return {
        "archetype": random.choice(archetypes),
        "pain_point": random.choice(pain_points),
        "budget": random.choice(budgets),
        "interest": random.uniform(0.1, 0.9) # Initial interest level
    }

def analyze_transcript(transcript_text):
    """Uses AI to find impactful phrases in a conversation."""
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = f"""
    Analyze this sales conversation transcript. Identify specific, short phrases (3-10 words) from the 'assistant' (salesperson) that caused a clear positive or negative reaction from the 'user' (client).

    - A **positive** impact means the user became more agreeable, interested, or moved towards a 'yes'.
    - A **negative** impact means the user became resistant, annoyed, or started objecting.

    Transcript:
    {transcript_text}

    Return your analysis as a JSON object with two keys: "good_phrases" and "bad_phrases".
    The value for each key should be a list of strings.
    Example: {{ "good_phrases": ["that's a great question"], "bad_phrases": ["you need to buy this now"] }}
    """
    try:
        response = model.generate_content(prompt)
        analysis = json.loads(response.text.replace("```json", "").replace("```", "").strip())
        return analysis
    except Exception:
        return {"good_phrases": [], "bad_phrases": []}

def run_single_simulation(scenario_id):
    """Runs one full simulation from start to finish."""
    
    # 1. Load Scenario and Generate Customer
    scenario_json = database.get_scenario(scenario_id)
    if not scenario_json:
        print(f"Scenario {scenario_id} not found.")
        return

    customer = generate_customer_persona()
    model = genai.GenerativeModel(MODEL_NAME)

    # 2. Build Graph from Scenario
    nodes = scenario_json["nodes"]
    edges = scenario_json["edges"]
    node_to_id = {name: i for i, name in enumerate(nodes.keys())}
    id_to_node = {i: name for i, name in enumerate(nodes.keys())}
    graph = Graph(len(nodes), directed=True)
    for edge in edges:
        if edge["from"] in node_to_id and edge["to"] in node_to_id:
            graph.add_edge(node_to_id[edge["from"]], node_to_id[edge["to"]], edge["weight"])

    # 3. Simulate Conversation
    current_node = "start"
    transcript = []
    
    # Initial Greeting
    sales_response = nodes[current_node]
    transcript.append({"role": "assistant", "content": sales_response})

    for _ in range(15): # Max 15 turns
        # Customer response
        customer_prompt = f"""
        You are a potential customer. Your persona is: {json.dumps(customer)}.
        The salesperson just said: "{sales_response}"
        Based on your persona, how do you reply? Keep it brief.
        """
        customer_response = model.generate_content(customer_prompt).text.strip()
        transcript.append({"role": "user", "content": customer_response})

        # Determine next step using Bellman-Ford
        current_id = node_to_id[current_node]
        target_id = node_to_id.get("close_standard", len(nodes) - 1)
        
        distances = bellman_ford_list(graph, current_id)
        if not distances or distances[target_id] == float('inf'):
            current_node = "exit_bad" # No path to close
            break

        best_next_id = None
        min_total_dist = float('inf')
        for neighbor_id, weight in graph.adj_list[current_id]:
            if distances[neighbor_id] != float('inf'):
                total_dist = weight + distances[neighbor_id]
                if total_dist < min_total_dist:
                    min_total_dist = total_dist
                    best_next_id = neighbor_id
        
        if best_next_id is None:
            current_node = "exit_bad"
            break
        
        current_node = id_to_node[best_next_id]
        
        if current_node in ["close_standard", "exit_bad"]:
            break

        # Salesperson response
        sales_response = nodes[current_node]
        transcript.append({"role": "assistant", "content": sales_response})

    # 4. Score and Analyze
    outcome = "Success" if "close" in current_node else "Fail"
    score = 100 if outcome == "Success" else -50
    score -= len(transcript) # Penalty for long calls
    
    transcript_text = "\n".join([f"{m['role']}: {m['content']}" for m in transcript])
    
    log_data = {
        "scenario_id": scenario_id,
        "customer_persona": customer,
        "outcome": outcome,
        "score": score,
        "transcript": transcript_text
    }
    database.log_simulation(log_data)
    
    # 5. Phrase Analysis
    phrase_analysis = analyze_transcript(transcript_text)
    analytics_to_save = []
    for phrase in phrase_analysis.get("good_phrases", []):
        analytics_to_save.append({"scenario_id": scenario_id, "node_name": current_node, "phrase": phrase, "impact": "positive"})
    for phrase in phrase_analysis.get("bad_phrases", []):
        analytics_to_save.append({"scenario_id": scenario_id, "node_name": current_node, "phrase": phrase, "impact": "negative"})
    
    if analytics_to_save:
        database.update_phrase_analytics(analytics_to_save)

    # 6. Update Fitness Score
    database.update_scenario_fitness(scenario_id)
    
    print(f"Simulation for scenario {scenario_id} complete. Outcome: {outcome}, Score: {score}")


def main(api_key):
    """Main function to run the Colosseum simulation."""
    print("--- Starting Colosseum Simulation ---")
    genai.configure(api_key=api_key)
    database.init_db()

    # Check if we need to create the first generation
    scenarios = pd.read_sql_query("SELECT id FROM scenarios", sqlite3.connect(database.DB_FILE))
    if scenarios.empty:
        print("No scenarios found in the database. Generating initial population.")
        generate_initial_population()
        scenarios = pd.read_sql_query("SELECT id FROM scenarios", sqlite3.connect(database.DB_FILE))

    if scenarios.empty:
        print("Failed to create initial population. Exiting.")
        return

    # Run one simulation for a random scenario
    random_scenario_id = random.choice(scenarios['id'].tolist())
    print(f"\nRunning simulation for random scenario ID: {random_scenario_id}")
    run_single_simulation(random_scenario_id)
    
    print("\n--- Colosseum Simulation Finished ---")


if __name__ == "__main__":
    import sqlite3
    import os
    # This allows running the script standalone if the key is in an env var
    api_key_env = os.environ.get("GOOGLE_API_KEY")
    if api_key_env:
        main(api_key_env)
    else:
        print("Please run this module from app.py or set the GOOGLE_API_KEY environment variable.")
