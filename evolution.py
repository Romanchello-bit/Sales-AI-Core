import google.generativeai as genai
import json
import random
import sqlite3
import pandas as pd
import database

# --- CONFIGURATION ---
MODEL_NAME = "gemini-2.5-flash"
# API_KEY is now configured globally from app.py

# --- CORE FUNCTIONS ---

def run_evolution_cycle(top_n_champions=2, num_mutants=2, num_hybrids=1):
    """
    Runs a full evolution cycle: selection, mutation, and crossover.
    """
    print("--- Starting Evolution Cycle ---")
    
    # 1. SELECTION: Get the best scenarios from the last generation
    with sqlite3.connect(database.DB_FILE) as conn:
        scenarios_df = pd.read_sql_query(
            "SELECT * FROM scenarios ORDER BY fitness_score DESC", conn
        )
    
    if scenarios_df.empty:
        print("No scenarios to evolve. Run colosseum.py first.")
        return

    last_generation = scenarios_df['generation'].max()
    champions = scenarios_df.head(top_n_champions)
    
    print(f"Selected {len(champions)} champions from generation {last_generation}.")
    
    new_generation = last_generation + 1
    
    # Champions survive to the next generation
    for _, champ in champions.iterrows():
        champ_graph = json.loads(champ['graph_json'])
        database.add_scenario(champ_graph, generation=new_generation)
    
    print(f"Champions have been moved to generation {new_generation}.")

    # 2. MUTATION
    print(f"Creating {num_mutants} mutants...")
    model = genai.GenerativeModel(MODEL_NAME)
    
    for i in range(num_mutants):
        if champions.empty: continue
        mutant_base = champions.sample(1).iloc[0]
        scenario_id = mutant_base['id']
        
        with sqlite3.connect(database.DB_FILE) as conn:
            bad_phrases_df = pd.read_sql_query(f"""
                SELECT node_name, phrase FROM phrase_analytics
                WHERE scenario_id = {scenario_id} AND impact = 'negative'
                ORDER BY count DESC LIMIT 1
            """, conn)

        if bad_phrases_df.empty:
            print(f"No negative phrases found for champion {scenario_id} to mutate. Skipping mutation.")
            continue

        node_to_mutate = bad_phrases_df['node_name'].iloc[0]
        bad_phrase = bad_phrases_df['phrase'].iloc[0]
        
        mutant_graph = json.loads(mutant_base['graph_json'])
        original_text = mutant_graph['nodes'].get(node_to_mutate, "")

        prompt = f"""
        You are a sales script optimizer.
        The following text in a sales script node has been identified as performing poorly:
        Original Text: "{original_text}"
        Specifically, the phrase "{bad_phrase}" was received negatively by customers.

        Rewrite the 'Original Text' to achieve the same goal but without using the negative phrase and with a better, more positive tone.
        Return only the new text for the node.
        """
        
        try:
            response = model.generate_content(prompt)
            new_text = response.text.strip()
            mutant_graph['nodes'][node_to_mutate] = new_text
            database.add_scenario(mutant_graph, generation=new_generation)
            print(f"Created mutant from scenario {scenario_id}. Node '{node_to_mutate}' was changed.")
        except Exception as e:
            print(f"Could not generate mutation: {e}")

    # 3. CROSSOVER
    print(f"Creating {num_hybrids} hybrids...")
    if len(champions) < 2:
        print("Not enough champions to perform crossover. Need at least 2.")
    else:
        for i in range(num_hybrids):
            parents = champions.sample(2)
            parent_a_graph = json.loads(parents.iloc[0]['graph_json'])
            parent_b_graph = json.loads(parents.iloc[1]['graph_json'])
            
            hybrid_graph = {
                "nodes": {},
                "edges": []
            }
            
            # Simple crossover: take intro from A, closing from B
            intro_nodes = ["start", "qualification_1", "qualification_2"]
            
            for node_name, node_text in parent_a_graph["nodes"].items():
                if node_name in intro_nodes:
                    hybrid_graph["nodes"][node_name] = node_text
            
            for node_name, node_text in parent_b_graph["nodes"].items():
                if node_name not in intro_nodes:
                    hybrid_graph["nodes"][node_name] = node_text
            
            # Combine edges, removing duplicates
            hybrid_edges = parent_a_graph["edges"] + parent_b_graph["edges"]
            hybrid_graph["edges"] = [dict(t) for t in {tuple(d.items()) for d in hybrid_edges}]
            
            database.add_scenario(hybrid_graph, generation=new_generation)
            print(f"Created hybrid from scenarios {parents.iloc[0]['id']} and {parents.iloc[1]['id']}.")

    print(f"\n--- Evolution Cycle Complete. New generation {new_generation} is ready. ---")


def main(api_key):
    """Main function to run the evolution cycle."""
    print("--- Starting Evolution Cycle ---")
    genai.configure(api_key=api_key)
    run_evolution_cycle()
    print("\n--- Evolution Cycle Finished ---")


if __name__ == "__main__":
    import os
    # This allows running the script standalone if the key is in an env var
    api_key_env = os.environ.get("GOOGLE_API_KEY")
    if api_key_env:
        main(api_key_env)
    else:
        print("Please run this module from app.py or set the GOOGLE_API_KEY environment variable.")
