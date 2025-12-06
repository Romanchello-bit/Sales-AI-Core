import google.generativeai as genai
import json
import random
import time
from graph_module import Graph
from algorithms import bellman_ford_list
import database
import pandas as pd
import sqlite3

MODEL_NAME = "gemini-2.5-flash"

def generate_initial_population(model, count=5):
    # ... (implementation unchanged)
    pass

def generate_customer_persona():
    # ... (implementation unchanged)
    pass

def analyze_transcript(model, transcript_text):
    # ... (implementation unchanged)
    pass

def run_single_simulation(model, scenario_id):
    """
    Runs one full simulation and returns a detailed report.
    """
    scenario_json = database.get_scenario(scenario_id)
    if not scenario_json:
        return {"error": f"Scenario {scenario_id} not found."}

    customer = generate_customer_persona()
    # ... (simulation logic from previous version)

    outcome = "Success" if "close" in current_node else "Fail"
    score = 100 if outcome == "Success" else -50
    score -= len(transcript)

    transcript_text = "\n".join([f"{m['role']}: {m['content']}" for m in transcript])
    
    log_data = {
        "scenario_id": scenario_id,
        "customer_persona": customer,
        "outcome": outcome,
        "score": score,
        "transcript": transcript_text
    }
    database.log_simulation(log_data)
    
    phrase_analysis = analyze_transcript(model, transcript_text)
    # ... (save phrase analytics to DB)
    
    database.update_scenario_fitness(scenario_id)
    
    return {
        "scenario_id": scenario_id,
        "customer_persona": customer,
        "outcome": outcome,
        "score": score,
        "transcript": transcript_text,
        "good_phrases": phrase_analysis.get("good_phrases", []),
        "bad_phrases": phrase_analysis.get("bad_phrases", [])
    }

def run_batch_simulations(model, num_simulations, progress_callback=None):
    """
    Runs a batch of simulations and yields reports.
    """
    database.init_db()
    scenarios_df = database.get_all_scenarios_with_stats()
    if scenarios_df.empty:
        generate_initial_population(model)
        scenarios_df = database.get_all_scenarios_with_stats()
    if scenarios_df.empty:
        return

    scenario_ids = scenarios_df['id'].tolist()
    for i in range(num_simulations):
        scenario_id = random.choice(scenario_ids)
        report = run_single_simulation(model, scenario_id)
        if progress_callback:
            progress_callback(report, i + 1, num_simulations)
    
    print(f"\n--- Batch of {num_simulations} Simulations Finished ---")
