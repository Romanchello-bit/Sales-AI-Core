import google.generativeai as genai
import json
import random
import time
import os
from graph_module import Graph
from algorithms import bellman_ford_list
import database
import pandas as pd
import sqlite3

MODEL_NAME = "gemini-2.5-flash"

def _build_graph_from_json(graph_json):
    nodes = graph_json.get("nodes", {})
    edges = graph_json.get("edges", [])
    node_to_id = {name: i for i, name in enumerate(nodes.keys())}
    g = Graph(len(node_to_id), directed=True)
    for e in edges:
        f = e.get("from"); t = e.get("to"); w = e.get("weight", 1)
        if f in node_to_id and t in node_to_id:
            g.add_edge(node_to_id[f], node_to_id[t], w)
    return g, node_to_id

def generate_initial_population(model, count=5):
    """Seed the DB with at least one scenario from sales_script.json or a trivial default."""
    database.init_db()
    # If scenarios already exist, do nothing
    df = database.get_all_scenarios_with_stats()
    if df is not None and not df.empty:
        return df['id'].tolist()
    # Try to load sales_script.json
    scenario_graph = None
    script_path = "sales_script.json"
    if os.path.exists(script_path):
        with open(script_path, "r", encoding="utf-8") as f:
            scenario_graph = json.load(f)
    else:
        # Minimal fallback graph
        scenario_graph = {
            "nodes": {
                "start": "Вітання та визначення потреб",
                "qualify": "Уточнюючі запитання",
                "pitch": "Коротка презентація цінності",
                "close_deal": "Погодження наступних кроків"
            },
            "edges": [
                {"from": "start", "to": "qualify", "weight": 1},
                {"from": "qualify", "to": "pitch", "weight": 1},
                {"from": "pitch", "to": "close_deal", "weight": 1}
            ]
        }
    # Insert single scenario; ignore count for now (can be extended later)
    scenario_id = database.add_scenario(scenario_graph, generation=0)
    return [scenario_id]

def generate_customer_persona():
    """Return a simple random customer persona."""
    archetypes = ["DRIVER", "ANALYST", "EXPRESSIVE", "CONSERVATIVE"]
    industries = ["SaaS", "E-commerce", "Healthcare", "Manufacturing"]
    persona = {
        "name": random.choice(["Olena", "Taras", "Iryna", "Andrii"]),
        "company": random.choice(["Acme Corp", "Globex", "Initech", "Umbrella"]),
        "archetype": random.choice(archetypes),
        "industry": random.choice(industries)
    }
    return persona

def analyze_transcript(model, transcript_text):
    """Very simple heuristic analysis: classify phrases as good/bad by keyword."""
    good_kw = ["дякую", "цінність", "покращ", "результат", "економ"]
    bad_kw = ["дорого", "неможливо", "не можу", "проблема"]
    good = []
    bad = []
    for line in transcript_text.lower().splitlines():
        if any(k in line for k in good_kw):
            good.append(line.strip())
        if any(k in line for k in bad_kw):
            bad.append(line.strip())
    return {"good_phrases": good, "bad_phrases": bad}

def run_single_simulation(model, scenario_id):
    """
    Runs one full simulation and returns a detailed report.
    """
    scenario_json = database.get_scenario(scenario_id)
    if not scenario_json:
        return {"error": f"Scenario {scenario_id} not found."}

    customer = generate_customer_persona()

    # Build graph and plan path from start to close_deal greedily by BF distances
    g, node_to_id = _build_graph_from_json(scenario_json)
    id_to_node = {i: s for s, i in node_to_id.items()}
    start_name = "start" if "start" in node_to_id else next(iter(node_to_id.keys()))
    target_name = "close_deal" if "close_deal" in node_to_id else None
    current = node_to_id[start_name]
    transcript = []
    visited = []
    steps = 0
    max_steps = len(node_to_id) * 3 if len(node_to_id) > 0 else 10
    path_nodes = [current]

    while steps < max_steps:
        node_name = id_to_node[current]
        visited.append(current)
        # Agent speaks instruction
        transcript.append({"role": "agent", "content": f"[{node_name}] Рухаємося далі..."})
        if target_name and node_name == target_name:
            break
        # Choose best neighbor by distance to target
        best_next = None
        best_total = float("inf")
        for (nbr, w) in g.get_list()[current]:
            dists = bellman_ford_list(g, nbr, visited_nodes=set(visited))
            if target_name:
                close_id = node_to_id[target_name]
                to_goal = dists[close_id]
            else:
                to_goal = 0
            total = w + to_goal
            if total < best_total:
                best_total = total
                best_next = nbr
        if best_next is None:
            break
        # Customer reply stub
        transcript.append({"role": "customer", "content": "Звучить цікаво."})
        current = best_next
        path_nodes.append(current)
        steps += 1

    final_node_name = id_to_node[current]
    outcome = "Success" if final_node_name == target_name else "Fail"
    # Score: reward success and fewer steps
    base = 100 if outcome == "Success" else -20
    score = base - steps * 2

    transcript_text = "\n".join([f"{m['role']}: {m['content']}" for m in transcript])

    log_data = {
        "scenario_id": scenario_id,
        "customer_persona": customer,
        "outcome": outcome,
        "score": score,
        "transcript": transcript_text
    }
    try:
        database.log_simulation(log_data)
    except Exception:
        pass

    phrase_analysis = analyze_transcript(model, transcript_text)

    # Save phrase analytics per GLOBAL node bucket
    analytics_rows = []
    for p in phrase_analysis.get("good_phrases", []):
        analytics_rows.append({
            "scenario_id": scenario_id,
            "node_name": "GLOBAL",
            "phrase": p,
            "impact": "GOOD",
            "count": 1
        })
    for p in phrase_analysis.get("bad_phrases", []):
        analytics_rows.append({
            "scenario_id": scenario_id,
            "node_name": "GLOBAL",
            "phrase": p,
            "impact": "BAD",
            "count": 1
        })
    if analytics_rows:
        try:
            database.update_phrase_analytics(analytics_rows)
        except Exception:
            pass

    try:
        database.update_scenario_fitness(scenario_id)
    except Exception:
        pass

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
    if scenarios_df is None or scenarios_df.empty:
        generate_initial_population(model)
        scenarios_df = database.get_all_scenarios_with_stats()
    if scenarios_df is None or scenarios_df.empty:
        return

    scenario_ids = scenarios_df['id'].tolist()
    for i in range(num_simulations):
        scenario_id = random.choice(scenario_ids)
        report = run_single_simulation(model, scenario_id)
        if progress_callback:
            progress_callback(report, i + 1, num_simulations)

    print(f"\n--- Batch of {num_simulations} Simulations Finished ---")
