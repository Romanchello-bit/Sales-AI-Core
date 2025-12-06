import sqlite3
import pandas as pd
import json

DB_FILE = "leads.db"

def init_db():
    """Initializes all tables for the application."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        # Main leads table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Date TEXT, Name TEXT, Company TEXT, Type TEXT, Context TEXT,
                Pain_Point TEXT, Budget TEXT, Outcome TEXT, Summary TEXT,
                Archetype TEXT, Transcript TEXT
            )
        """)
        # --- Colosseum Tables ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scenarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation INTEGER DEFAULT 0,
                fitness_score REAL DEFAULT 0.0,
                graph_json TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS simulations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scenario_id INTEGER,
                customer_persona TEXT,
                outcome TEXT,
                score INTEGER,
                transcript TEXT,
                FOREIGN KEY (scenario_id) REFERENCES scenarios (id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS phrase_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scenario_id INTEGER,
                node_name TEXT,
                phrase TEXT,
                impact TEXT,
                count INTEGER DEFAULT 1,
                UNIQUE(scenario_id, node_name, phrase, impact),
                FOREIGN KEY (scenario_id) REFERENCES scenarios (id)
            )
        """)
        conn.commit()

def add_lead(lead_data):
    """Adds a new lead to the database."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        columns = ', '.join(lead_data.keys())
        placeholders = ', '.join(['?'] * len(lead_data))
        sql = f"INSERT INTO leads ({columns}) VALUES ({placeholders})"
        cursor.execute(sql, tuple(lead_data.values()))
        conn.commit()

def get_all_leads():
    """Retrieves all leads from the database."""
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql_query("SELECT * FROM leads", conn)

# --- Colosseum Functions ---

def add_scenario(graph_json, generation=0):
    """Adds a new scenario to the database."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO scenarios (generation, graph_json) VALUES (?, ?)",
            (generation, json.dumps(graph_json))
        )
        conn.commit()
        return cursor.lastrowid

def get_scenario(scenario_id):
    """Retrieves a specific scenario."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT graph_json FROM scenarios WHERE id = ?", (scenario_id,))
        row = cursor.fetchone()
        return json.loads(row[0]) if row else None

def log_simulation(log_data):
    """Logs the result of a single simulation."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO simulations (scenario_id, customer_persona, outcome, score, transcript)
            VALUES (?, ?, ?, ?, ?)
        """, (
            log_data['scenario_id'], json.dumps(log_data['customer_persona']),
            log_data['outcome'], log_data['score'], log_data['transcript']
        ))
        conn.commit()

def update_phrase_analytics(analytics_data):
    """Updates the analytics for a given phrase."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        for phrase_info in analytics_data:
            cursor.execute("""
                INSERT INTO phrase_analytics (scenario_id, node_name, phrase, impact)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(scenario_id, node_name, phrase, impact) DO UPDATE SET count = count + 1
            """, (
                phrase_info['scenario_id'], phrase_info['node_name'],
                phrase_info['phrase'], phrase_info['impact']
            ))
        conn.commit()

def update_scenario_fitness(scenario_id):
    """Recalculates and updates the fitness score for a scenario."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT AVG(score) FROM simulations WHERE scenario_id = ?",
            (scenario_id,)
        )
        avg_score = cursor.fetchone()[0]
        if avg_score is not None:
            cursor.execute(
                "UPDATE scenarios SET fitness_score = ? WHERE id = ?",
                (round(avg_score, 2), scenario_id)
            )
        conn.commit()

# --- Evolution Hub Functions ---

def get_all_scenarios_with_stats():
    """Retrieves all scenarios with aggregated stats."""
    with sqlite3.connect(DB_FILE) as conn:
        query = """
        SELECT
            s.id,
            s.generation,
            s.fitness_score,
            COUNT(sim.id) as simulation_count
        FROM scenarios s
        LEFT JOIN simulations sim ON s.id = sim.scenario_id
        GROUP BY s.id
        ORDER BY s.fitness_score DESC
        """
        return pd.read_sql_query(query, conn)

def get_simulations_for_scenario(scenario_id, limit=10):
    """Retrieves recent simulations for a specific scenario."""
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql_query(
            f"SELECT outcome, score, customer_persona FROM simulations WHERE scenario_id = {scenario_id} ORDER BY id DESC LIMIT {limit}",
            conn
        )

def get_phrase_analytics_for_scenario(scenario_id):
    """Retrieves phrase analytics for a specific scenario."""
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql_query(
            f"SELECT phrase, impact, count, node_name FROM phrase_analytics WHERE scenario_id = {scenario_id} ORDER BY count DESC",
            conn
        )

if __name__ == '__main__':
    print("Initializing database for Colosseum...")
    init_db()
    print("Database initialized.")
