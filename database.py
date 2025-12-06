import sqlite3
import pandas as pd
import json
import streamlit as st

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
        # ... (implementation unchanged)
        pass

# --- Functions that write data don't get cached ---

@st.cache_data
def get_all_leads():
    """Retrieves all leads from the database."""
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql_query("SELECT * FROM leads", conn)

@st.cache_data
def get_scenario(scenario_id):
    """Retrieves a specific scenario."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT graph_json FROM scenarios WHERE id = ?", (scenario_id,))
        row = cursor.fetchone()
        return json.loads(row[0]) if row else None

# --- Evolution Hub Read Functions ---

@st.cache_data
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

@st.cache_data
def get_simulations_for_scenario(scenario_id, limit=10):
    """Retrieves recent simulations for a specific scenario."""
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql_query(
            f"SELECT outcome, score, customer_persona FROM simulations WHERE scenario_id = {scenario_id} ORDER BY id DESC LIMIT {limit}",
            conn
        )

@st.cache_data
def get_phrase_analytics_for_scenario(scenario_id):
    """Retrieves phrase analytics for a specific scenario."""
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql_query(
            f"SELECT phrase, impact, count, node_name FROM phrase_analytics WHERE scenario_id = {scenario_id} ORDER BY count DESC",
            conn
        )

# --- Write functions (no caching) ---
def add_scenario(graph_json, generation=0):
    # ... (implementation unchanged)
    pass
def log_simulation(log_data):
    # ... (implementation unchanged)
    pass
def update_phrase_analytics(analytics_data):
    # ... (implementation unchanged)
    pass
def update_scenario_fitness(scenario_id):
    # ... (implementation unchanged)
    pass

if __name__ == '__main__':
    print("Initializing database for Colosseum...")
    init_db()
    print("Database initialized.")
