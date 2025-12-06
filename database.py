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
    """Adds a new lead to the database.
    lead_data: dict with optional keys matching leads table columns.
    Returns inserted row id.
    """
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        # Ensure DB exists
        init_db()
        # Valid columns as per schema
        columns = [
            "Date", "Name", "Company", "Type", "Context",
            "Pain_Point", "Budget", "Outcome", "Summary",
            "Archetype", "Transcript"
        ]
        cols_used = []
        vals_used = []
        for col in columns:
            if col in lead_data:
                cols_used.append(col)
                vals_used.append(lead_data[col])
        if not cols_used:
            return None
        placeholders = ", ".join(["?"] * len(cols_used))
        cols_sql = ", ".join(cols_used)
        cursor.execute(
            f"INSERT INTO leads ({cols_sql}) VALUES ({placeholders})",
            tuple(vals_used)
        )
        conn.commit()
        # Invalidate cached readers
        try:
            st.cache_data.clear()
        except Exception:
            pass
        return cursor.lastrowid

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
        query = (
            "SELECT outcome, score, customer_persona FROM simulations "
            "WHERE scenario_id = ? ORDER BY id DESC LIMIT ?"
        )
        return pd.read_sql_query(query, conn, params=(scenario_id, limit))

@st.cache_data
def get_phrase_analytics_for_scenario(scenario_id):
    """Retrieves phrase analytics for a specific scenario."""
    with sqlite3.connect(DB_FILE) as conn:
        query = (
            "SELECT phrase, impact, count, node_name FROM phrase_analytics "
            "WHERE scenario_id = ? ORDER BY count DESC"
        )
        return pd.read_sql_query(query, conn, params=(scenario_id,))

# --- Write functions (no caching) ---
def add_scenario(graph_json, generation=0):
    """Insert a new scenario and return its ID."""
    init_db()
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO scenarios (generation, fitness_score, graph_json) VALUES (?, ?, ?)",
            (generation, 0.0, json.dumps(graph_json))
        )
        conn.commit()
        try:
            st.cache_data.clear()
        except Exception:
            pass
        return cursor.lastrowid

def log_simulation(log_data):
    """Insert a simulation log. Expects keys: scenario_id, customer_persona, outcome, score, transcript"""
    required = ["scenario_id", "customer_persona", "outcome", "score", "transcript"]
    for k in required:
        if k not in log_data:
            raise ValueError(f"Missing field in log_data: {k}")
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO simulations (scenario_id, customer_persona, outcome, score, transcript) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                log_data["scenario_id"],
                json.dumps(log_data["customer_persona"]) if not isinstance(log_data["customer_persona"], str) else log_data["customer_persona"],
                log_data["outcome"],
                int(log_data["score"]),
                log_data["transcript"],
            )
        )
        conn.commit()
        try:
            st.cache_data.clear()
        except Exception:
            pass
        return cursor.lastrowid

def update_phrase_analytics(analytics_data):
    """Update phrase analytics using upsert-like logic.
    Expects list of dicts with keys: scenario_id, node_name, phrase, impact, count
    """
    if not analytics_data:
        return 0
    updated = 0
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        for item in analytics_data:
            scenario_id = item.get("scenario_id")
            node_name = item.get("node_name")
            phrase = item.get("phrase")
            impact = item.get("impact")
            count = int(item.get("count", 1))
            if not all([scenario_id, node_name, phrase, impact]):
                continue
            # Try insert; if conflict, update count
            cursor.execute(
                "INSERT OR IGNORE INTO phrase_analytics (scenario_id, node_name, phrase, impact, count) "
                "VALUES (?, ?, ?, ?, ?)",
                (scenario_id, node_name, phrase, impact, count)
            )
            cursor.execute(
                "UPDATE phrase_analytics SET count = count + ? WHERE scenario_id = ? AND node_name = ? AND phrase = ? AND impact = ?",
                (count, scenario_id, node_name, phrase, impact)
            )
            updated += 1
        conn.commit()
        try:
            st.cache_data.clear()
        except Exception:
            pass
    return updated

def update_scenario_fitness(scenario_id):
    """Recompute and update the fitness score of a scenario as the average of its simulations' scores."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT AVG(score) FROM simulations WHERE scenario_id = ?",
            (scenario_id,)
        )
        row = cursor.fetchone()
        avg_score = row[0] if row and row[0] is not None else 0.0
        cursor.execute(
            "UPDATE scenarios SET fitness_score = ? WHERE id = ?",
            (avg_score, scenario_id)
        )
        conn.commit()
        try:
            st.cache_data.clear()
        except Exception:
            pass
        return avg_score

if __name__ == '__main__':
    print("Initializing database for Colosseum...")
    init_db()
    print("Database initialized.")
