import sqlite3
import pandas as pd

DB_FILE = "leads.db"

def init_db():
    """Initializes the database and creates the 'leads' table if it doesn't exist."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Date TEXT,
                Name TEXT,
                Company TEXT,
                Type TEXT,
                Context TEXT,
                Pain_Point TEXT,
                Budget TEXT,
                Outcome TEXT,
                Summary TEXT,
                Archetype TEXT,
                Transcript TEXT
            )
        """)
        conn.commit()

def add_lead(lead_data):
    """
    Adds a new lead to the database.
    
    Args:
        lead_data (dict): A dictionary containing all lead information.
    """
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        columns = ', '.join(lead_data.keys())
        placeholders = ', '.join(['?'] * len(lead_data))
        sql = f"INSERT INTO leads ({columns}) VALUES ({placeholders})"
        cursor.execute(sql, tuple(lead_data.values()))
        conn.commit()

def get_all_leads():
    """
    Retrieves all leads from the database.
    
    Returns:
        pandas.DataFrame: A DataFrame containing all lead records.
    """
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query("SELECT * FROM leads", conn)
        return df

if __name__ == '__main__':
    # Example usage and migration from CSV
    print("Initializing database...")
    init_db()
    print("Database initialized.")

    # Check if old CSV exists and migrate data
    try:
        if pd.io.common.file_exists("leads_database.csv"):
            print("Found old CSV file. Migrating data...")
            old_df = pd.read_csv("leads_database.csv")
            
            # Ensure all columns match the new schema
            db_cols = ["Date", "Name", "Company", "Type", "Context", "Pain_Point", "Budget", "Outcome", "Summary", "Archetype", "Transcript"]
            for col in db_cols:
                if col not in old_df.columns:
                    old_df[col] = None # Add missing columns with None
            
            # Rename columns to match DB schema (e.g., "Pain Point" -> "Pain_Point")
            old_df.rename(columns={"Pain Point": "Pain_Point"}, inplace=True)

            with sqlite3.connect(DB_FILE) as conn:
                old_df.to_sql('leads', conn, if_exists='append', index=False)
            
            print(f"Migrated {len(old_df)} records.")
            # Optional: rename the old file to prevent re-migration
            import os
            os.rename("leads_database.csv", "leads_database.csv.migrated")
            print("Renamed old CSV file to 'leads_database.csv.migrated'")

    except Exception as e:
        print(f"Could not migrate from CSV: {e}")

    print("\nTesting database functions:")
    print("Total leads in DB:", len(get_all_leads()))
