import pandas as pd
from database import get_all_leads, init_db

def get_analytics():
    """
    Reads data from the SQLite database for the dashboard.
    
    Returns:
        A tuple of (DataFrame, dict) containing the data and statistics.
    """
    # Ensure the database is initialized
    init_db()
    
    try:
        df = get_all_leads()
        
        if df is None or df.empty:
            return None, None
            
        stats = {
            "total": len(df),
            "success_rate": 0,
        }
        
        if "Outcome" in df.columns and not df.empty:
            # Filter out non-success/fail outcomes for accurate rate calculation
            relevant_outcomes = df[df["Outcome"].isin(["Success", "Fail"])]
            if not relevant_outcomes.empty:
                success_count = len(relevant_outcomes[relevant_outcomes["Outcome"] == "Success"])
                stats["success_rate"] = round(success_count / len(relevant_outcomes) * 100, 1)
            
        return df, stats
    except Exception as e:
        print(f"Error getting analytics from database: {e}")
        return None, None
