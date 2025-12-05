import pandas as pd
import os
from datetime import datetime

LEADS_FILE = "leads.csv"

def init_db():
    """Створює файл, якщо його немає"""
    if not os.path.exists(LEADS_FILE):
        df = pd.DataFrame(columns=[
            "Date", "Client Name", "Company", "Type", "Context", 
            "Outcome", "Final Step", "Summary"
        ])
        df.to_csv(LEADS_FILE, index=False)

def save_lead(lead_data, outcome, final_step, chat_history):
    """Зберігає результат розмови"""
    init_db()
    
    # Робимо просте самарі (останні 2 повідомлення або статус)
    summary = f"Ended at {final_step}. Total msgs: {len(chat_history)}"
    
    new_row = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Client Name": lead_data.get("name", "Unknown"),
        "Company": lead_data.get("company", "-"),
        "Type": lead_data.get("type", "B2B"),
        "Context": lead_data.get("context", "Cold Call"),
        "Outcome": outcome,
        "Final Step": final_step,
        "Summary": summary
    }
    
    df = pd.read_csv(LEADS_FILE)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(LEADS_FILE, index=False)
    return True

def get_analytics():
    """Повертає статистику для дашборду"""
    init_db()
    df = pd.read_csv(LEADS_FILE)
    if df.empty:
        return None
        
    stats = {
        "total": len(df),
        "success_rate": round(len(df[df["Outcome"] == "Success"]) / len(df) * 100, 1),
        "top_fail_reasons": df[df["Outcome"] == "Fail"]["Final Step"].value_counts().head(3)
    }
    return df, stats
