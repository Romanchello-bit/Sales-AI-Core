import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# Назва твоєї таблиці в Google Sheets (має співпадати буква в букву!)
SHEET_NAME = "SellMe_Leads"

def connect_to_gsheet():
    """Підключення до Google Sheets через Secrets"""
    try:
        # Створюємо об'єкт облікових даних із секретів Streamlit
        # Streamlit автоматично конвертує TOML секцію [gcp_service_account] у словник
        if "gcp_service_account" not in st.secrets:
            return None

        creds_dict = dict(st.secrets["gcp_service_account"])
        
        # Виправляємо переноси рядків у приватному ключі (часта проблема при копіюванні)
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")

        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        # Відкриваємо таблицю
        sheet = client.open(SHEET_NAME).sheet1
        return sheet
    except Exception as e:
        # st.error(f"❌ Помилка підключення до Google Sheets: {e}") 
        # При кожному рерані може бути помилка якщо немає секретів, краще тихо
        return None

def save_lead_to_db(lead_info, chat_history, outcome):
    """Зберігає ліда в Google Таблицю"""
    sheet = connect_to_gsheet()
    if not sheet:
        return # Якщо немає зв'язку, виходимо
    
    # Якщо таблиця порожня, додамо заголовки
    try:
        if not sheet.get_all_values():
            sheet.append_row([
                "Date", "Name", "Company", "Type", "Context", 
                "Pain Point", "Budget", "Outcome", "Transcript"
            ])
    except:
        pass # Таблиця може бути новою

    # Формуємо рядок даних
    # Збираємо весь текст діалогу для навчання
    transcript = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        lead_info.get("name", "-"),
        lead_info.get("company", "-"),
        lead_info.get("type", "-"),
        lead_info.get("context", "-"),
        "AI Pending", # Тут можна додати AI аналіз
        "Unknown",
        outcome,
        transcript
    ]
    
    # Додаємо рядок
    sheet.append_row(row)
    print("✅ Дані збережено в Google Sheets!")

def get_analytics():
    """Читає дані з Google Таблиці для дашборду"""
    sheet = connect_to_gsheet()
    if not sheet:
        return None, None
        
    # Отримуємо всі записи
    try:
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        
        if df.empty:
            return None, None
            
        stats = {
            "total": len(df),
            "success_rate": 0,
            "top_fail_reasons": None
        }
        
        if "Outcome" in df.columns and not df.empty:
            success_count = len(df[df["Outcome"] == "Success"])
            stats["success_rate"] = round(success_count / len(df) * 100, 1)
            
        return df, stats
    except:
        return None, None
