import os
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# ✅ Add this at the top
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

def log_to_gsheet(timestamp, query, agent, curriculum_mode, latency, is_fallback, result):
    # Decide whether to use local service account file or Streamlit secrets
    if os.path.exists("logs/gcp_service_account.json"):
        creds = Credentials.from_service_account_file("logs/gcp_service_account.json", scopes=SCOPES)
    else:
        # Use secrets on Streamlit Cloud
        creds_dict = {
            "type": st.secrets["gcp_service_account"]["type"],
            "project_id": st.secrets["gcp_service_account"]["project_id"],
            "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
            "private_key": st.secrets["gcp_service_account"]["private_key"].replace("\\n", "\n"),
            "client_email": st.secrets["gcp_service_account"]["client_email"],
            "client_id": st.secrets["gcp_service_account"]["client_id"],
            "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
            "token_uri": st.secrets["gcp_service_account"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"],
            "universe_domain": st.secrets["gcp_service_account"]["universe_domain"],
        }
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)

    # Authorize and log
    client = gspread.authorize(creds)
    sheet = client.open("WorkflowLogs").sheet1

    row = [
        timestamp,
        (query or "").replace("\n", " ").strip(),
        (agent or "").strip(),
        (curriculum_mode or "").strip(),
        round(latency, 2) if latency else "",
        "Yes" if is_fallback else "No",
        (result or "").replace("\n", " ").strip()[:500]
    ]
    sheet.append_row(row)
