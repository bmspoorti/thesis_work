import requests
import os
import streamlit as st

class GoogleSearch:
    def __init__(self, params=None):
        self.api_key = os.getenv("SERPAPI_API_KEY") or st.secrets.get("SERPAPI_API_KEY") # Make sure this is in your .env
        self.params = params or {}
        self.params["api_key"] = self.api_key
        self.base_url = "https://serpapi.com/search"

    def get_dict(self):
        response = requests.get(self.base_url, params=self.params)
        response.raise_for_status()
        return response.json()
