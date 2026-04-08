"""
Streamlit dashboard that reads from Supabase only.

Displays:
- Prediction history / runs
- Prediction results
- LLM narrative summary
- Key insights
- Reconstructed chart from stored spec
- Graceful empty states
"""

import streamlit as st
import pandas as pd
from app.storage.repository import Repository


# TODO: Implement dashboard
# - Connect to Supabase via repository
# - Fetch latest run metadata
# - Fetch prediction results
# - Fetch LLM analysis (title, summary, key_insights)
# - Reconstruct and render chart from chart_spec_json + chart_data_json
# - Add manual refresh button
# - Display "Last updated" timestamp
# - Handle graceful empty states
