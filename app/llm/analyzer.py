"""
Ollama integration for narrative analysis.

Calls local LLM to generate data insights in structured JSON format.
Validates and parses response.
"""

import requests
import json
from typing import Dict, Any, Optional
import os


# TODO: Implement LLM analysis
# - Connect to Ollama at OLLAMA_BASE_URL from .env
# - Build system prompt: "You are a data analyst"
# - Send aggregated prediction stats (salary by role, by seniority, etc.)
# - Parse strict JSON response with title, summary, key_insights, chart
# - Validate chart_type against whitelist {bar, line, scatter}
# - Fallback to bar if invalid chart_type
# - Handle Ollama failures gracefully (log warning, return None)
