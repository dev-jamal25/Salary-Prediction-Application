"""
Supabase client initialization and connection management.

Loads configuration from environment variables:
- SUPABASE_URL: Supabase project URL
- SUPABASE_KEY: Supabase anon/public key
- SUPABASE_SERVICE_ROLE_KEY: (optional) Service role key for admin operations

Usage:
    from app.storage.supabase_client import get_client
    client = get_client()
    response = client.table("prediction_runs").select("*").execute()
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global client instance (lazy-loaded)
_client = None


def get_client():
    """
    Get or initialize the singleton Supabase client.
    
    Loads credentials from environment variables.
    Raises ValueError if required env vars are missing.
    
    Returns:
        Initialized Supabase client instance
    """
    global _client
    
    if _client is not None:
        return _client
    
    # Load configuration from environment
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    # Validate required configuration
    if not supabase_url:
        raise ValueError(
            "SUPABASE_URL environment variable not set. "
            "Add it to .env or set it in your environment."
        )
    if not supabase_key:
        raise ValueError(
            "SUPABASE_KEY environment variable not set. "
            "Add it to .env or set it in your environment."
        )
    
    try:
        from supabase import create_client
    except ImportError:
        raise ImportError(
            "supabase library not installed. "
            "Install it with: pip install supabase"
        )
    
    # Initialize and cache the client
    _client = create_client(supabase_url, supabase_key)
    logger.info(f"Supabase client initialized: {supabase_url}")
    
    return _client


def init_client_from_env():
    """
    Initialize the client from environment variables.
    Useful for explicit initialization before operations.
    
    Raises:
        ValueError: If required environment variables are missing
    """
    get_client()


def get_service_role_client():
    """
    Get a Supabase client with service role credentials (admin operations).
    
    Useful for operations that require elevated permissions.
    Requires SUPABASE_SERVICE_ROLE_KEY environment variable.
    
    Returns:
        Initialized Supabase client with service role key
        
    Raises:
        ValueError: If SUPABASE_SERVICE_ROLE_KEY is not set
    """
    supabase_url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url:
        raise ValueError("SUPABASE_URL environment variable not set.")
    if not service_role_key:
        raise ValueError(
            "SUPABASE_SERVICE_ROLE_KEY environment variable not set. "
            "This is required for admin operations."
        )
    
    try:
        from supabase import create_client
    except ImportError:
        raise ImportError("supabase library not installed. Install with: pip install supabase")
    
    client = create_client(supabase_url, service_role_key)
    logger.info("Service role Supabase client initialized")
    return client
