import streamlit as st

def get_db_connection():
    """
    Get a database connection using Streamlit's native st.connection
    configured in .streamlit/secrets.toml
    """
    return st.connection("postgresql", type="sql")

def run_query(query, params=None):
    """
    Run a SQL query and return a pandas DataFrame
    """
    conn = get_db_connection()
    # Streamlit's SQL connection .query() method handles parameters 
    # and returns a pandas DataFrame directly.
    return conn.query(query, params=params, ttl=600)
