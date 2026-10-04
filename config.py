import os
import streamlit as st
from crewai import LLM

def get_groq_llm():
    # Fetch key from Streamlit Secrets or system environment
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    
    if not api_key:
        st.error("Missing Groq API Key! Please configure GROQ_API_KEY in Streamlit Secrets.")
        st.stop()
        
    # Return CrewAI native LLM instance configured for Groq
    return LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.2
    )
