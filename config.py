import os
import streamlit as st
from langchain_groq import ChatGroq

def get_groq_llm():
    # Fetch key from Streamlit Secrets or system environment
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    
    if not api_key:
        st.error("Missing Groq API Key! Please configure GROQ_API_KEY in Streamlit Secrets.")
        st.stop()
        
    return ChatGroq(
        temperature=0.2,
        model_name="openai/gpt-oss-120b",
        groq_api_key=api_key
    )
