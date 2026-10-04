from crewai import Agent
from config import get_groq_llm
from tools import profile_csv_dataset

def get_profiling_agent():
    return Agent(
        role="Data Profiling & ML Insights Specialist",
        goal="Profile dataset metrics, assess revenue, correlation, and data hygiene.",
        backstory="Expert lead data scientist specializing in automated e-commerce and marketing data profiling.",
        tools=[profile_csv_dataset],
        llm=get_groq_llm(),
        verbose=True
    )
