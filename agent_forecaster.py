from crewai import Agent
from config import get_groq_llm

def get_forecasting_agent():
    return Agent(
        role="Predictive Forecasting & Trend Specialist",
        goal="Analyze directional growth trends and build 3-month predictive projections.",
        backstory="Quantitative financial analyst skilled in slope calculations and demand forecasting.",
        llm=get_groq_llm(),
        verbose=True
    )
