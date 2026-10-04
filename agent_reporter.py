from crewai import Agent
from config import get_groq_llm

def get_reporting_agent():
    return Agent(
        role="Executive Business Reporting Strategist",
        goal="Synthesize ML findings into actionable executive business reports.",
        backstory="Former McKinsey management consultant specializing in turning data insights into growth strategy.",
        llm=get_groq_llm(),
        verbose=True
    )
