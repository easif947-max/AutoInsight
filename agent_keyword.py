from crewai import Agent
from config import get_groq_llm
from tools import filter_top_keywords

def get_keyword_agent():
    return Agent(
        role="Semantic Keyword & Synonym Optimization Engine",
        goal="Filter top keywords, expand product root terms, and match search trends.",
        backstory="SEO and e-commerce search algorithm expert.",
        tools=[filter_top_keywords],
        llm=get_groq_llm(),
        verbose=True
    )
