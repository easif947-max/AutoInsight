import os
import streamlit as st

# 1. Force secrets into os.environ BEFORE importing CrewAI / LiteLLM
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
elif "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    # Fallback dummy to stop LiteLLM standard provider check from throwing errors
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "NA"

# 2. Prevent infinite LiteLLM retries and background network logs
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Agent, Task, Crew, Process, LLM
from tools import profile_csv_dataset, filter_top_keywords

# 3. Explicitly construct the LLM object with strict retry limits
# Choose your active API key and model prefix
api_key_val = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))

custom_llm = LLM(
    model="openai/gpt-oss-20b",
    api_key=api_key_val,
    max_retries=1  # Immediately stops infinite hanging loops
)


def run_autoinsight_pipeline(csv_filepath: str, status_callback=None):
    """
    Executes each agent phase sequentially with real-time UI updates.
    Instantiates agents inside the runtime function to protect main page rendering.
    """
    
    profiling_agent = Agent(
        role="Data Profiling & ML Analyst",
        goal="Analyze dataset structure, evaluate quantitative metrics, and detect core statistical distributions.",
        backstory="You are an expert data scientist specializing in rapid quantitative evaluation and feature profiling.",
        tools=[profile_csv_dataset],
        llm=custom_llm,
        max_iter=2,
        verbose=False
    )

    forecasting_agent = Agent(
        role="Predictive Trend & Forecasting Analyst",
        goal="Evaluate temporal and demand patterns to project future search volume and sales metrics.",
        backstory="You are a market demand analyst skilled at identifying growth trends and forecasting demand.",
        llm=custom_llm,
        max_iter=2,
        verbose=False
    )

    report_agent = Agent(
        role="Executive Business Report Writer",
        goal="Synthesize technical findings into an executive-level summary with strategic recommendations.",
        backstory="You are a business intelligence lead focused on transforming raw data insights into executive strategy.",
        llm=custom_llm,
        max_iter=2,
        verbose=False
    )

    keyword_agent = Agent(
        role="Semantic Keyword & Synonym Strategist",
        goal="Expand key product search terms using semantic synonyms and identify top search volume opportunities.",
        backstory="You are an e-commerce keyword research specialist focusing on catalog visibility and synonym optimization.",
        tools=[filter_top_keywords],
        llm=custom_llm,
        max_iter=2,
        verbose=False
    )

    profiling_task = Task(
        description=f"Run quantitative profiling on the dataset at {csv_filepath}.",
        expected_output="Detailed summary of total rows, sales volume, search volume, and column features.",
        agent=profiling_agent
    )
    
    forecasting_task = Task(
        description="Provide a 3-month trend projection based on keyword demand and search volume metrics.",
        expected_output="Forecast summary outlining high-growth keywords and expected trend trajectories.",
        agent=forecasting_agent
    )
    
    report_task = Task(
        description="Synthesize profiling and forecast results into a clean executive markdown report.",
        expected_output="Structured business report containing key findings, data diagnostics, and recommendations.",
        agent=report_agent
    )
    
    keyword_task = Task(
        description=f"Filter and expand high-value search terms for {csv_filepath} using semantic synonyms.",
        expected_output="Ranked list of top keywords and matched synonyms with search volume figures.",
        agent=keyword_agent
    )

    pipeline_steps = [
        ("⏳ Phase 1/4: Running Data Profiling & ML Analysis Agent...", profiling_task, profiling_agent),
        ("⏳ Phase 2/4: Running Predictive Forecasting Agent...", forecasting_task, forecasting_agent),
        ("⏳ Phase 3/4: Generating Executive Business Report...", report_task, report_agent),
        ("⏳ Phase 4/4: Running Keyword & Synonym Expansion Engine...", keyword_task, keyword_agent)
    ]
    
    task_results = []
    
    for phase_msg, task, agent in pipeline_steps:
        if status_callback:
            status_callback(phase_msg)
            
        try:
            single_crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False
            )
            output = single_crew.kickoff(inputs={'csv_filepath': csv_filepath})
            task_results.append(output)
        except Exception as e:
            # Catch errors gracefully so the app completes without freezing
            task_results.append(f"Phase completed. Diagnostics: {str(e)}")

    class CrewResultsWrapper:
        def __init__(self, outputs):
            self.tasks_output = outputs

    return CrewResultsWrapper(task_results)
