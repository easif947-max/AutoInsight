import os
from crewai import Agent, Task, Crew, Process
from tools import profile_csv_dataset, filter_top_keywords

# Explicitly set model environment variable for CrewAI / LiteLLM compatibility
os.environ["OPENAI_MODEL_NAME"] = "openai/gpt-oss-20b"
MODEL_NAME = "openai/gpt-oss-20b"

# --- AGENTS CONFIGURATION ---

profiling_agent = Agent(
    role="Data Profiling & ML Analyst",
    goal="Analyze dataset structure, evaluate quantitative metrics, and detect core statistical distributions.",
    backstory="You are an expert data scientist specializing in rapid quantitative evaluation and feature profiling.",
    tools=[profile_csv_dataset],
    llm=MODEL_NAME,
    max_iter=2,
    verbose=False
)

forecasting_agent = Agent(
    role="Predictive Trend & Forecasting Analyst",
    goal="Evaluate temporal and demand patterns to project future search volume and sales metrics.",
    backstory="You are a market demand analyst skilled at identifying growth trends and forecasting demand.",
    llm=MODEL_NAME,
    max_iter=2,
    verbose=False
)

report_agent = Agent(
    role="Executive Business Report Writer",
    goal="Synthesize technical findings into an executive-level summary with strategic recommendations.",
    backstory="You are a business intelligence lead focused on transforming raw data insights into executive strategy.",
    llm=MODEL_NAME,
    max_iter=2,
    verbose=False
)

keyword_agent = Agent(
    role="Semantic Keyword & Synonym Strategist",
    goal="Expand key product search terms using semantic synonyms and identify top search volume opportunities.",
    backstory="You are an e-commerce keyword research specialist focusing on catalog visibility and synonym optimization.",
    tools=[filter_top_keywords],
    llm=MODEL_NAME,
    max_iter=2,
    verbose=False
)


def run_autoinsight_pipeline(csv_filepath: str, status_callback=None):
    """
    Executes each agent sequentially directly in the main thread to ensure
    Streamlit receives updates and generates the full response without hanging.
    """
    
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
            # Fallback output so pipeline never gets stuck indefinitely
            task_results.append(f"Phase completed with fallback response. Notice: {str(e)}")

    class CrewResultsWrapper:
        def __init__(self, outputs):
            self.tasks_output = outputs

    return CrewResultsWrapper(task_results)
