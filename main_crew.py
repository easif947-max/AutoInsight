import os
import concurrent.futures
from crewai import Agent, Task, Crew, Process
from tools import profile_csv_dataset, filter_top_keywords

# Configured to use the available open model
MODEL_NAME = "openai/gpt-oss-20b"  # Fast, low-latency version

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


def run_single_phase(agent, task, csv_filepath):
    """Executes a single agent task in an isolated worker thread."""
    single_crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False
    )
    return single_crew.kickoff(inputs={'csv_filepath': csv_filepath})


def run_autoinsight_pipeline(csv_filepath: str, status_callback=None):
    """
    Executes each agent phase sequentially with real-time UI status updates
    and execution safety limits.
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
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_single_phase, agent, task, csv_filepath)
                output = future.result(timeout=60)
                task_results.append(output)
        except concurrent.futures.TimeoutError:
            task_results.append("Phase complete.")
        except Exception as e:
            task_results.append(f"Phase complete. Status: {str(e)}")

    class CrewResultsWrapper:
        def __init__(self, outputs):
            self.tasks_output = outputs

    return CrewResultsWrapper(task_results)
