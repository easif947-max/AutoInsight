from crewai import Crew, Task, Process
from agent_profiler import get_profiling_agent
from agent_forecaster import get_forecasting_agent
from agent_reporter import get_reporting_agent
from agent_keyword import get_keyword_agent

def run_autoinsight_pipeline(csv_path: str, status_callback=None):
    profiler = get_profiling_agent()
    forecaster = get_forecasting_agent()
    reporter = get_reporting_agent()
    keyword_agent = get_keyword_agent()

    if status_callback:
        status_callback("Phase 1/4: Running Data Profiling & ML Analysis Agent...")

    t1 = Task(
        description=f"Run quantitative profiling on the dataset located at {csv_path}. Calculate total revenue, identify top products, and check correlations.",
        expected_output="Detailed data profiling summary with total metrics and key trends.",
        agent=profiler
    )

    t2 = Task(
        description="Based on profiling results, generate a 3-month predictive forecast for units sold and search volume growth using slope trends.",
        expected_output="3-Month forecasting report with explicit numerical projections.",
        agent=forecaster
    )

    t3 = Task(
        description="Synthesize technical findings into an executive summary report with 3 key growth recommendations.",
        expected_output="Executive business report formatted in Markdown.",
        agent=reporter
    )

    t4 = Task(
        description=f"Run synonym expansion on the target terms in {csv_path} using tool `Keyword and Synonym Filter Engine` for keyword 'cat collar'. Output top converting terms.",
        expected_output="Ranked table of expanded semantic keywords and search volume impact.",
        agent=keyword_agent
    )

    crew = Crew(
        agents=[profiler, forecaster, reporter, keyword_agent],
        tasks=[t1, t2, t3, t4],
        process=Process.sequential,
        verbose=True
    )

    results = crew.kickoff()
    return results
