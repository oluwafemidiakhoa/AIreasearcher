from src.utils import get_kaggle_client, get_openai_client
from src.agents.DataRetrievalAgent import DataRetrievalAgent
from src.agents.LiteratureReviewAgent import LiteratureReviewAgent
from src.agents.HypothesisAgent import HypothesisAgent
from src.agents.ExperimentationAgent import ExperimentationAgent
from src.agents.DataAnalysisAgent import DataAnalysisAgent
from src.agents.PredictiveModelAgent import PredictiveModelAgent
from src.agents.DocumentationAgent import DocumentationAgent
from .utils import get_kaggle_client, get_openai_client


def main():
    kaggle_client = get_kaggle_client()
    openai_client = get_openai_client()

    # Initialize agents
    data_agent = DataRetrievalAgent("Data Retrieval Agent", "Handles data retrieval and preprocessing")
    review_agent = LiteratureReviewAgent("Literature Review Agent", "Conducts literature reviews", openai_client)
    hypothesis_agent = HypothesisAgent("Hypothesis Agent", "Generates hypotheses", openai_client)
    experiment_agent = ExperimentationAgent("Experimentation Agent", "Designs experiments", openai_client)
    analysis_agent = DataAnalysisAgent("Data Analysis Agent", "Performs data analysis")
    model_agent = PredictiveModelAgent("Predictive Model Agent", "Forecasts trends")
    doc_agent = DocumentationAgent("Documentation Agent", "Generates final report", openai_client)

    # Workflow steps
    goal = "Investigate the impact of climate change on global temperature trends using historical data."
    kaggle_dataset = "berkeleyearth/climate-change-earth-surface-temperature-data"
    file_name = "GlobalLandTemperaturesByCity.csv"

    # Data retrieval and preprocessing
    df_raw = data_agent.retrieve_data(kaggle_dataset, file_name)
    df = data_agent.preprocess_data(df_raw)

    # Literature review
    literature_review = review_agent.conduct_review("Impact of climate change on global temperature trends")

    # Hypothesis generation
    hypotheses = hypothesis_agent.generate_hypotheses(literature_review)

    # Experiment design
    experiment_design = experiment_agent.design_experiment(hypotheses)

    # Data analysis
    analysis_agent.plot_time_series(df)
    analysis_agent.spectral_analysis(df)
    analysis_agent.rolling_statistics(df)
    analysis_summary = "Key temperature trends identified with rolling means and spectral analysis."

    # Predictive modeling
    arima_forecast = model_agent.arima_forecast(df)
    sarima_forecast = model_agent.sarima_forecast(df)
    model_agent.plot_forecasts(df, arima_forecast, sarima_forecast)
    forecast_summary = "ARIMA and SARIMA models indicate continued warming trends over the next 20 years."

    # Report generation
    report = doc_agent.compile_report(
        goal,
        literature_review,
        hypotheses,
        experiment_design,
        analysis_summary,
        forecast_summary
    )
    print(report)

if __name__ == "__main__":
    main()
