from src.utils import get_kaggle_client, get_openai_client
from src.agents.DataRetrievalAgent import DataRetrievalAgent
from src.agents.LiteratureReviewAgent import LiteratureReviewAgent
from src.agents.HypothesisAgent import HypothesisAgent
from src.agents.ExperimentationAgent import ExperimentationAgent
from src.agents.DataAnalysisAgent import DataAnalysisAgent
from src.agents.PredictiveModelAgent import PredictiveModelAgent
from src.agents.DocumentationAgent import DocumentationAgent

def test_agents():
    kaggle_client = get_kaggle_client()
    openai_client = get_openai_client()

    data_agent = DataRetrievalAgent("Data Retrieval Agent", "Handles data retrieval and preprocessing")
    df_raw = data_agent.retrieve_data("berkeleyearth/climate-change-earth-surface-temperature-data", "GlobalLandTemperaturesByCity.csv")
    df = data_agent.preprocess_data(df_raw)

    review_agent = LiteratureReviewAgent("Literature Review Agent", "Conducts literature reviews", openai_client)
    literature_review = review_agent.conduct_review("Impact of climate change on global temperature trends")

    hypothesis_agent = HypothesisAgent("Hypothesis Agent", "Generates hypotheses", openai_client)
    hypotheses = hypothesis_agent.generate_hypotheses(literature_review)

    experiment_agent = ExperimentationAgent("Experiment Agent", "Designs experiments", openai_client)
    experiment_design = experiment_agent.design_experiment(hypotheses)

    analysis_agent = DataAnalysisAgent("Data Analysis Agent", "Performs data analysis")
    analysis_agent.plot_time_series(df)
    analysis_agent.spectral_analysis(df)
    analysis_agent.rolling_statistics(df)

    model_agent = PredictiveModelAgent("Predictive Model Agent", "Forecasts trends")
    arima_forecast = model_agent.arima_forecast(df)
    sarima_forecast = model_agent.sarima_forecast(df)
    model_agent.plot_forecasts(df, arima_forecast, sarima_forecast)

    doc_agent = DocumentationAgent("Documentation Agent", "Compiles final report", openai_client)
    report = doc_agent.compile_report(
        "Investigate climate change effects on temperature",
        literature_review,
        hypotheses,
        experiment_design,
        "Analysis summary",
        "Forecast summary"
    )
    print("Test Report:", report)

if __name__ == "__main__":
    test_agents()
