from openai import OpenAI

class DocumentationAgent:
    def __init__(self, name, role, api_key):
        self.name = name
        self.role = role
        self.api_client = OpenAI(api_key=api_key)

    def compile_report(self, goal, literature_review, hypotheses, experiment_design, data_analysis_summary, forecast_summary):
        try:
            prompt = (
                f"Generate a structured research report with the following sections:\n\n"
                f"Goal: {goal}\n\n"
                f"Literature Review: {literature_review}\n\n"
                f"Hypotheses: {hypotheses}\n\n"
                f"Experiment Design: {experiment_design}\n\n"
                f"Data Analysis Summary: {data_analysis_summary}\n\n"
                f"Forecast Summary: {forecast_summary}\n\n"
            )
            print(f"[{self.name}] Compiling research report...")
            response = self.api_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            report = response.choices[0].message["content"]
            print(f"[{self.name}] Report generation completed.")
            return report
        except Exception as e:
            print(f"[{self.name}] Error in report generation: {e}")
            return None
