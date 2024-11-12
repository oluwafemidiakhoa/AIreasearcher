from openai import OpenAI

class ExperimentationAgent:
    def __init__(self, name, role, api_key):
        self.name = name
        self.role = role
        self.api_client = OpenAI(api_key=api_key)

    def design_experiment(self, hypothesis):
        try:
            prompt = f"Design an experiment to test the following hypothesis: {hypothesis}"
            print(f"[{self.name}] Designing experiment...")
            response = self.api_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            experiment_plan = response.choices[0].message["content"]
            print(f"[{self.name}] Experiment design completed.")
            return experiment_plan
        except Exception as e:
            print(f"[{self.name}] Error in experiment design: {e}")
            return None
