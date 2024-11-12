from openai import OpenAI

class HypothesisAgent:
    def __init__(self, name, role, api_key):
        self.name = name
        self.role = role
        self.api_client = OpenAI(api_key=api_key)

    def generate_hypotheses(self, literature):
        try:
            prompt = f"Based on the following literature review, suggest possible hypotheses: {literature}"
            print(f"[{self.name}] Generating hypotheses...")
            response = self.api_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            hypotheses = response.choices[0].message["content"]
            print(f"[{self.name}] Hypotheses generation completed.")
            return hypotheses
        except Exception as e:
            print(f"[{self.name}] Error in hypothesis generation: {e}")
            return None
