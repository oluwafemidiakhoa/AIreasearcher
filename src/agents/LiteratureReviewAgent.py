from openai import OpenAI

class LiteratureReviewAgent:
    def __init__(self, name, role, api_key):
        self.name = name
        self.role = role
        self.api_client = OpenAI(api_key=api_key)

    def conduct_review(self, topic):
        try:
            prompt = f"Conduct a literature review on the topic: {topic}"
            print(f"[{self.name}] Starting literature review...")
            response = self.api_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            review_text = response.choices[0].message["content"]
            print(f"[{self.name}] Literature review completed.")
            return review_text
        except Exception as e:
            print(f"[{self.name}] Error in literature review: {e}")
            return None
