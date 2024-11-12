import unittest
from src.agents.DataAnalysisAgent import DataAnalysisAgent

class TestDataAnalysisAgent(unittest.TestCase):
    def setUp(self):
        self.agent = DataAnalysisAgent("Data Analysis Agent", "Performs data analysis")
    
    def test_rolling_statistics(self):
        # Add your test case here, e.g., pass a sample dataset and assert expected output
        pass

if __name__ == "__main__":
    unittest.main()
