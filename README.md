# AI Researcher: A Multi-Agent System for Autonomous Scientific Research

**Author:** Oluwafemi Diakhoa

## Overview
The AI Researcher is a multi-agent system that automates various research tasks, from data collection and literature review to hypothesis generation and predictive modeling. It is built to streamline research workflows in domains that involve complex data analysis, such as climate science, medicine, and social sciences.

## Features
- **Data Retrieval and Preprocessing:** Automatic dataset download and cleanup.
- **Literature Review Generation:** Automated review synthesis using OpenAI.
- **Hypothesis Generation:** Data-driven hypothesis creation.
- **Experiment Design and Data Analysis:** Structured analysis and rolling statistics.
- **Predictive Modeling:** Trend forecasting using ARIMA and SARIMA models.
- **Documentation and Report Generation:** Auto-generated, publication-ready reports.

## Project Structure
- **notebooks/**: Main research notebook
- **src/**: Source code containing agent modules and utility functions
- **data/**: Directory for storing raw and processed datasets
- **tests/**: Contains test cases for different modules

## Setup and Installation
### Prerequisites
- Python 3.8 or above
- Kaggle API for data retrieval
- OpenAI API key

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/oluwafemidiakhoa/AIResearcher.git
   cd AIResearcher
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Set up your environment variables for the API keys:

bash
Copy code
export OPENAI_API_KEY="your_openai_key"
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_key"
Usage
Run the main.py script to start the research workflow:

bash
Copy code
python src/main.py
For details on each component, refer to the notebook.

Contributing
Feel free to contribute! See CONTRIBUTING.md for guidelines.

License
This project is licensed under the MIT License - see the LICENSE file for details.

csharp
Copy code

### 3. Add a `.gitignore` File

Add common file types that shouldn’t be tracked by Git in a `.gitignore` file:

Python
*.pyc pycache/ *.pkl *.h5

Notebooks
.ipynb_checkpoints/

Data files
data/raw/ data/processed/

Environment and IDE files
.env *.vscode/ .DS_Store

r
Copy code

### 4. Create `requirements.txt`

List all Python packages required to run your project. Generate it using:

```bash
pip freeze > requirements.txt
