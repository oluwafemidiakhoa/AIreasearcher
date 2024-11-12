import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

class DataRetrievalAgent:
    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.api = KaggleApi()
        self.api.authenticate()

    def retrieve_data(self, dataset, file_name):
        try:
            print(f"[{self.name}] Downloading dataset {dataset}...")
            self.api.dataset_download_files(dataset, path="data/raw", unzip=True)
            data_path = os.path.join("data/raw", file_name)
            df = pd.read_csv(data_path)
            print(f"[{self.name}] Data downloaded and loaded successfully.")
            return df
        except Exception as e:
            print(f"[{self.name}] Error in data retrieval: {e}")
            return None

    def preprocess_data(self, df):
        try:
            print(f"[{self.name}] Preprocessing data...")
            df['dt'] = pd.to_datetime(df['dt'], errors='coerce')
            df = df.dropna(subset=['dt', 'AverageTemperature']).set_index('dt')
            df['AverageTemperature'] = df['AverageTemperature'].astype(float)
            print(f"[{self.name}] Data preprocessing complete.")
            return df
        except Exception as e:
            print(f"[{self.name}] Error in preprocessing: {e}")
            return None
