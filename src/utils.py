import os
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from openai import OpenAI

load_dotenv()  # Load environment variables from .env file

def get_kaggle_client():
    api = KaggleApi()
    api.authenticate()
    return api

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set it in .env file.")
    return OpenAI(api_key=api_key)
