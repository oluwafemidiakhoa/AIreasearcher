import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

class DataAnalysisAgent:
    def __init__(self, name, role):
        self.name = name
        self.role = role

    def plot_time_series(self, df):
        df_yearly = df['AverageTemperature'].resample('Y').mean()
        plt.figure(figsize=(14, 7))
        plt.plot(df_yearly.index, df_yearly, label="Yearly Avg Temp")
        plt.fill_between(df_yearly.index,
                         df_yearly.rolling(window=5).mean() - 1.96 * df_yearly.rolling(window=5).std(),
                         df_yearly.rolling(window=5).mean() + 1.96 * df_yearly.rolling(window=5).std(),
                         alpha=0.2, label="95% Confidence Interval")
        plt.title("Yearly Global Average Temperature Trend with Confidence Interval")
        plt.xlabel("Year")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.show()

    def spectral_analysis(self, df):
        temperature_data = df['AverageTemperature'].dropna()
        temperature_fft = np.fft.fft(temperature_data)
        freqs = np.fft.fftfreq(len(temperature_data))

        plt.figure(figsize=(12, 6))
        plt.plot(freqs[:len(freqs) // 2], np.abs(temperature_fft[:len(freqs) // 2]))
        plt.title("Spectral Analysis of Temperature Anomalies")
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.show()

    def rolling_statistics(self, df):
        df_yearly = df['AverageTemperature'].resample('Y').mean()
        plt.figure(figsize=(14, 7))
        plt.plot(df_yearly.index, df_yearly, label="Yearly Avg Temp")
        plt.plot(df_yearly.index, df_yearly.rolling(window=5).mean(), label="5-Year Rolling Mean", linestyle="--")
        plt.plot(df_yearly.index, df_yearly.rolling(window=20).mean(), label="20-Year Rolling Mean", linestyle="--")
        plt.title("5-Year and 20-Year Rolling Mean of Global Temperature")
        plt.xlabel("Year")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.show()

    def hypothesis_testing(self, df):
        df_yearly = df['AverageTemperature'].resample('Y').mean().dropna()
        pre_1950 = df_yearly[df_yearly.index.year < 1950]
        post_1950 = df_yearly[df_yearly.index.year >= 1950]

        t_stat, p_value = ttest_ind(pre_1950, post_1950)
        print(f"T-Statistic: {t_stat}, P-Value: {p_value}")
        if p_value < 0.05:
            print("Significant difference between pre-1950 and post-1950 average temperatures.")
        else:
            print("No significant difference detected.")
