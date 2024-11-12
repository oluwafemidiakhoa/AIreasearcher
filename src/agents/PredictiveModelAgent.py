import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

class PredictiveModelAgent:
    def __init__(self, name, role):
        self.name = name
        self.role = role

    def arima_forecast(self, df, steps=20):
        try:
            df_yearly = df['AverageTemperature'].resample('Y').mean().dropna()
            arima_model = ARIMA(df_yearly, order=(5, 1, 0))
            arima_fit = arima_model.fit()
            forecast = arima_fit.forecast(steps=steps)
            print(f"[{self.name}] ARIMA forecasting completed.")
            return forecast
        except Exception as e:
            print(f"[{self.name}] Error in ARIMA forecasting: {e}")
            return None

    def sarima_forecast(self, df, steps=20):
        try:
            df_yearly = df['AverageTemperature'].resample('Y').mean().dropna()
            sarima_model = SARIMAX(df_yearly, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            sarima_fit = sarima_model.fit()
            forecast = sarima_fit.get_forecast(steps=steps).predicted_mean
            print(f"[{self.name}] SARIMA forecasting completed.")
            return forecast
        except Exception as e:
            print(f"[{self.name}] Error in SARIMA forecasting: {e}")
            return None

    def plot_forecasts(self, df, arima_forecast, sarima_forecast):
        try:
            df_yearly = df['AverageTemperature'].resample('Y').mean().dropna()
            plt.figure(figsize=(14, 7))
            plt.plot(df_yearly.index, df_yearly, label="Observed")
            plt.plot(pd.date_range(df_yearly.index[-1], periods=len(arima_forecast), freq='Y'), 
                     arima_forecast, label="ARIMA Forecast", linestyle="--")
            plt.plot(pd.date_range(df_yearly.index[-1], periods=len(sarima_forecast), freq='Y'), 
                     sarima_forecast, label="SARIMA Forecast", linestyle="--")
            plt.title("Temperature Forecast: ARIMA vs SARIMA")
            plt.xlabel("Year")
            plt.ylabel("Temperature (°C)")
            plt.legend()
            plt.show()
            print(f"[{self.name}] Forecast plot generated.")
        except Exception as e:
            print(f"[{self.name}] Error in plotting forecasts: {e}")
