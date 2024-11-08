import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import talib as ta
import vectorbt as vbt
# Define the ticker symbol and interval
ticker = "AAPL"

data = yf.download(ticker, interval="1d", start="2024-01-01", end="2024-10-01")
# Resample data to create K-line intervals, for example, every 5 days
data.columns = data.columns.droplevel("Ticker")
# backtest vectorbt


hammer = ta.CDLHAMMER(data.Open, data.High, data.Low, data.Close)
hanging_man = ta.CDLHANGINGMAN(data.Open, data.High, data.Low, data.Close)
buys = (hammer ==100)
sells = (hanging_man == -100)
pf = vbt.Portfolio.from_signals(data.Close, buys, sells)
print(pf.stats())

pf.plot().show()
