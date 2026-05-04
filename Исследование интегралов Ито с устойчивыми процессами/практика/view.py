import yfinance as yf

data = yf.download("BTC-USD", start="2019-01-01", end="2024-01-01", progress=False)

file_name = "bitcoin_full_data.csv"
data.to_csv(file_name)