from models import black_scholes
import yfinance as yf

print(black_scholes(5, 339.45, 0.044, 0.32, 0.641, 0.044, "call"))
ticker_init = yf.Ticker("AAPL") # initialize ticker in yf
q = ticker_init.info.get("dividendYield",0)
print(f"Ticker yield: {q}")