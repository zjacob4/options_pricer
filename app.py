import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date, timedelta
from models import black_scholes
import requests
from ratelimit import limits, sleep_and_retry

CALLS = 10
RATE_LIMIT = 60

@sleep_and_retry
@limits(calls=CALLS,period=RATE_LIMIT)
def homepage():

    # page text
    st.title("Options Pricer")
    st.write("Welcome to the homepage")

    with st.form("options_form"):

        # store user input variables
        ticker = st.text_input("Underlying Ticker", key="ticker")

        # enter strike price
        k = st.text_input("Strike Price",key="k")
        
        # enter option type
        cp = st.text_input("Call or Put?",key="cp")

        # get expiration date
        expiry = st.text_input("Expiry", placeholder="mm/dd/yyyy",key="expiry")

        submitted = st.form_submit_button("Calculate Options Price",key="submit_options_form")

    if submitted:

        with st.spinner("Calculating..."):
            ticker_init = yf.Ticker(ticker.upper()) # initialize ticker in yf

            k = float(k) # convert string strike price to float

            # get underlying stock price from yfinance api
            s = ticker_init.history(period="1d")["Close"].iloc[-1]
            price_history = ticker_init.history(period="1y")["Close"]

            cp = cp.lower() # standardize option type text format

            # get annualized historical volatiliy
            sigma = get_sigma(price_history)

            # get time to expiry
            t = get_t(expiry)

            # get dividend yield
            q = get_q(ticker_init)

            # handle negative time
            if t < 0:
                raise ValueError("Time to expiry cannot be negative.")

            # get risk-free rate
            if t < 0.33:
                treasury_ticker = "^IRX"   # 3-Month Treasury Bill
            elif t < 5:
                treasury_ticker = "^FVX"    # 5-Year Treasury Note
            elif t < 10:
                treasury_ticker = "^TNX"   # 10-Year Treasury Note
            else:
                treasury_ticker = "^TYX"    # 30-Year Treasury Bond
            r = yf.Ticker(treasury_ticker).info.get("previousClose") / 100

            # run black-scholes with inputted variables
            options_price = black_scholes(k,s,r,sigma,t,q,cp)
        st.success(options_price)

def get_sigma(price_history):
    log_return = np.log(price_history / price_history.shift(1)) # create new column documenting log returns
    daily_vol = log_return.std() # daily standard deviation
    sigma = daily_vol * np.sqrt(252) # annualized volatility
    return sigma

def get_t(expiry):
    today = date.today()
    [month,day,year] = expiry.split("/")
    t_days = (date(int(year),int(month),int(day)) - today) / timedelta(days=1)
    t = t_days / 365
    return t

def get_q(ticker_init):
    
    # get dividend yield
    q = ticker_init.info.get("dividendYield",0)
    if q == None:
        q = 0 # handling no dividend for calculation
    if q > 1:
        raise ValueError("Dividend greater than 100%")
    
    return q

if __name__ == "__main__":
    homepage()