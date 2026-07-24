import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date, timedelta
from models import black_scholes, binomial_crr
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

            # get dividend yield
            q = get_q(ticker_init)

            # get yfinance options chain to compare
            yoc, new_expiry = get_yf_chain(ticker_init,expiry,cp)
            yoc = yoc[["strike","bid","ask"]]
            yoc["price"] = (yoc["bid"] + yoc["ask"]) / 2

            # get time to expiry using new_expiry, closest date in options chain
            t = get_t(new_expiry)

            # get risk-free rate
            r = get_r(t)

            # create black_scholes option chain based on yf chain strike prices
            yoc["bs_price"] = create_bs_option_chain(yoc["strike"],s,r,sigma,t,q,cp)
            
            # create binomial crr option chain based on yf chain strike prices
            yoc["binomial_price"] = create_binomial_option_chain(yoc["strike"],s,r,sigma,t,q,cp,n=100,o_type="american") # edit later, binomial convergence > 100

            options_report = yoc.to_csv(index=False).encode('utf-8')

        st.success("Options Report Created Successfully.")
        st.download_button(
            label="Download data as CSV",
            data = options_report,
            file_name="Options report",
            mime="text/csv",
            key="download_csv"
        )


def create_binomial_option_chain(k_list,s,r,sigma,t,q,cp,n,o_type):    
    price_list = []
    for k in k_list:
        price_list.append(binomial_crr(k,s,r,sigma,t,q,cp,n,o_type))
    return price_list

def create_bs_option_chain(k_list,s,r,sigma,t,q,cp):
    price_list = []
    for k in k_list:
        price_list.append(black_scholes(k,s,r,sigma,t,q,cp))
    return price_list

def get_yf_chain(ticker_init,expiry,cp):
    df = pd.DataFrame()
    df["expiry_dates"] = pd.to_datetime(ticker_init.options)
    target_expiry = pd.to_datetime(expiry)
    nearest_row_idx = (df["expiry_dates"] - target_expiry).abs().idxmin()
    nearest_date = str(df.loc[nearest_row_idx,"expiry_dates"].date())
    if cp == "call":
        return ticker_init.option_chain(nearest_date).calls, nearest_date # get options chain for the specified expiry date
    else:
        return ticker_init.option_chain(nearest_date).puts, nearest_date # get options chain for the specified expiry date

def get_r(t):
    if t < 0.33:
        treasury_ticker = "^IRX"   # 3-Month Treasury Bill
    elif t < 5:
        treasury_ticker = "^FVX"    # 5-Year Treasury Note
    elif t < 10:
        treasury_ticker = "^TNX"   # 10-Year Treasury Note
    else:
        treasury_ticker = "^TYX"    # 30-Year Treasury Bond
    r = yf.Ticker(treasury_ticker).info.get("previousClose") / 100

    return r

def get_sigma(price_history):
    log_return = np.log(price_history / price_history.shift(1)) # create new column documenting log returns
    daily_vol = log_return.std() # daily standard deviation
    sigma = daily_vol * np.sqrt(252) # annualized volatility
    return sigma

def get_t(expiry):
    today = date.today()
    [year,month,day] = expiry.split("-")
    t_days = (date(int(year),int(month),int(day)) - today) / timedelta(days=1)
    t = t_days / 365

    # handle negative time
    if t < 0:
        raise ValueError("Time to expiry cannot be negative.")

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