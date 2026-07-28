import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date, timedelta
from models import black_scholes, binomial_crr
import requests
from ratelimit import limits, sleep_and_retry
import math

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

            # add variables for review
            yoc["s"] = s
            yoc["q"] = q
            yoc["r"] = r
            yoc["t"] = t

            # check that no option is less than its intrinsic value
            yoc = check_intrinsic_value(yoc,s,cp,q,t,r)

            # add column on price stalenss
            yoc = check_stale_price(yoc)

            # add a column on break taxonomy
            yoc = break_taxonomy(yoc)

            options_report = yoc.to_csv(index=False).encode('utf-8')

        st.success("Options Report Created Successfully.")
        st.download_button(
            label="Download data as CSV",
            data = options_report,
            file_name="options_report.csv",
            mime="text/csv",
            key="download_csv"
        )


def break_taxonomy(df):
    price = df["price"]
    valid = price > 0
    bs_dev  = ((df["price"] - df["bs_price"]).abs() / price).where(valid)
    bin_dev = ((df["price"] - df["binomial_price"]).abs() / price).where(valid)

    conditions = [
        ~valid,
        bs_dev  > 0.2,
        bs_dev  > 0.1,
        bin_dev > 0.2,
        bin_dev > 0.1,
    ]
    choices = [
        "No Price",
        "Large Break",
        "Small Break",
        "Large Break",
        "Small Break",
    ]

    df["break"] = np.select(conditions, choices, default="Within Tolerance")
    df["break"] = np.where(df["stale"], "Stale", df["break"])
    return df

def check_stale_price(df):
    df["stale"] = (df["bid"] == 0) | ((df["ask"] - df["bid"]) > df["price"]*0.5)
    return df


def check_intrinsic_value(df,s,cp,q,t,r):
    if cp == "call":
        df["intrinsic_value"] = np.maximum(s*math.exp(-q*t) - df["strike"]*math.exp(-r*t), 0)
    else:
        df["intrinsic_value"] = np.maximum(df["strike"]*math.exp(-r*t) - s*math.exp(-q*t), 0)

    tol = 1e-6  # relative
    df["model_valid"] = (
    (df["bs_price"] >= df["intrinsic_value"] * (1 - tol) - 0.01) &
    (df["binomial_price"] >= df["intrinsic_value"] * (1 - tol) - 0.01))

    if not df["model_valid"].all():
        bad = df[~df["model_valid"]]
        print(bad[["strike","bs_price","binomial_price","intrinsic_value","s","q","r","t"]].to_string())
        n_bad = (~df["model_valid"]).sum()
        st.error(f'{n_bad} contracts are invalid in pricing (contract value less than current intrinsic value).')
        st.stop()
    
    return df

def create_binomial_option_chain(k_list,s,r,sigma,t,q,cp,n,o_type):    
    price_list = []
    for k in k_list:
        price_list.append(binomial_crr(k,s,r,sigma,t,q,cp,n,o_type))
    return price_list

def create_bs_option_chain(k_list,s,r,sigma,t,q,cp):
    price_list = []
    for k in k_list:
        price = black_scholes(k,s,r,sigma,t,q,cp)
        price_list.append(price)
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
    if sigma < 0.05 or sigma > 3.0:
        st.error(f'Sigma {sigma} is outside of expected range.')
        st.stop()
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
    q = ticker_init.info.get("dividendYield",0) / 100
    if q == None:
        q = 0 # handling no dividend for calculation
    if q > 0.2:
        st.error("Dividend yield is greater than 100%")
        st.stop()
    if q < 0:
        st.error("Dividend yield is negative")
        st.stop()
    
    return q

if __name__ == "__main__":
    homepage()