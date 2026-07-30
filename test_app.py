from streamlit.testing.v1 import AppTest
from datetime import date, timedelta
from app import get_q, get_sigma, check_stale_price, break_taxonomy
import yfinance as yf
import pytest
import numpy as np
import pandas as pd


def test_q_from_app():

    ticker_init = yf.Ticker("AAPL")

    q = get_q(ticker_init)

    assert q == pytest.approx(0.0033, abs=0.001), f"Dividend is off, expected 0.33, got {q}"

def test_sigma_from_app():

    ticker_init = yf.Ticker("AAPL") # initialize ticker in yf
    price_history = ticker_init.history(period="1y")["Close"]
    sigma = get_sigma(price_history)
    
    assert np.isfinite(sigma) and sigma > 0 and sigma < 3.0, f"Computed volatility looks wrong ({sigma}). Can't price reliably."

def test_stale_price():
    df = pd.DataFrame([{
        "strike": 30.0, "bid": 238.0, "ask": 243.0, "price": 240.5,
        "intrinsic_value": 310.2168868789183,  # market price 240.5 < intrinsic, must be stale
        "stale": False,  # WRONG on purpose; function must flip it to True
    }])
    df = check_stale_price(df)

    assert df["stale"].iloc[0] == True, "Expected Stale==True, got False"

    df = pd.DataFrame([{
    "strike": 340.0, "bid": 34.2, "ask": 35.65, "price": 34.925,
    "intrinsic_value": 8.83,  # market 34.9 > intrinsic, tight spread, nonzero bid,  valid
    "stale": True,  # WRONG on purpose; must flip to False
    }])

    df = check_stale_price(df)

    assert df["stale"].iloc[0] == False, f"Did not flip valid market price to 'False' from 'Stale'. Still shows {df['stale'].iloc[0]}"

def test_breaks():

    df = pd.DataFrame([{
        "price": 100.0, "bs_price": 102.0, "binomial_price": 102.0,  # 2% dev, under 10%
        "stale": False, "break": None,
    }])
    
    df = break_taxonomy(df)
    
    assert df.loc[0, "break"] == "Within Tolerance", f"Break should say 'Within Tolerance', instead got {df['break']}"


    df = pd.DataFrame([{
    "price": 100.0, "bs_price": 115.0, "binomial_price": 115.0,  # 15% dev: >10%, <20%
    "stale": False, "break": None,
    }])

    df = break_taxonomy(df)
    
    assert df.loc[0, "break"] == "Small Break", f"Break should say 'Small Break', instead got {df['break']}"

    df = pd.DataFrame([{
    "price": 100.0, "bs_price": 125.0, "binomial_price": 125.0,  # 25% dev: >20%
    "stale": False, "break": None,
    }])

    df = break_taxonomy(df)
    
    assert df.loc[0, "break"] == "Large Break", f"Break should say 'Large Break', instead got {df['break']}"

    df = pd.DataFrame([{
    "price": 240.5, "bs_price": 310.0, "binomial_price": 310.0,  # huge dev, but...
    "stale": True, "break": None,  # ...stale, so must be labeled "Stale" not "Large Break"
    }])

    df = break_taxonomy(df)
    
    assert df.loc[0, "break"] == "Stale", f"Break should say 'Large Break', instead got {df['break']}"