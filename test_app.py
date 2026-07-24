from streamlit.testing.v1 import AppTest
from datetime import date, timedelta
from app import get_q, get_sigma
import yfinance as yf
import pytest
import numpy as np

def test_negative_t():
    
    # Initialize streamlit app
    at = AppTest.from_file("app.py")
    at.run(timeout=30)

    # set inputs
    at.text_input(key="ticker").set_value("aapl")
    at.text_input(key="k").set_value("200")
    at.text_input(key="cp").set_value("call")
    at.text_input(key="expiry").set_value("05/22/2026") # set a date in the past for negative t

    # click the submit button with new inputs
    at.button(key="submit_options_form").click()

    # run the app to process submission
    at.run(timeout=30)

    # assert that an exception was raised
    assert len(at.exception) > 0, "No exception was raised for negative time to expiry."

    # assert that no additional exceptions were raised
    assert len(at.exception) == 1, "More than one exception was raised."

    # extract and inspection the exception object
    app_exception = at.exception[0]
    
    # confirm it's a negative time to expiry error
    assert "Time to expiry cannot be negative" in app_exception.value

def test_q_from_app():

    ticker_init = yf.Ticker("AAPL")

    q = get_q(ticker_init)

    assert q == pytest.approx(0.33, abs=0.1), f"Dividend is off, expected 0.33, got {q}"

def test_sigma_from_app():

    ticker_init = yf.Ticker("AAPL") # initialize ticker in yf
    price_history = ticker_init.history(period="1y")["Close"]
    sigma = get_sigma(price_history)
    
    assert np.isfinite(sigma) and sigma > 0 and sigma < 3.0, f"Computed volatility looks wrong ({sigma}). Can't price reliably."