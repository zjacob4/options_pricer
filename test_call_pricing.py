import math
import pytest

from models import black_scholes, black_scholes_delta

def test_pc_parity():
    # textbook test inputs for c = 4.76    
    k = 40 # strike price
    s = 42 # stock price
    r = 0.1 # risk-free interest rate
    sigma = 0.2 # volatility of underlying stock
    t = 0.5 # time to expiry
    q = 0.0 # no dividend for continuous dividend yield
    
    expected_diff = s*math.exp(-q*t) - k*math.exp(-r*t)

    actual_diff = black_scholes(k,s,r,sigma,t,q,"call") - black_scholes(k,s,r,sigma,t,q,"put")

    assert actual_diff == pytest.approx(expected_diff)


def test_textbook_bs():

    # textbook test inputs for c = 4.76    
    k = 40 # strike price
    s = 42 # stock price
    r = 0.1 # risk-free interest rate
    sigma = 0.2 # volatility of underlying stock
    t = 0.5 # time to expiry
    q = 0.0 # no dividend for continuous dividend yield
    cp = "call"

    actual_call_price = black_scholes(k,s,r,sigma,t,q,cp)

    assert actual_call_price == pytest.approx(4.76, abs=0.005), f"Call BS w/ textbook inputs did not yield the expected 4.76, yielded{actual_call_price}."


def test_zero_vol_bs():

    # inputs for zero volatility test
    k = 40 # strike price
    s = 42 # stock price
    r = 0.1 # risk-free interest rate
    sigma = 0.0 # volatility of underlying stock
    t = 0.5 # time to expiry
    q = 0.0 # no continuous dividend yield
    cp = "call"

    # ground truth, zero vol should collapse to discounted forward intrinsic value
    expected_call_price = max(s*math.exp(-q*t) - k*math.exp(-r*t),0)

    # assert whether the calculated zero vol call price equals the discounted forward intrinsic value
    actual_call_price = black_scholes(k,s,r,sigma,t,q,cp)

    assert actual_call_price == pytest.approx(expected_call_price), f"Zero volatility call price doesn't collapse to discounted forward instrinsic value. Expected {expected_call_price}, but got {actual_call_price}."
    

def test_zero_t_bs():

    # inputs for zero volatility test
    k = 40 # strike price
    s = 42 # stock price
    r = 0.1 # risk-free interest rate
    sigma = 0.1 # volatility of underlying stock
    t = 0.0 # time to expiry is zero to test
    q = 0.0 # no continuous dividend yield
    cp = "call"

    # ground truth, zero time to expiry should collapse to the instrinsic value
    expected_call_price = max(s - k,0)

    # assert whether the calculated zero time call price equals the intrinsic value of the derivative
    actual_call_price = black_scholes(k,s,r,sigma,t,q,cp)

    assert expected_call_price == pytest.approx(actual_call_price, abs=0.001), f"Zero time-to-expiry call price doesn't collapse to instrinsic option value. Expected {expected_call_price}, but got {actual_call_price}."
    

def test_deep_itm_bs():

    # inputs for deep itm test
    k = 50 # strike price deep in the money
    s = 100 # stock price
    r = 0.1 # risk-free interest rate
    sigma = 0.1 # volatility of underlying stock, low to test N drift
    t = 0.5 # time to expiry, low to test N drift
    q = 0.0 # no continuous dividend yield
    cp = "call"

    # ground truth call price should be N approaching 1 for deep ITM options, as the certainty to exercise is high
    expected_call_price = s*math.exp(-q*t) - k*math.exp(-r*t)

    # assert that the calculated call price matches N approaching 1
    actual_call_price = black_scholes(k,s,r,sigma,t,q,cp)

    assert actual_call_price == pytest.approx(expected_call_price, abs=0.001), f"N should approach 1 as a contract becomes deep in-the-money. Expected a call price of {expected_call_price}, but got {actual_call_price}."


def test_deep_otm_bs():

    # inputs for deep itm test
    k = 200 # strike price deep out of the money
    s = 100 # stock price
    r = 0.1 # risk-free interest rate
    sigma = 0.1 # volatility of underlying stock, low to test N drift
    t = 0.5 # time to expiry, low to test N drift
    q = 0.0 # no continuous dividend yield
    cp = "call"

    # assert that the calculated call price matches N approaching 0, for a deep OTM option that will likely not be exercised
    actual_call_price = black_scholes(k,s,r,sigma,t,q,cp)

    assert actual_call_price < 0.01, f"N should approach 0 as contract becomes deep out-of-the-money, but call price is {actual_call_price}."
    

def test_delta_bs():
    
    k = 40 # strike price
    s = 42 # stock price
    r = 0.1 # risk-free interest rate
    sigma = 0.2 # volatility of underlying stock, low to test N drift
    t = 0.5 # time to expiry, low to test N drift
    q = 0.0 # no continuous dividend yield
    cp = "call"
    eps = 0.01

    expected_delta = (black_scholes(k,s+eps,r,sigma,t,q,cp) - black_scholes(k,s-eps,r,sigma,t,q,cp)) / (2*eps)

    actual_delta = black_scholes_delta(k,s,r,sigma,t,q,cp)

    assert actual_delta == pytest.approx(expected_delta,abs=1e-4), f"Delta test failed, expected {expected_delta}, but got {actual_delta}."


def test_binomial_convergence():
    pytest.skip("No binomial convergence tests created.")
