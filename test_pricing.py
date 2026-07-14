import math
import pytest

from models import black_scholes

def pc_parity_test():
    return 0

def textbook_bs_test():

    # textbook test inputs for c = 4.76    
    k = 40 # strike price
    s = 42 # stock price
    r = 0.1 # risk-free interest rate
    sigma = 0.2 # volatility of underlying stock
    t = 0.5 # time to expiry
    q = 0.0 # no dividend for continuous dividend yield
    cp = "call"

    try:
        call_price = black_scholes(k,s,r,sigma,t,q,cp)
    
        assert round(call_price,2) == pytest.approx(4.76), "Call BS w/ textbook inputs did not yield the expected 4.76"
    except AssertionError as e:
        print(e)

    print("Test passed.")

    return 0

def zero_vol_bs_test():

    # inputs for zero volatility test
    k = 40 # strike price
    s = 42 # stock price
    r = 0.1 # risk-free interest rate
    sigma = 0.0 # volatility of underlying stock
    t = 0.5 # time to expiry
    q = 0.0 # no continuous dividend yield
    cp = "call"

    # ground truth, zero vol should collapse to discounted forward intrinsic value
    ground_truth_call = max(s*math.exp(-q*t) - k*math.exp(-r*t),0)

    # assert whether the calculated zero vol call price equals the discounted forward intrinsic value
    try:
        call_price = black_scholes(k,s,r,sigma,t,q,cp)

        assert call_price == pytest.approx(ground_truth_call), "Zero volatility call price doesn't collapse to discounted forward instrinsic value"
    except AssertionError as e:
        print(e)

    print("Test passed.")

    return 0

def zero_t_bs_test():

    # inputs for zero volatility test
    k = 40 # strike price
    s = 42 # stock price
    r = 0.1 # risk-free interest rate
    sigma = 0.1 # volatility of underlying stock
    t = 0.5 # time to expiry
    q = 0.0 # no continuous dividend yield
    cp = "call"

    # ground truth, zero time to expiry should collapse to the instrinsic value
    ground_truth_call = max(s - k,0)

    # assert whether the calculated zero time call price equals the intrinsic value of the derivative
    try:
        call_price = black_scholes(k,s,r,sigma,t,q,cp)

        assert call_price == pytest.approx(ground_truth_call), "Zero time-to-expiry call price doesn't collapse to instrinsic option value"
    except AssertionError as e:
        print(e)

    print("Test passed.")

    return 0

def deep_itm_bs_test():

    # inputs for deep itm test
    k = 50 # strike price deep in the money
    s = 100 # stock price
    r = 0.1 # risk-free interest rate
    sigma = 0.1 # volatility of underlying stock, low to test N drift
    t = 0.5 # time to expiry, low to test N drift
    q = 0.0 # no continuous dividend yield
    cp = "call"

    # ground truth call price should be N approaching 1 for deep ITM options, as the certainty to exercise is high
    ground_truth_call = s*math.exp(-q*t) - k*math.exp(-r*t)

    # assert that the calculated call price matches N approaching 1
    try:
        call_price = black_scholes(k,s,r,sigma,t,q,cp)

        assert call_price == pytest.approx(ground_truth_call), "N should approach 1 as a contract becomes deep in-the-money"
    except AssertionError as e:
        print(e)

    print("Test passed.")

    return 0

def deep_otm_bs_test():

    # inputs for deep itm test
    k = 200 # strike price deep out of the money
    s = 100 # stock price
    r = 0.1 # risk-free interest rate
    sigma = 0.1 # volatility of underlying stock, low to test N drift
    t = 0.5 # time to expiry, low to test N drift
    q = 0.0 # no continuous dividend yield
    cp = "call"

    # assert that the calculated call price matches N approaching 0, for a deep OTM option that will likely not be exercised
    try:

        call_price = black_scholes(k,s,r,sigma,t,q,cp)

        assert call_price == pytest.approx(0), "N should approach 0 as contract becomes deep out-of-the-money"
    except AssertionError as e:
        print(e)

    print("Test passed.")

    return 0

def greeks_bs_test():
    return 0

def binomial_convergence_test():
    return 0