import math
import pytest
from scipy.stats import norm 

from models import black_scholes, black_scholes_delta, binomial_crr

@pytest.mark.parametrize("cp, expected", [("call",4.76),("put",0.8086)])
def test_textbook_bs(cp,expected):

    # textbook test inputs for c = 4.76    
    k = 40 # strike price
    s = 42 # stock price
    r = 0.1 # risk-free interest rate
    sigma = 0.2 # volatility of underlying stock
    t = 0.5 # time to expiry
    q = 0.0 # no dividend for continuous dividend yield

    actual_price = black_scholes(k,s,r,sigma,t,q,cp)

    assert actual_price == pytest.approx(expected, abs=0.005), f"Call BS w/ textbook inputs did not yield the expected {expected}, yielded{actual_price}."


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


@pytest.fixture(params=["call","put"]) 
def cp(request):
    return request.param # test for call and put, directionally

@pytest.fixture(params=[0,0.1])
def q(request):
    return request.param # test w/ and w/o dividends

@pytest.fixture
def r():
    return 0.04

@pytest.fixture
def sigma():
    return 0.1

@pytest.fixture
def t():
    return 2

@pytest.fixture
def k():
    return 40

@pytest.fixture()
def s():
    return 42

def test_zero_vol_bs(cp,q,r,t,s,k):

    # inputs for zero volatility test
    sigma = 0.0 # volatility of underlying stock

    # ground truth, zero vol should collapse to discounted forward intrinsic value
    if cp == "call":
        expected_price = max(s*math.exp(-q*t) - k*math.exp(-r*t),0)
    else:
        expected_price = max(k*math.exp(-r*t) - s*math.exp(-q*t),0)

    # assert whether the calculated zero vol call price equals the discounted forward intrinsic value
    actual_price = black_scholes(k,s,r,sigma,t,q,cp)

    assert actual_price == pytest.approx(expected_price), f"Zero volatility call price doesn't collapse to discounted forward instrinsic value. Expected {expected_price}, but got {actual_price}."
    
def test_zero_t_bs(cp,q,r,sigma,s,k):

    # inputs for zero volatility test
    t = 0.0 # time to expiry is zero to test

    # ground truth, zero time to expiry should collapse to the instrinsic value
    if cp == "call":
        expected_price = max(s - k,0)
    else:
        expected_price = max(k - s,0)

    # assert whether the calculated zero time call price equals the intrinsic value of the derivative
    actual_price = black_scholes(k,s,r,sigma,t,q,cp)

    assert expected_price == pytest.approx(actual_price, abs=0.001), f"Zero time-to-expiry call price doesn't collapse to instrinsic option value. Expected {expected_price}, but got {actual_price}."
    

def test_deep_itm_bs(cp,q,r,t,sigma):

    t = 0.5 # setting lower t to test deep itm convergence of N to 1

    if cp =="call":
        k = 50 # strike price deep in the money
        s = 100 # stock price
        # ground truth price should be N approaching 1 for deep ITM options, as the certainty to exercise is high
        expected_price = s*math.exp(-q*t) - k*math.exp(-r*t)
    else:
        k = 100 # strike price deep in the money
        s = 50 # stock price
        # ground truth price should be N approaching 1 for deep ITM options, as the certainty to exercise is high
        expected_price = k*math.exp(-r*t) - s*math.exp(-q*t)

    # assert that the calculated call price matches N approaching 1
    actual_price = black_scholes(k,s,r,sigma,t,q,cp)

    d1 = (math.log(s/k) + (r - q + sigma**2/2)*t) / (sigma*t**0.5)
    d2 = d1 - sigma*math.sqrt(t)
    # the N term that must ->1 for this test's premise to hold
    premise = norm.cdf(d2) if cp == "call" else norm.cdf(-d2)
    assert premise > 0.999, f"{cp} deep-ITM premise broken at q={q}, t={t}: N={premise}"

    assert actual_price == pytest.approx(expected_price, abs=0.001), f"N should approach 1 as a contract becomes deep in-the-money. Expected a call price of {expected_price}, but got {actual_price}."


def test_deep_otm_bs(cp,q,r,t,sigma):

    t = 0.5 # setting lower t to test deep itm convergence of N to 0

    if cp =="call":
        k = 100 # strike price deep in the money
        s = 50 # stock price
    else:
        k = 50 # strike price deep in the money
        s = 100 # stock price

    # assert that the calculated price matches N approaching 0, for a deep OTM option that will likely not be exercised
    actual_price = black_scholes(k,s,r,sigma,t,q,cp)

    assert actual_price < 0.01, f"N should approach 0 as contract becomes deep out-of-the-money, but price is {actual_price}."
    
def test_delta_bs(cp,q,r,t,sigma,s,k):
    
    eps = 0.01

    expected_delta = (black_scholes(k,s+eps,r,sigma,t,q,cp) - black_scholes(k,s-eps,r,sigma,t,q,cp)) / (2*eps)

    actual_delta = black_scholes_delta(k,s,r,sigma,t,q,cp)

    assert actual_delta == pytest.approx(expected_delta,abs=1e-4), f"Delta test failed, expected {expected_delta}, but got {actual_delta}."

def test_div_monotonicity(cp,r,t,sigma,s,k):
    # inputs for monotonicity test
    q1 = 0.1 # lower yield
    q2 = 0.2 # higher yield

    if cp == "call":
        lower_yield_call_price = black_scholes(k,s,r,sigma,t,q1,cp)
        higher_yield_call_price = black_scholes(k,s,r,sigma,t,q2,cp)

        assert higher_yield_call_price < lower_yield_call_price, "The call price for the higher dividend yield should've been lower, but wasn't."
    else:
        lower_yield_put_price = black_scholes(k,s,r,sigma,t,q1,cp)
        higher_yield_put_price = black_scholes(k,s,r,sigma,t,q2,cp)

        assert higher_yield_put_price > lower_yield_put_price, "The put price for the higher dividend yield should've been higher, but wasn't."

@pytest.mark.parametrize("n",[1,2])
def test_binomial_crr(n):
    s = 100 # keeps at the money
    k = 100
    r = 0.05
    q = 0.0
    sigma = 0.2
    t = 1.0
    cp = "call"
    o_type = "european"

    if n == 1:
        assert binomial_crr(k,s,r,sigma,t,q,cp,n,o_type) == pytest.approx(12.17, abs=0.01)
    elif n == 2:
        assert binomial_crr(k,s,r,sigma,t,q,cp,n,o_type) == pytest.approx(9.55, abs=0.01)

def test_binomial_convergence(cp,q,r,s,k,t,sigma):

    # use n = 500 to see if binomial crr converges on black scholes
    assert binomial_crr(k,s,r,sigma,t,q,cp,500,"european") == pytest.approx(black_scholes(k,s,r,sigma,t,q,cp), abs=0.01)
