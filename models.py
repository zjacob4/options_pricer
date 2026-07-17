import math
from scipy.stats import norm

def black_scholes(k,s,r,sigma,t,q,cp):

    if sigma == 0:
        if cp == "call":
            price = max(s*math.exp(-q*t) - k*math.exp(-r*t), 0)
        elif cp == "put":
            price = max(k*math.exp(-r*t) - s*math.exp(-q*t),0)
    elif t == 0:
        if cp == "call":
            price = max(s - k, 0)
        elif cp == "put":
            price = max(k - s, 0)
    else:        
        d1 = (math.log(s/k) + (r - q + sigma**2/2)*t) / (sigma * t**0.5)
        d2 = d1 - sigma*t**0.5
        if cp == "call":
            price = norm.cdf(d1)*s*math.exp(-q*t) - norm.cdf(d2)*k*math.exp(-r*t)
        elif cp == "put":
            price = norm.cdf(-d2)*k*math.exp(-r*t) - norm.cdf(-d1)*s*math.exp(-q*t)

    return price


def black_scholes_delta(k,s,r,sigma,t,q,cp):

    if cp == "call":
        if sigma == 0:
            delta = math.exp(-q*t)
        elif t == 0:
            delta = 1
        else:
            d1 = (math.log(s/k) + (r - q + sigma**2/2)*t) / (sigma * t**0.5)
            delta = math.exp(-q*t)*norm.cdf(d1)
    else:
        if sigma == 0:
            delta = 0
        elif t == 0:
            delta = 0
        else:
            d1 = (math.log(s/k) + (r - q + sigma**2/2)*t) / (sigma * t**0.5)
            delta = math.exp(-q*t)*(norm.cdf(d1) - 1)

    return delta