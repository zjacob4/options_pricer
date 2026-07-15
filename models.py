import math
from scipy.stats import norm

def black_scholes(k,s,r,sigma,t,q,cp):

    if sigma == 0:
        sigma = 0.000001
    if t == 0:
        t = 0.000001


    if sigma == 0:
        if cp == "call":
            price = s*math.exp(-q*t) - k*math.exp(-r*t)
        elif cp == "put":
            price = k*math.exp(-r*t)*norm.cdf(-d2) - s*norm.cdf(-d1)
    elif t == 0:
        if cp == "call":
            price = max(s - k, 0)
        elif cp == "put":
            price = (k - s, 0)
    else:
        
        d1 = (math.log(s/k) + (r - q + sigma**2/2)*t) / (sigma * t**0.5)
        d2 = d1 - sigma*t**0.5
        if cp == "call":
            price = norm.cdf(d1)*s*math.exp(-q*t) - norm.cdf(d2)*k*math.exp(-r*t)
        elif cp == "put":
            price = k*math.exp(-r*t)*norm.cdf(-d2) - s*norm.cdf(-d1)

    return price