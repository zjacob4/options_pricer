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

def binomial_crr(k,s,r,sigma,t,q,cp,n,o_type):

    dt = t/n # time left divided by number of intervals/steps
    u = math.exp(sigma*dt**0.5) # up multiplier
    d = 1 / u # down multiplier

    p = (math.exp((r-q)*dt) - d) / (u - d) # up probability
    discount = math.exp(-r*dt) # time discount

    option_values = [0.0] *(n+1) # array holding option values at termination, step_number = n

    # calculations at expiration
    for j in range(n+1):

        s_end = s * (u**(n-j)) * (d**j) # calculate terminal stock price

        # calculate payoff at expiration
        if cp == "call":
            option_values[j] = max(s_end - k, 0.0)
        else:
            option_values[j] = max(k - s_end,0.0)

    # dynamic programming, work backwards to calculate the options price at each n node
    for i in range(n - 1, -1, -1): # working up the tree from the final layer (expiration) back to current
        for j in range(i+1): # working through each node in the layer of len(layer), so layer 1 has 1 node, layer 2 has 2 nodes, layer 3 has 3 because middle nodes converge

            # calculate the stock price for the underlying node
            s_current = s*(u**(i-j)) * (d**j) # calculate stock price at current node

            # exercise value is straightforward for immediate exercise, difference between strike and expected price
            if cp == "call":
                exercise_current = max(s_current - k, 0.0)
            else:
                exercise_current = max(k - s_current, 0.0)

            hold_value = discount * (p*option_values[j] + (1-p)*option_values[j+1]) # discounted forward value of upper prob * upper outcome from forward layer and lower prob * lower outcome from forward layer

            # update this layer's array
            if o_type.lower() == "american":
                option_values[j] = max(hold_value,exercise_current)
            else:
                option_values[j] = hold_value

    return option_values[0] # first element once this is done is the current option value

