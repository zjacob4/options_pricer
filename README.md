# Options Pricer

A Python library for pricing European and American options, with an emphasis on
correctness verified against textbook values, boundary conditions, and cross-model
convergence.

## Models

All implemented in [models.py](models.py):

### Black–Scholes (`black_scholes`)
Closed-form pricing for European calls and puts, with continuous dividend yield
support. Handles the degenerate cases cleanly:
- **Zero volatility** collapses to the discounted forward intrinsic value.
- **Zero time-to-expiry** collapses to the intrinsic value.

### Black–Scholes Delta (`black_scholes_delta`)
Analytic delta (∂price/∂spot) for calls and puts, including the zero-vol and
zero-time edge cases.

### Cox-Ross-Rubinstein Binomial Tree (`binomial_crr`)
Lattice-based pricing supporting both **European** and **American** exercise. Builds
terminal payoffs and works backward via dynamic programming, comparing hold value
against immediate exercise value at each node for American-style contracts. As the
number of steps `n` grows, prices converge to Black–Scholes.

## Parameters

| Symbol  | Meaning                                            |
|---------|----------------------------------------------------|
| `k`     | Strike price                                       |
| `s`     | Spot (underlying) price                            |
| `r`     | Risk-free interest rate                            |
| `sigma` | Volatility of the underlying                       |
| `t`     | Time to expiry (in years)                          |
| `q`     | Continuous dividend yield                          |
| `cp`    | `"call"` or `"put"`                                |
| `n`     | Number of steps (binomial only)                    |
| `o_type`| `"european"` or `"american"` (binomial only)       |

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Only dependency is `scipy` (for the normal CDF).

## Usage

```python
from models import black_scholes, black_scholes_delta, binomial_crr

# European call via Black–Scholes
price = black_scholes(k=40, s=42, r=0.1, sigma=0.2, t=0.5, q=0.0, cp="call")

# Its delta
delta = black_scholes_delta(k=40, s=42, r=0.1, sigma=0.2, t=0.5, q=0.0, cp="call")

# American put via a 500-step binomial tree
price = binomial_crr(k=100, s=100, r=0.05, sigma=0.2, t=1.0, q=0.0,
                     cp="put", n=500, o_type="american")
```

## Testing

The test suite in [test_master.py](test_master.py) covers:

- **Textbook values** — Black–Scholes prices matched against known reference figures.
- **Put–call parity** — the call/put price difference equals the discounted forward.
- **Boundary conditions** — zero volatility and zero time-to-expiry collapse to intrinsic value.
- **Deep ITM / OTM** — N(d) approaches 1 and 0 respectively.
- **Delta accuracy** — analytic delta matched against a central finite difference.
- **Dividend monotonicity** — higher yields lower calls and raise puts.
- **Binomial hand-calcs** — 1- and 2-step trees matched to hand-computed values.
- **Convergence** — a 500-step binomial tree converges to Black–Scholes.

Run with:

```bash
pytest
```
