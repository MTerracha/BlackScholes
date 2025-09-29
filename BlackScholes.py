# Black-Scholes Terminal (Clean Pro Version)
# With dividend yield q and IV in %

import math
from scipy.stats import norm
from scipy.optimize import brentq

# ---- Colors -----------------------------------------------------------------
try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
except Exception:
    class _D: 
        def __getattr__(self, _): return ""
    Fore = Style = _D()

TITLE = Style.BRIGHT + Fore.YELLOW
LABEL = Style.BRIGHT + Fore.WHITE
VALUE = Style.BRIGHT + Fore.GREEN
SUB   = Fore.CYAN
OK    = Style.BRIGHT + Fore.GREEN
ERR   = Style.BRIGHT + Fore.RED
RST   = Style.RESET_ALL

def line(w=72): return "─" * w
def box(title: str, width=72):
    top = "┌" + ("─" * (width - 2)) + "┐"
    mid = f"│{title.center(width-2)}│"
    bot = "└" + ("─" * (width - 2)) + "┘"
    return top, mid, bot

def f4(x): return f"{x:,.4f}"
def f6(x): return f"{x:,.6f}"
def money(x): return f"${x:,.2f}"

def ask_float(prompt, default=None):
    s = input(prompt).strip()
    if s == "":
        if default is None:
            raise ValueError("Missing required numeric input.")
        return float(default)
    return float(s)

# ---- Black–Scholes with dividend yield q ------------------------------------
def d1_d2(S, K, T, r, q, vol):
    sqrtT = math.sqrt(T)
    d1 = (math.log(S/K) + (r - q + 0.5*vol*vol)*T) / (vol*sqrtT)
    d2 = d1 - vol*sqrtT
    return d1, d2, sqrtT

def bs_price(S, K, T, r, q, vol, opt_type="call"):
    d1, d2, _ = d1_d2(S, K, T, r, q, vol)
    disc_r = math.exp(-r*T); disc_q = math.exp(-q*T)
    if opt_type.lower().startswith("c"):
        return S*disc_q*norm.cdf(d1) - K*disc_r*norm.cdf(d2)
    else:
        return K*disc_r*norm.cdf(-d2) - S*disc_q*norm.cdf(-d1)

def greeks(S, K, T, r, q, vol):
    d1, d2, sqrtT = d1_d2(S, K, T, r, q, vol)
    disc_r = math.exp(-r*T); disc_q = math.exp(-q*T)
    pdf = norm.pdf(d1)

    call = S*disc_q*norm.cdf(d1) - K*disc_r*norm.cdf(d2)
    put  = K*disc_r*norm.cdf(-d2) - S*disc_q*norm.cdf(-d1)

    delta_c = disc_q*norm.cdf(d1)
    delta_p = disc_q*(norm.cdf(d1) - 1.0)
    gamma   = disc_q*pdf/(S*vol*sqrtT)
    vega    = S*disc_q*pdf*sqrtT
    theta_c = -(S*disc_q*pdf*vol)/(2*sqrtT) - r*K*disc_r*norm.cdf(d2) + q*S*disc_q*norm.cdf(d1)
    theta_p = -(S*disc_q*pdf*vol)/(2*sqrtT) + r*K*disc_r*norm.cdf(-d2) - q*S*disc_q*norm.cdf(-d1)
    rho_c   =  K*T*disc_r*norm.cdf(d2)
    rho_p   = -K*T*disc_r*norm.cdf(-d2)

    return (call, put, d1, d2,
            delta_c, delta_p, gamma, vega, theta_c, theta_p, rho_c, rho_p)

def implied_vol(price_mkt, S, K, T, r, q, opt_type="call"):
    disc_r = math.exp(-r*T); disc_q = math.exp(-q*T)
    intrinsic = max(0.0, S*disc_q - K*disc_r) if opt_type[0].lower()=="c" else max(0.0, K*disc_r - S*disc_q)
    if price_mkt < intrinsic:
        return float("nan")
    f = lambda sig: bs_price(S, K, T, r, q, sig, opt_type) - price_mkt
    try:
        return brentq(f, 1e-6, 5.0, maxiter=200, xtol=1e-10)
    except Exception:
        return float("nan")

# ---- Header -----------------------------------------------------------------
top, mid, bot = box(" BLACK–SCHOLES OPTION PRICING MODEL ", 72)
print(TITLE + top + RST)
print(TITLE + mid + RST)
print(TITLE + bot + RST)

# ---- Inputs -----------------------------------------------------------------
try:
    S      = ask_float(LABEL + "Underlying price (S): " + RST)
    K      = ask_float(LABEL + "Strike price (K): " + RST)
    T_days = ask_float(LABEL + "Time to expiration (days): " + RST)
    r      = ask_float(LABEL + "Risk-free rate r (decimal): " + RST)
    vol    = ask_float(LABEL + "Volatility σ (decimal): " + RST)
    q      = ask_float(LABEL + "Dividend yield q (decimal) [default 0]: " + RST, default=0.0)
    m_raw  = input(LABEL + "Market price for IV (blank to skip): " + RST).strip()
    mkt_price = float(m_raw) if m_raw else None
except Exception as e:
    print(ERR + f"Input error: {e}" + RST)
    raise SystemExit(1)

if S<=0 or K<=0 or T_days<=0 or vol<=0:
    print(ERR + "Inputs must be positive; T_days and σ > 0." + RST)
    raise SystemExit(1)

T = T_days/365.0

# ---- Compute ----------------------------------------------------------------
(C, P, d1, d2,
 Dc, Dp, Gm, Vg, Tc, Tp, Rc, Rp) = greeks(S, K, T, r, q, vol)

# ---- RESULTS ----------------------------------------------------------------
print()
print(SUB + line() + RST)
print(TITLE + " RESULTS ".center(72) + RST)
print(SUB + line() + RST)

print(f"{LABEL}{'Moneyness factor (d1)':<28}{VALUE}{f4(d1):>12}{RST}")
print(f"{LABEL}{'Risk-adjusted moneyness (d2)':<28}{VALUE}{f4(d2):>12}{RST}")
print(f"{LABEL}{'Call Price':<28}{VALUE}{money(C):>12}{RST}")
print(f"{LABEL}{'Put Price':<28}{VALUE}{money(P):>12}{RST}")

# ---- GREEKS -----------------------------------------------------------------
print()
print(SUB + line() + RST)
print(TITLE + " GREEKS ".center(72) + RST)
print(SUB + line() + RST)

print(f"{LABEL}{'Greek':<14}{'Call':>18}{'Put':>18}{RST}")
print("-"*72)
print(f"{LABEL}{'Delta':<14}{VALUE}{f4(Dc):>18}{VALUE}{f4(Dp):>18}{RST}")
print(f"{LABEL}{'Gamma':<14}{VALUE}{f6(Gm):>18}{'':>18}{RST}")
print(f"{LABEL}{'Vega':<14}{VALUE}{f4(Vg):>18}{'':>18}{RST}")
print(f"{LABEL}{'Theta':<14}{VALUE}{f4(Tc):>18}{VALUE}{f4(Tp):>18}{RST}")
print(f"{LABEL}{'Rho':<14}{VALUE}{f4(Rc):>18}{VALUE}{f4(Rp):>18}{RST}")

# ---- IMPLIED VOL (optional) -------------------------------------------------
if mkt_price is not None:
    iv_c = implied_vol(mkt_price, S, K, T, r, q, "call")
    iv_p = implied_vol(mkt_price, S, K, T, r, q, "put")
    print()
    print(SUB + line() + RST)
    print(TITLE + " IMPLIED VOLATILITY ".center(72) + RST)
    print(SUB + line() + RST)
    print(f"{LABEL}{'Market Price':<28}{VALUE}{money(mkt_price)}{RST}")
    if not math.isnan(iv_c):
        print(f"{LABEL}{'IV (Call)':<28}{VALUE}{iv_c*100:,.2f}%{RST}")
    else:
        print(f"{LABEL}{'IV (Call)':<28}{VALUE}NaN{RST}")
    if not math.isnan(iv_p):
        print(f"{LABEL}{'IV (Put)':<28}{VALUE}{iv_p*100:,.2f}%{RST}")
    else:
        print(f"{LABEL}{'IV (Put)':<28}{VALUE}NaN{RST}")

print()
print(OK + "Done." + RST)
