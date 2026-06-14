"""
STAGE 0 — Financial data fetch (real share price data; NOT AI)
=================================================================

WHAT THIS DOES
--------------
Fetches recent share price history for each FTSE100 company in the
dataset using yfinance, and computes two simple, real financial metrics
per company:

  - `volatility_6m`: annualised standard deviation of daily returns over
     the trailing 6 months (a common proxy for market-perceived risk)
  - `return_6m`: total return over the trailing 6 months

These become the TARGET VARIABLE candidates for Stage 2's predictive
model. The hypothesis being tested (not assumed!) is: do companies whose
materiality profile deviates more from their sector peers (in either
direction) show different volatility/returns than companies that are
"sector-typical"? The ML model in Stage 2 will tell us whether this
hypothesis holds in this dataset — if it doesn't, that's a valid and
reportable finding too (and the dashboard should say so honestly).

NETWORK NOTE
------------
This sandbox's network egress does not allow query1/query2.finance.yahoo.com,
so this script will fail to fetch real data here. In a normal dev
environment (VS Code, your own machine) it should work directly. If it
fails, the script falls back to writing `financial_data.json` with
`status: "unavailable"` for each company, clearly marked — Stage 2 will
then skip the financial-target model and proceed with the
materiality-only explainability model instead (see 02_train_model.py).

HOW TO RUN
----------
    python3 00_fetch_financial_data.py

OUTPUT
------
    ../data/financial_data.json
    [{ "symbol": "AAL", "ticker": "AAL.L", "volatility_6m": 0.31,
       "return_6m": 0.04, "status": "ok" }, ...]
"""

import json
import pandas as pd
import numpy as np

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


def fetch_for_symbol(symbol):
    ticker = f"{symbol}.L"  # London Stock Exchange suffix
    try:
        hist = yf.Ticker(ticker).history(period="6mo")
        if hist.empty or len(hist) < 20:
            return {"ticker": ticker, "status": "no_data"}
        returns = hist["Close"].pct_change().dropna()
        volatility_6m = float(returns.std() * np.sqrt(252))  # annualised
        return_6m = float(hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1)
        return {
            "ticker": ticker,
            "status": "ok",
            "volatility_6m": round(volatility_6m, 4),
            "return_6m": round(return_6m, 4),
            "n_days": len(hist),
        }
    except Exception as e:
        return {"ticker": ticker, "status": "error", "error": str(e)}


def main():
    df = pd.read_csv("../data/ftse100_materiality.csv")

    results = []
    for _, row in df.iterrows():
        symbol = row["Symbol"]
        name = row["Name"]

        if not YFINANCE_AVAILABLE:
            results.append({"company": name, "symbol": symbol, "status": "unavailable",
                             "reason": "yfinance not installed"})
            continue

        print(f"Fetching {name} ({symbol})...", end=" ", flush=True)
        r = fetch_for_symbol(symbol)
        r["company"] = name
        r["symbol"] = symbol
        results.append(r)
        print(r["status"])

    with open("../data/financial_data.json", "w") as f:
        json.dump(results, f, indent=2)

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nSaved ../data/financial_data.json — {ok}/{len(results)} companies with real data")
    if ok == 0:
        print("\nNOTE: No real financial data was retrieved (likely a network restriction in "
              "this environment). Run this script in an environment with internet access to "
              "Yahoo Finance. Stage 2 (02_train_model.py) will detect this and fall back to a "
              "materiality-only explainability model.")


if __name__ == "__main__":
    main()
