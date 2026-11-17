# Robust Avellaneda–Stoikov Market Maker

An extended version of the classic **Avellaneda–Stoikov (2008)** market-making model, built to handle the situations where the textbook model breaks: changing volatility, informed traders, stale quotes, price jumps, and unbounded inventory.

A **C++** implementation is included, with a simulator that replays the exact same market data through the vanilla and the extended strategy so they can be compared fairly.

> All data is simulated. This is a research / demonstration project, not a production trading system.

---

## 1. Background: what vanilla Avellaneda–Stoikov does

A market maker posts a bid and an ask around a **reservation price** that moves away from the mid price when it holds inventory:

```
reservation price   r = s − q · γ · σ² · (T − t)
total spread        δ = γ · σ² · (T − t) + (2/γ) · ln(1 + γ/k)
bid = r − δ/2       ask = r + δ/2
```

`s` = mid price, `q` = inventory, `γ` = risk aversion, `σ` = volatility, `k` = order-book depth parameter, `T − t` = time left in the day.

It is elegant, but it relies on assumptions that rarely hold in real markets.

## 2. What is extended

| # | Limitation of vanilla A-S | Extension in this project |
|---|---|---|
| 1 | Volatility `σ` is a fixed input | **Online volatility estimate** (EWMA, with big moves winsorized so one jump does not corrupt it) |
| 2 | Depth parameter `k` is a fixed input | **`k` learned live** from the trade tape (maximum-likelihood estimate of how deep market orders reach) |
| 3 | Ignores informed traders (adverse selection) | **Order-flow predictor**: an online regression on signed order flow that shifts the reservation price and widens quotes when flow looks toxic |
| 4 | Assumes zero latency, no price jumps | **Latency buffer** in the spread, **jump detector**, and a **cooldown mode** that widens and shrinks quotes (instead of going dark) |
| 5 | No inventory limits | **Soft cap** (size scaling, one-sided quoting) and **hard cap** (automatic flatten) |
| 6 | Risk term shrinks to zero at the end of the day | **Rolling horizon** plus extra risk aversion in the last 10% of the day, so the book ends flat |
| 7 | Ignores fees and tick size | **Edge floor** above fees, quotes **rounded to ticks** |
| 8 | No loss control | **Drawdown breaker** that widens quotes after a loss threshold |

### Extended quote logic (simplified)

```
r      = s + predicted_move − q · γ_eff · σ̂² · h
half   = 0.5 · γ_eff · σ̂² · h + (1/γ) · ln(1 + γ/k̂)      # A-S core, with estimated σ̂, k̂
half  += c_tox · |predicted_move| + c_lat · σ̂ · √(dt · latency)
half   = max(half, fee + min_edge)
cooldown / drawdown → half × 1.8, sizes × 0.5
inventory → shrink size on the side that adds risk, flatten above the hard cap
```

## 3. How it is better

The extended version is a **risk-control upgrade**. It lowers inventory and drawdown risk. It does **not** make more money than vanilla.

Results on 200 simulated days (seed 999, not used while adjusting parameters). The same price and order tape is used for every strategy.

| Metric | Vanilla  | Extended  | Vanilla (C++) | Extended (C++) |
|---|---|---|---|---|
| Mean daily PnL | 59.0 | 56.3 | 58.5 | 55.2 |
| Sharpe (daily) | 5.16 | 4.90 | 4.66 | 4.93 |
| Max inventory held | 14 | **8.9** | 14 | **8.5** |
| Avg max drawdown | 5.5 | **4.7** | 5.5 | **4.8** |
| Avg quoted spread | 1.50 | 2.00 | 1.50 | 1.98 |
| Spread during stress | 1.49 | 2.93 | 1.49 | 2.85 |
| Two-sided quote uptime | 100% | 100% | 100% | 100% |

(C++ use different random number generators, so their exact figures differ. The behaviour is the same.)

### Where it is clearly better
- **Inventory risk:** peak inventory falls by roughly 35–50%.
- **Drawdowns:** average intraday drawdown falls by roughly 12–20%.
- **Stress behaviour:** spreads widen when volatility rises, instead of staying at the same level while being picked off. Quotes stay visible (100% uptime).
- **Robustness to estimation:** `σ` and `k` are learned, so the model does not depend on knowing them in advance.

### Where it is not better (honest results)
- **Mean PnL is lower** by about 2–3 per day (about 5%). The paired difference is statistically significant.
- **Sharpe is mixed:** better in 3 of 4 comparisons, worse in the Python fresh-seed run.
- **Customers pay wider spreads:** about 33% wider on average and about double in stress.
- **Order-flow predictor is unproven:** removing it gave a higher Sharpe in several runs, so treat it as experimental.
- **Inventory caps rarely trigger** with the default settings, so they work as insurance, not a measured gain.

## 4. Who it is meant to help

**Trading firm:** smaller inventory swings, smaller drawdowns, and safeguards for jumps and stale quotes, at the cost of slightly lower average profit.

**Customers trading against the quotes:** quotes stay available during stress and spreads adapt to risk. The cost is a wider average spread. Whether stable liquidity brings in more business is not tested here.

## 5. Project files

| File | Description |
|---|---|
| `robust_market_maker.py` | Python: simulator, vanilla A-S, extended A-S, ablations, metrics |
| `robust_market_maker.cpp` | C++ port with the same logic |
| `daily_results.csv` | Per-day results written when you run either program |

## 6. How to run

**Python** (needs `numpy`):
```bash
python3 robust_market_maker.py
```

**C++** (C++17):
```bash
g++ -O2 -std=c++17 robust_market_maker.cpp -o robust_mm
./robust_mm              # 120 days, seed 2024
./robust_mm 200 999      # custom number of days and seed
```

Output: a comparison table of PnL, Sharpe, tail loss, drawdown, inventory, spreads and quote uptime for the vanilla strategy, the extended strategy, and three ablations (without flow predictor, without inventory limits, without jump/drawdown guard), plus a paired PnL test.

## 7. Simulator

- Mid price with regime-switching volatility (calm / stress) and random jumps
- Hidden drift that informed traders follow (creates adverse selection)
- Market orders arrive as a Poisson process; each reaches a random depth, and the depth distribution changes in stress
- Quotes are based on a delayed price (2 steps of latency), so stale quotes can be picked off
- Fees per unit traded and a cost to cross the spread when flattening
- End-of-day inventory is liquidated at a cost

## 8. Limitations

- Synthetic data only. Real results require real tick data.
- Our own orders do not move the price.
- No queue position or order-book modelling.
- Parameters were adjusted once after the first run, so the first-seed numbers are slightly in-sample.
- Single-asset, single-venue.

## 9. Possible future work

- Test and tune on real market data
- Improve or replace the order-flow predictor
- Add queue-position and partial-fill modelling
- Multi-asset inventory risk
- Parameter calibration by optimisation with out-of-sample validation

## 10. Reference

Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book.* Quantitative Finance, 8(3), 217–224.
