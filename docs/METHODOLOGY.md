# Methodology Notes

Supplementary methodological details for the replication code in this repository. For full theoretical derivations, see the dissertation Chapter 3 and Appendix D.

## 1. Liquidation Identification

### Hyperliquid (dual filter)

The main panel applies a **dual filter** to identify liquidation trades:

1. **Counterparty address**: `0x0000…0000` (the protocol-controlled liquidator address)
2. **Fee tag**: indicates `liquidationFee`

Both conditions must hold. This dual filter eliminates ~3% false positives from internal vault rebalances.

Implementation: `scripts/01_panel_build/hl_parse_l2.py`

### Binance (Appendix E — implied liquidation)

Binance Vision's `liquidationSnapshot` archive is unavailable for the event period (last archived 2024). Appendix E uses **implied liquidation from open interest changes** in `metrics`:

```
liq_implied_t = max(0, ΔOI_t × sign(ret_t < 0)) for forced longs
              + max(0, |ΔOI_t| × sign(ret_t > 0)) for forced shorts
```

This is the methodology used by industry data vendors (Coinalyze, CoinGlass).

Implementation: `scripts/07_cross_exchange/binance_panel_build.py`

## 2. Spread Estimators

### Hyperliquid: quoted spread from L2 snapshots

```
spread_bps_t = (best_ask_t - best_bid_t) / mid_t × 10,000
```

aggregated to minute via time-weighted average.

### Binance: Roll (1984) effective spread

Binance Vision stopped daily `bookTicker` archival after 2024-03-30. Appendix E uses the Roll high-frequency estimator:

```
spread_roll = 2 × sqrt(-cov(Δp_t, Δp_{t-1}))
```

computed within each minute. Biases the level downward but preserves event-vs-pre comparisons.

Implementation: `scripts/07_cross_exchange/binance_panel_build.py`

## 3. Standardization

The interaction term `liq_x_event = liq_vol_std × D_event` uses **pre-event standardization** (per-coin mean and standard deviation computed only on the 3-day pre-event window). This avoids look-ahead bias.

```python
for coin in coin_list:
    pre_data = df[(df.coin == coin) & df.is_pre]
    mu = pre_data.liq_vol_usd.mean()
    sd = pre_data.liq_vol_usd.std()
    df.loc[df.coin == coin, "liq_vol_std"] = (df.liq_vol_usd - mu) / sd
```

## 4. Standard Errors

### Driscoll-Kraay (main spec)

For panel regression with cross-sectional dependence, the main specification uses Driscoll-Kraay (1998) standard errors:

```python
from linearmodels.panel import PanelOLS

mod = PanelOLS.from_formula(
    "log_spread ~ liq_vol_std + liq_x_event + ret + log_vol + EntityEffects + TimeEffects",
    data=df.set_index(["coin", "minute"]),
).fit(cov_type="kernel", kernel="bartlett", bandwidth=4)
```

Bandwidth choice: AR(1) rule of thumb. Robustness checks at bw = 2, 4, 8.

### Newey-West (alternative spec)

For single-coin or pooled regressions, Newey-West HAC standard errors are used (similar bandwidth).

### Boehmer-Musumeci-Poulsen (Chapter 4 event study)

Cross-sectional inference on Cumulative Abnormal Spread / Depth uses the BMP (1991) standardization to address event-induced variance.

## 5. Identification Assumptions

### Parallel trends

Tested via pre-period regression: no significant differential trend in spread/depth between treated (event window) and control (pre-period) periods.

Test: `scripts/02_event_study/ch4_volume_analysis.py` (pre_signal table).

### SUTVA (Stable Unit Treatment Value Assumption)

Assumes one coin's treatment does not affect another coin's outcome — a strong assumption given cross-asset spillovers. Mitigated by:
1. Including all 9 coins simultaneously (within-event variation in spillover intensity)
2. Cross-asset correlation matrix analysis (Chapter 4 §4.5)
3. Triple-difference cross-exchange identification (Appendix E)

### No reverse causality

Liquidations precede spread/depth changes within the same minute (event ordering check). The 30-second lag structure rules out simultaneity.

## 6. Placebo Tests (Appendix B)

Three-layer placebo design (51 total tests):

| Layer | Type | n | Description |
|---|---|---|---|
| L1 | Time-of-day | 3 | Same hour-of-day on non-event dates |
| L2 | Random timestamps | 20 | Uniform sampling from pre-event window |
| L3 | Rolling 24h windows | 28 | All possible 24h windows in extended sample |

Decision rule: Pass = 5% left-tail of placebo β distribution (i.e., observed β more extreme than 95% of placebo β).

Result: 0 of 51 passed → H3 effect is event-specific, not artifact.

Implementation: `scripts/04_placebo/placebo_test.py`

## 7. Multi-Event Replication (Appendix F)

Three events tested:
1. **2025-10-10** (main): Trump tariff announcement → $1.5B HL liquidations
2. **2025-11-21**: BLS jobs invalidation → BTC flash crash, $0.8B HL liquidations
3. **2026-01-30**: Trump's Kevin Warsh Fed Chair nomination + US budget deadline + BTC support-level breakdown

All 3 events show amp > 1, confirming H3 is not unique to the main event.

Daily peak day identification: `scripts/08_multi_event/hl_all_coin_hourly_liq.py` extracts all-coin hourly liquidation totals to rank candidate event days.

## 8. GM Model Calibration (Appendix D)

Two-stage equilibrium extension of Glosten-Milgrom (1985) with:
- Transparency parameter V ∈ [0, 1]
- Speculative absorption intensity γ ∈ [0, 1]

Grid scan: 21 × 21 = 441 (V, γ) combinations.

Best-fit point: V* = 0.464, γ* = 0.600 → amp = 3.94× (matches empirical).

Implementation: `scripts/06_calibration/model_simulation.py`

## References

Key methodological references (full list in dissertation References):

- Brunnermeier, M. K., & Pedersen, L. H. (2009). Market liquidity and funding liquidity. *RFS*, 22(6).
- Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *Review of Economics and Statistics*, 80(4).
- Easley, D., Lopez de Prado, M. M., & O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. *RFS*, 25(5).
- Glosten, L. R., & Milgrom, P. R. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *JFE*, 14(1).
- Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an efficient market. *JF*, 39(4).
