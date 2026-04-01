# QuantBT

QuantBT is a modular, event-style trading backtesting engine built to look and behave like a serious quant research project rather than a notebook script. It supports multi-asset OHLCV data, realistic execution assumptions, portfolio accounting, train/test validation, in-sample parameter sweeps, out-of-sample robustness checks, benchmark comparison, and automated report generation.

## Highlights

- Event-style simulation loop with delayed execution to avoid same-bar fill assumptions
- Cleanly separated modules for data, strategies, execution, portfolio, metrics, and analysis
- Realistic market order fills with transaction costs, slippage, spread, and volume caps
- Three built-in strategies:
  - Moving average crossover
  - Mean reversion using rolling z-score
  - Momentum breakout using lagged breakout bands
- Built-in train/test split, in-sample grid search, out-of-sample evaluation, and slippage sensitivity checks
- Walk-forward optimization with stitched out-of-sample equity and parameter tracking
- Monte Carlo moving-block bootstrap for return-path robustness analysis
- Regime segmentation using inferred bull/bear and high/low-volatility states
- Multi-asset allocation constraints including capped single-name weights, idle cash buffers, active-name limits, and inverse-volatility sizing
- Quant-style metrics including Sharpe, Sortino, Calmar, drawdown, turnover, exposure, win rate, and profit factor
- Deterministic sample-data generator for reproducible local demos and tests
- Pytest coverage for signals, costs, execution timing, PnL accounting, and metrics

## Project Structure

```text
.
├── README.md
├── main.py
├── pyproject.toml
├── data/
├── reports/
├── src/
│   └── quantbt/
│       ├── analysis/
│       ├── backtester/
│       ├── data/
│       ├── execution/
│       ├── metrics/
│       ├── portfolio/
│       ├── strategies/
│       ├── config.py
│       ├── types.py
│       └── utils.py
└── tests/
```

## Research Design Choices

### No look-ahead bias

- Strategy indicators are built from data available at or before each bar.
- Orders generated from bar `t` signals are filled on bar `t+1` based on configurable `next_open` or `next_close` execution.
- The breakout strategy uses `shift(1)` breakout bands so the current bar never compares against a level that already includes itself.

### Realistic execution assumptions

- Commission, slippage, spread, and a volume participation cap are included by default.
- Orders can be partially filled when requested size exceeds the configured fraction of bar volume.
- Position sizing allocates gross notional across active signals rather than pretending every signal gets infinite liquidity.
- Portfolio construction can reserve cash, cap single-name weights, restrict the number of active positions, and switch between equal-weight and inverse-volatility allocation.

### Overfitting control

- Parameter selection happens only on the in-sample segment.
- The selected parameter set is then evaluated out of sample.
- Reports include slippage sensitivity, comparison with a cost-free run, walk-forward validation, bootstrap robustness statistics, and regime-level performance breakdowns.

### Survivorship-bias caveat

- The engine itself supports any universe supplied to it, including delisted names in CSV form.
- If you use present-day Yahoo Finance tickers, constituent survivorship is a property of the upstream data source rather than the engine.
- For interview-grade research, use a historical point-in-time universe and corporate-action-clean data when available.

## Installation

The global Python scientific stack on some systems can be inconsistent, so using a local virtual environment is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

## Quick Start

Generate deterministic sample data:

```bash
python main.py generate-sample-data --output-dir data --periods 504 --assets SPY,QQQ,IWM
```

Run the research workflow on CSV data:

```bash
python main.py run --data-dir data --output-dir reports
```

Run directly from Yahoo Finance:

```bash
python main.py run --tickers SPY,QQQ,IWM --start 2018-01-01 --end 2024-12-31 --output-dir reports
```

Allow shorting and change execution assumptions:

```bash
python main.py run \
  --data-dir data \
  --allow-short \
  --price-source next_open \
  --commission-bps 5 \
  --slippage-bps 3 \
  --spread-bps 2 \
  --allocation-scheme inverse_volatility \
  --max-asset-weight 0.25 \
  --max-active-positions 5 \
  --min-cash-buffer 0.03 \
  --bootstrap-iterations 500 \
  --walk-forward-train-bars 126 \
  --walk-forward-test-bars 63
```

## Output Artifacts

For each strategy, QuantBT writes:

- `reports/<strategy>/in_sample/metrics.csv`
- `reports/<strategy>/out_of_sample/metrics.csv`
- `reports/<strategy>/out_of_sample/benchmark_metrics.csv`
- `reports/<strategy>/out_of_sample/robustness.csv`
- `reports/<strategy>/out_of_sample/walk_forward_segments.csv`
- `reports/<strategy>/out_of_sample/walk_forward_metrics.csv`
- `reports/<strategy>/out_of_sample/bootstrap_stats.csv`
- `reports/<strategy>/out_of_sample/regime_summary.csv`
- `reports/<strategy>/out_of_sample/regime_assignments.csv`
- `reports/<strategy>/summary.txt`
- Plot images for equity, drawdown, rolling risk, monthly heatmap, parameter sweep, walk-forward equity, bootstrap distributions, and regime performance

## Testing

```bash
pytest
```

## Extending The Engine

To add a new strategy:

1. Create a new class in `src/quantbt/strategies/` that inherits from `BaseStrategy`.
2. Implement `generate_signals(self, market_data)`.
3. Add a default parameter grid for research sweeps.
4. Register the strategy in `main.py`.

To plug in a new data source:

1. Load the source into a DataFrame with `timestamp`, `open`, `high`, `low`, `close`, and `volume`.
2. Convert it into the canonical `MarketData` container.
3. Reuse the same engine, execution, and reporting pipeline unchanged.
