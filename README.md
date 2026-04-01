# QuantBT

A professional-grade event-driven backtesting engine for systematic trading research in Python.

QuantBT is built to look and feel like a serious quant research project rather than a notebook demo. It includes realistic execution modeling, multi-asset portfolio simulation, in-sample and out-of-sample evaluation, walk-forward optimization, Monte Carlo robustness checks, regime segmentation, and a modular architecture that is easy to extend with new strategies and data sources.

## Why This Project Stands Out

- Event-driven simulation with delayed execution to avoid naive same-bar fills
- Explicit guardrails against look-ahead bias
- Realistic execution costs:
  commission, slippage, spread, and participation caps
- Multi-asset portfolio construction with allocation constraints
- Research workflow with:
  train/test split, parameter sweep, walk-forward validation, bootstrap robustness, benchmark comparison
- Quant-style performance analytics:
  Sharpe, Sortino, Calmar, drawdown, turnover, exposure, trade-level statistics
- Clean engineering:
  typed modules, reusable abstractions, deterministic sample data, pytest coverage

## Built-In Strategies

- Moving Average Crossover
- Mean Reversion using rolling z-score / Bollinger-style logic
- Momentum Breakout using lagged breakout bands

Each strategy exposes a common `generate_signals(data)` interface, making it straightforward to add more alpha models later.

## Core Features

### Data Layer

- Load OHLCV data from CSV
- Optionally download data from Yahoo Finance
- Standardize timestamps and column names
- Validate sorting, duplicates, and missing values
- Support multiple assets
- Resample data into coarser bars if needed

### Execution Layer

- Market-order execution at `next_open` or `next_close`
- Commission model in basis points plus optional fixed fees
- Slippage model with volume-share impact
- Optional spread costs
- Partial fills under bar-volume participation limits
- Long-only or long/short portfolio support

### Portfolio Construction

- Equal-weight or inverse-volatility allocation
- Per-asset weight caps
- Max active positions constraint
- Cash reserve buffer
- Gross leverage control

### Research Workflow

- In-sample parameter sweep
- Out-of-sample evaluation
- Buy-and-hold benchmark comparison
- Cost-free vs cost-aware comparison
- Walk-forward optimization with stitched out-of-sample equity
- Monte Carlo moving-block bootstrap on returns
- Regime segmentation across bull/bear and high/low-volatility environments

## Architecture

```text
.
├── README.md
├── main.py
├── pyproject.toml
├── requirements.txt
├── data/
├── reports/
├── src/
│   └── quantbt/
│       ├── analysis/
│       │   ├── regimes.py
│       │   ├── reporting.py
│       │   └── robustness.py
│       ├── backtester/
│       │   ├── engine.py
│       │   ├── sweep.py
│       │   └── walk_forward.py
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

## Quant Design Principles

### No Look-Ahead Bias

- Indicators use only data available up to the signal bar
- Orders created from bar `t` are filled on bar `t + 1`
- Breakout thresholds are shifted so the current bar never sees itself

### Realistic Simulation

- Transaction costs are included by default
- Slippage scales with participation
- Fill sizes are constrained by bar volume
- Portfolio weights can be capped, throttled, and cash-buffered

### Overfitting Controls

- Parameter selection happens on in-sample data only
- Final evaluation is done out of sample
- Walk-forward validation checks stability across rolling windows
- Bootstrap analysis measures path dependence and tail outcomes
- Regime segmentation shows where performance is robust and where it is fragile

### Survivorship Bias Note

The engine can handle any universe supplied to it, including delisted names in CSV form. If you use present-day Yahoo Finance tickers, survivorship bias comes from the upstream dataset, not the backtester. For production-quality research, point-in-time universes and corporate-action-clean data are recommended.

## Installation

Using a virtual environment is recommended.

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

Run the default research workflow on local CSV data:

```bash
python main.py run --data-dir data --output-dir reports
```

Run directly on Yahoo Finance data:

```bash
python main.py run --tickers SPY,QQQ,IWM --start 2018-01-01 --end 2024-12-31 --output-dir reports
```

Run with richer portfolio construction and research settings:

```bash
python main.py run \
  --data-dir data \
  --strategies moving_average,mean_reversion,breakout \
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
  --walk-forward-test-bars 63 \
  --walk-forward-step-bars 63 \
  --regime-lookback 63 \
  --output-dir reports
```

## CLI Options You’ll Likely Use

- `--strategies`
  choose one or more strategies
- `--price-source`
  `next_open` or `next_close`
- `--allow-short`
  enable long/short simulation
- `--allocation-scheme`
  `equal_weight` or `inverse_volatility`
- `--max-asset-weight`
  cap single-name concentration
- `--max-active-positions`
  limit simultaneous holdings
- `--bootstrap-iterations`
  control Monte Carlo robustness depth
- `--walk-forward-train-bars`, `--walk-forward-test-bars`
  tune rolling validation windows
- `--regime-lookback`
  tune market-state inference horizon

## Example Output Artifacts

For each strategy, QuantBT writes a research bundle under `reports/<strategy>/`.

Typical artifacts include:

- `in_sample/metrics.csv`
- `in_sample/parameter_sweep.csv`
- `out_of_sample/metrics.csv`
- `out_of_sample/benchmark_metrics.csv`
- `out_of_sample/robustness.csv`
- `out_of_sample/walk_forward_segments.csv`
- `out_of_sample/walk_forward_metrics.csv`
- `out_of_sample/bootstrap_stats.csv`
- `out_of_sample/regime_summary.csv`
- `summary.txt`

Plots include:

- Equity curve
- Drawdown curve
- Rolling Sharpe / rolling volatility
- Monthly return heatmap
- Parameter sweep visualization
- Walk-forward equity curve
- Bootstrap distribution plots
- Regime performance chart

## Testing

Run the test suite:

```bash
pytest
```

Current coverage includes:

- Signal generation correctness
- No future leakage behavior
- Execution timing
- Transaction cost and slippage calculations
- Position and PnL accounting
- Metric calculations
- Walk-forward research utilities
- Allocation constraints
- Regime segmentation analysis

## Extending The Engine

### Add a New Strategy

1. Create a class in `src/quantbt/strategies/`
2. Inherit from `BaseStrategy`
3. Implement `generate_signals(self, market_data)`
4. Add a default parameter grid
5. Register it in `main.py`

### Add a New Data Source

1. Load data into a DataFrame with `timestamp`, `open`, `high`, `low`, `close`, `volume`
2. Convert it into the canonical `MarketData` container
3. Reuse the same backtester, execution, metrics, and reporting pipeline

## Interview-Ready Talking Points

This project supports saying, truthfully:

- I built an event-driven backtesting engine in Python
- I modeled realistic execution costs and slippage
- I tested multiple systematic strategies
- I evaluated risk-adjusted performance and trade-level behavior
- I compared in-sample vs out-of-sample results
- I ran walk-forward validation and Monte Carlo robustness checks
- I analyzed performance across different market regimes

## Roadmap Ideas

- Correlation-aware portfolio optimization
- Regime-aware strategy activation filters
- Sector or factor exposure constraints
- Multi-frequency strategies
- Intraday support
- Live paper-trading adapter
