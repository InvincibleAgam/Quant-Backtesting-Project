# QuantBT

QuantBT is an event-driven backtesting framework for systematic trading research in Python. The codebase is organized around a clean separation between data ingestion, signal generation, execution simulation, portfolio accounting, performance measurement, and report generation.

The project is designed for research workflows where execution timing, transaction costs, portfolio constraints, and out-of-sample validation matter. It supports both local CSV-based workflows and Yahoo Finance data, and it produces reproducible analysis artifacts for each strategy run.

## Scope

The framework currently includes:

- Multi-asset OHLCV data loading and validation
- Three reference strategies:
  moving average crossover, mean reversion, and momentum breakout
- Event-style execution with delayed fills
- Commission, spread, and slippage modeling
- Portfolio construction with leverage, cash buffer, and position concentration constraints
- Standard performance and risk metrics
- In-sample parameter sweeps
- Out-of-sample evaluation
- Walk-forward optimization
- Moving-block bootstrap robustness analysis
- Regime segmentation analysis
- Automated report and plot generation
- Unit tests for core engine behavior

## Architecture

The project follows a modular structure under `src/quantbt/`.

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

### Module Responsibilities

- `data/`
  loaders and helpers for standardized OHLCV data
- `strategies/`
  signal generation logic with a common strategy interface
- `execution/`
  order fill modeling and position sizing
- `portfolio/`
  cash, positions, realized/unrealized PnL, exposure, and trade tracking
- `backtester/`
  event loop, parameter sweep runner, and walk-forward runner
- `metrics/`
  return, risk, drawdown, and trade-level statistics
- `analysis/`
  report generation, plots, robustness summaries, and regime analysis

## Data Model

Market data is normalized into a canonical `MarketData` container with a `MultiIndex` of `timestamp` and `asset`. Required fields are:

- `open`
- `high`
- `low`
- `close`
- `volume`

This standardization keeps strategy logic independent from raw data-source details and allows the rest of the engine to operate on a uniform interface.

## Strategies

Three strategies are included as reference implementations.

### Moving Average Crossover

A directional trend-following strategy based on fast and slow moving averages. Signals are long when the fast average is above the slow average and short when it is below, unless the strategy is configured as long-only.

### Mean Reversion

A Bollinger-style mean reversion strategy that uses rolling z-scores to enter on dislocations and exit when the deviation compresses. The implementation maintains position state explicitly to avoid unrealistic bar-to-bar oscillation around the threshold.

### Momentum Breakout

A breakout strategy using lagged breakout bands. The breakout thresholds are computed from prior bars only, so the current bar does not contribute to its own trigger level.

All strategies implement a common `generate_signals(market_data)` interface and can expose a default parameter grid for research sweeps.

## Execution Model

Execution is modeled explicitly rather than assuming that strategy signals are filled at the same price used to create them.

### Fill Timing

Signals are generated on bar `t`. Orders are then scheduled for execution on the next eligible bar, using either:

- `next_open`
- `next_close`

This is the main mechanism used to avoid same-bar execution assumptions and reduce look-ahead bias.

### Transaction Costs

The execution model supports:

- basis-point commission
- optional fixed commission
- half-spread cost
- slippage in basis points
- volume-share slippage impact

### Liquidity Constraint

Orders are capped by a configurable fraction of bar volume. If the requested size exceeds the cap, the remaining quantity is deferred and can be carried forward as residual orders.

## Portfolio Construction

The position-sizing layer converts signals into target share counts subject to portfolio constraints.

Supported controls include:

- long-only or long/short mode
- maximum gross leverage
- per-asset weight cap
- minimum cash buffer
- maximum number of active positions
- equal-weight or inverse-volatility allocation

These controls make the default research workflow closer to an actual portfolio construction problem than a single-asset toy backtest.

## Portfolio Accounting

The portfolio layer tracks:

- cash
- current positions
- average cost basis
- realized PnL
- unrealized PnL
- market value
- equity
- returns
- turnover
- gross exposure
- net exposure
- closed trades
- raw fills

Trade records are generated when a position is reduced or closed, and portfolio snapshots are recorded at each bar close.

## Metrics

The metrics module computes a standard set of performance and risk statistics:

- cumulative return
- annualized return
- annualized volatility
- Sharpe ratio
- Sortino ratio
- maximum drawdown
- Calmar ratio
- win rate
- profit factor
- average trade return
- turnover
- average gross exposure
- number of trades

These metrics are computed both for strategy results and for the benchmark series used in the reports.

## Research Workflow

The default research flow is:

1. Load and validate historical data.
2. Split the sample into in-sample and out-of-sample segments.
3. Run a parameter sweep on the in-sample segment.
4. Select the best parameter set according to the target metric.
5. Evaluate the selected strategy on the out-of-sample segment.
6. Compare the result with a simple buy-and-hold benchmark.
7. Re-run the out-of-sample result under alternative slippage assumptions.
8. Optionally compare cost-aware and cost-free performance.
9. Run walk-forward optimization across rolling windows.
10. Run moving-block bootstrap analysis on returns.
11. Segment out-of-sample behavior by inferred market regime.
12. Write summary tables, CSV artifacts, and plots.

## Bias Control and Modeling Assumptions

### Look-Ahead Bias

The framework is structured to reduce look-ahead bias in several places:

- signal generation uses information available up to the signal bar
- execution occurs after signal generation
- breakout thresholds are lagged
- walk-forward parameter selection is performed using prior data only

### Survivorship Bias

The engine itself does not impose a survivorship-biased universe. If the input dataset contains delisted assets, they can be processed like any other symbol. When using present-day Yahoo Finance tickers, any survivorship bias comes from the data source and universe definition rather than the simulation engine.

### Overfitting

The framework includes several controls intended for research discipline rather than optimization-by-default:

- in-sample vs out-of-sample separation
- parameter sweep reporting instead of single-run optimization only
- walk-forward validation
- bootstrap robustness analysis
- regime-level performance breakdowns

## Installation

Using a local virtual environment is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

## Usage

### Generate Sample Data

```bash
python main.py generate-sample-data --output-dir data --periods 504 --assets SPY,QQQ,IWM
```

### Run a Backtest from Local CSV Data

```bash
python main.py run --data-dir data --output-dir reports
```

### Run a Backtest from Yahoo Finance

```bash
python main.py run --tickers SPY,QQQ,IWM --start 2018-01-01 --end 2024-12-31 --output-dir reports
```

### Example Research Run

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

## Common CLI Options

- `--strategies`
  comma-separated strategy list
- `--price-source`
  execution timing assumption, `next_open` or `next_close`
- `--allow-short`
  enable long/short positioning
- `--allocation-scheme`
  `equal_weight` or `inverse_volatility`
- `--max-asset-weight`
  hard cap on single-name target weight
- `--max-active-positions`
  cap the number of simultaneous holdings
- `--min-cash-buffer`
  keep a portion of capital unallocated
- `--bootstrap-iterations`
  number of bootstrap resamples
- `--walk-forward-train-bars`
  training window size for walk-forward optimization
- `--walk-forward-test-bars`
  test window size for walk-forward optimization
- `--regime-lookback`
  lookback window used for regime inference

## Output

For each strategy, the framework writes a report bundle under `reports/<strategy>/`.

Typical outputs include:

- `in_sample/metrics.csv`
- `in_sample/parameter_sweep.csv`
- `out_of_sample/metrics.csv`
- `out_of_sample/benchmark_metrics.csv`
- `out_of_sample/robustness.csv`
- `out_of_sample/walk_forward_segments.csv`
- `out_of_sample/walk_forward_metrics.csv`
- `out_of_sample/bootstrap_stats.csv`
- `out_of_sample/regime_summary.csv`
- `out_of_sample/regime_assignments.csv`
- `summary.txt`

Generated plots include:

- equity curve
- drawdown curve
- rolling Sharpe / volatility
- monthly return heatmap
- parameter sweep plot
- walk-forward equity plot
- bootstrap distribution plots
- regime performance plot

## Testing

Run the test suite with:

```bash
pytest
```

The current test set covers:

- signal generation behavior
- no-look-ahead properties
- transaction cost and slippage calculations
- execution timing
- position and PnL accounting
- metric calculations
- walk-forward utilities
- allocation constraints
- regime segmentation logic

## Extending the Framework

### Adding a Strategy

1. Add a new strategy class under `src/quantbt/strategies/`.
2. Inherit from `BaseStrategy`.
3. Implement `generate_signals(self, market_data)`.
4. Provide a default parameter grid if the strategy should participate in sweeps.
5. Register the strategy in `main.py`.

### Adding a Data Source

1. Load raw data into a DataFrame with `timestamp`, `open`, `high`, `low`, `close`, and `volume`.
2. Convert the result into the canonical `MarketData` representation.
3. Reuse the existing execution, backtesting, metrics, and reporting layers.

## Roadmap

- correlation-aware portfolio optimization
- regime-aware strategy activation rules
- exposure constraints by sector or factor
- multi-frequency workflows
- intraday datasets and execution assumptions
- paper-trading or live-trading adapters
