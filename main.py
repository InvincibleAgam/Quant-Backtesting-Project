"""CLI entry point for the QuantBT research backtesting engine."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from quantbt.analysis.regimes import analyze_regime_performance, infer_market_regimes
from quantbt.analysis.reporting import create_strategy_report
from quantbt.analysis.robustness import (
    bootstrap_return_paths,
    slippage_sensitivity,
    write_research_summary,
)
from quantbt.backtester import BacktestEngine, ParameterSweepRunner, WalkForwardRunner
from quantbt.config import BacktestConfig, ExecutionConfig, ResearchConfig
from quantbt.data import CSVDataLoader, YFinanceDataLoader, generate_sample_ohlcv
from quantbt.strategies import (
    MomentumBreakoutStrategy,
    MovingAverageCrossoverStrategy,
    MeanReversionStrategy,
)
from quantbt.utils import set_random_seed, train_test_split_index


LOGGER = logging.getLogger("quantbt")
STRATEGIES = {
    "moving_average": MovingAverageCrossoverStrategy,
    "mean_reversion": MeanReversionStrategy,
    "breakout": MomentumBreakoutStrategy,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(description="QuantBT event-driven backtesting engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("generate-sample-data", help="Create deterministic sample CSV data")
    sample.add_argument("--output-dir", default="data", help="Where to write sample CSV files")
    sample.add_argument("--periods", type=int, default=504, help="Number of business-day bars to generate")
    sample.add_argument("--assets", default="SPY,QQQ,IWM", help="Comma-separated list of sample assets")

    run = subparsers.add_parser("run", help="Run research backtests")
    run.add_argument("--data-dir", default="data", help="Directory of OHLCV CSV files")
    run.add_argument("--tickers", default="", help="Optional comma-separated Yahoo Finance tickers")
    run.add_argument("--start", default="2018-01-01", help="Data start date for Yahoo Finance")
    run.add_argument("--end", default="2024-12-31", help="Data end date for Yahoo Finance")
    run.add_argument(
        "--strategies",
        default="moving_average,mean_reversion,breakout",
        help="Comma-separated strategies to evaluate",
    )
    run.add_argument("--output-dir", default="reports", help="Output report directory")
    run.add_argument("--initial-cash", type=float, default=1_000_000.0, help="Initial portfolio cash")
    run.add_argument("--train-fraction", type=float, default=0.6, help="In-sample fraction")
    run.add_argument("--price-source", choices=["next_open", "next_close"], default="next_open")
    run.add_argument("--commission-bps", type=float, default=5.0)
    run.add_argument("--slippage-bps", type=float, default=3.0)
    run.add_argument("--spread-bps", type=float, default=2.0)
    run.add_argument("--allow-short", action="store_true", help="Allow short positions")
    run.add_argument("--allocation-scheme", choices=["equal_weight", "inverse_volatility"], default="equal_weight")
    run.add_argument("--max-asset-weight", type=float, default=0.35)
    run.add_argument("--max-active-positions", type=int, default=0, help="0 means no explicit cap")
    run.add_argument("--min-cash-buffer", type=float, default=0.02)
    run.add_argument("--volatility-lookback", type=int, default=20)
    run.add_argument("--regime-lookback", type=int, default=63)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--bootstrap-iterations", type=int, default=500, help="Monte Carlo bootstrap iterations")
    run.add_argument("--bootstrap-block-size", type=int, default=5, help="Moving block size for bootstrap")
    run.add_argument("--walk-forward-train-bars", type=int, default=126, help="Training bars per walk-forward segment")
    run.add_argument("--walk-forward-test-bars", type=int, default=63, help="Test bars per walk-forward segment")
    run.add_argument("--walk-forward-step-bars", type=int, default=63, help="Step size between walk-forward segments")

    return parser


def write_sample_data(output_dir: str | Path, periods: int, assets: tuple[str, ...], seed: int = 7) -> None:
    """Generate sample OHLCV files that can be loaded by the CSV loader."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    market_data = generate_sample_ohlcv(assets=assets, periods=periods, seed=seed)
    for asset in market_data.assets:
        frame = market_data.asset_frame(asset).reset_index()
        frame.to_csv(output_path / f"{asset}.csv", index=False)


def load_market_data(args: argparse.Namespace):
    """Load data from CSV, Yahoo Finance, or deterministic synthetic generation."""

    tickers = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
    if tickers:
        LOGGER.info("Loading Yahoo Finance data for %s", tickers)
        return YFinanceDataLoader().load(tickers=tickers, start=args.start, end=args.end)

    data_dir = Path(args.data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    if csv_files:
        LOGGER.info("Loading CSV data from %s", data_dir)
        return CSVDataLoader().load_directory(data_dir)

    LOGGER.info("No CSV data found. Falling back to deterministic sample data.")
    return generate_sample_ohlcv()


def _subset_market_data(market_data, start: pd.Timestamp, end: pd.Timestamp):
    return market_data.between(start, end)


def _extract_best_parameters(sweep: pd.DataFrame) -> dict[str, object]:
    row = sweep.iloc[0].to_dict()
    excluded = {"strategy", "sharpe_ratio", "annualized_return", "maximum_drawdown", "number_of_trades"}
    return {key: value for key, value in row.items() if key not in excluded}


def run_research(args: argparse.Namespace) -> list[Path]:
    """Run the end-to-end research workflow."""

    set_random_seed(args.seed)
    backtest_config = BacktestConfig(initial_cash=args.initial_cash, train_fraction=args.train_fraction, random_seed=args.seed)
    execution_config = ExecutionConfig(
        price_source=args.price_source,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        spread_bps=args.spread_bps,
        allow_short=args.allow_short,
        allocation_scheme=args.allocation_scheme,
        max_asset_weight=args.max_asset_weight,
        max_active_positions=args.max_active_positions or None,
        min_cash_buffer=args.min_cash_buffer,
        volatility_lookback=args.volatility_lookback,
    )
    research_config = ResearchConfig(output_dir=Path(args.output_dir))
    research_config.bootstrap_iterations = args.bootstrap_iterations
    research_config.bootstrap_block_size = args.bootstrap_block_size
    research_config.walk_forward_train_bars = args.walk_forward_train_bars
    research_config.walk_forward_test_bars = args.walk_forward_test_bars
    research_config.walk_forward_step_bars = args.walk_forward_step_bars
    research_config.regime_lookback = args.regime_lookback
    market_data = load_market_data(args)
    train_end, test_start = train_test_split_index(market_data.timestamps, backtest_config.train_fraction)
    train_data = _subset_market_data(market_data, market_data.timestamps.min(), train_end)
    test_data = _subset_market_data(market_data, test_start, market_data.timestamps.max())
    output_paths: list[Path] = []

    selected_strategies = [name.strip() for name in args.strategies.split(",") if name.strip()]
    for strategy_key in selected_strategies:
        strategy_class = STRATEGIES[strategy_key]
        LOGGER.info("Running strategy %s", strategy_key)

        sweep_runner = ParameterSweepRunner(backtest_config, execution_config)
        sweep = sweep_runner.run(
            market_data=train_data,
            strategy_class=strategy_class,
            param_grid=strategy_class.default_parameter_grid(),
            long_only=not args.allow_short,
        )
        best_parameters = _extract_best_parameters(sweep)
        strategy = strategy_class(long_only=not args.allow_short, **best_parameters)

        train_result = BacktestEngine(backtest_config, execution_config).run(train_data, strategy)
        test_result = BacktestEngine(backtest_config, execution_config).run(test_data, strategy)

        no_cost_metrics = None
        if research_config.include_cost_free_run:
            no_cost_execution = replace(
                execution_config,
                commission_bps=0.0,
                slippage_bps=0.0,
                spread_bps=0.0,
                fixed_commission=0.0,
                volume_share_slippage_bps=0.0,
            )
            no_cost_result = BacktestEngine(backtest_config, no_cost_execution).run(test_data, strategy)
            no_cost_metrics = no_cost_result.metrics

        robustness = slippage_sensitivity(
            market_data=test_data,
            strategy=strategy,
            backtest_config=backtest_config,
            execution_config=execution_config,
            scenarios_bps=research_config.slippage_scenarios_bps,
        )
        bootstrap_samples, bootstrap_stats, bootstrap_paths = bootstrap_return_paths(
            returns=test_result.equity_curve["returns"],
            iterations=research_config.bootstrap_iterations,
            block_size=research_config.bootstrap_block_size,
            initial_cash=backtest_config.initial_cash,
            seed=backtest_config.random_seed,
        )
        walk_forward_segments, walk_forward_curve, walk_forward_metrics = WalkForwardRunner(
            backtest_config=backtest_config,
            execution_config=execution_config,
        ).run(
            market_data=market_data,
            strategy_class=strategy_class,
            param_grid=strategy_class.default_parameter_grid(),
            train_bars=research_config.walk_forward_train_bars,
            test_bars=research_config.walk_forward_test_bars,
            step_bars=research_config.walk_forward_step_bars,
            long_only=not args.allow_short,
        )
        regime_assignments = infer_market_regimes(test_data, lookback=research_config.regime_lookback)
        regime_summary = analyze_regime_performance(
            equity_curve=test_result.equity_curve,
            benchmark_curve=test_result.benchmark_curve,
            regime_frame=regime_assignments,
        )

        strategy_root = research_config.output_dir / strategy.name
        output_paths.extend(
            [
                Path(path)
                for path in create_strategy_report(train_result, strategy_root / "in_sample", sweep=sweep).values()
            ]
        )
        output_paths.extend(
            [
                Path(path)
                for path in create_strategy_report(
                    test_result,
                    strategy_root / "out_of_sample",
                    robustness=robustness,
                    walk_forward_segments=walk_forward_segments,
                    walk_forward_curve=walk_forward_curve,
                    walk_forward_metrics=walk_forward_metrics,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_stats=bootstrap_stats,
                    bootstrap_paths=bootstrap_paths,
                    regime_summary=regime_summary,
                    regime_assignments=regime_assignments,
                ).values()
            ]
        )
        summary_path = write_research_summary(
            strategy_root / "summary.txt",
            strategy=strategy,
            in_sample=train_result,
            out_of_sample=test_result,
            sweep=sweep,
            robustness=robustness,
            cost_free_metrics=no_cost_metrics,
            walk_forward_segments=walk_forward_segments,
            walk_forward_metrics=walk_forward_metrics,
            bootstrap_stats=bootstrap_stats,
            regime_summary=regime_summary,
        )
        output_paths.append(summary_path)
    return output_paths


def configure_logging() -> None:
    """Configure project logging."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def main() -> None:
    """CLI entry point."""

    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate-sample-data":
        assets = tuple(asset.strip().upper() for asset in args.assets.split(",") if asset.strip())
        write_sample_data(args.output_dir, args.periods, assets)
        LOGGER.info("Sample data written to %s", args.output_dir)
        return

    if args.command == "run":
        outputs = run_research(args)
        LOGGER.info("Generated %d output artifacts under %s", len(outputs), args.output_dir)


if __name__ == "__main__":
    main()
