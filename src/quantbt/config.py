"""Configuration models for the backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ExecutionConfig:
    """Execution assumptions for simulated fills."""

    price_source: str = "next_open"
    commission_bps: float = 5.0
    slippage_bps: float = 3.0
    spread_bps: float = 2.0
    fixed_commission: float = 0.0
    volume_limit: float = 0.15
    volume_share_slippage_bps: float = 10.0
    allow_short: bool = True
    max_gross_leverage: float = 1.0
    rebalance_buffer: float = 0.0
    max_asset_weight: float = 0.35
    min_cash_buffer: float = 0.02
    max_active_positions: int | None = None
    allocation_scheme: str = "equal_weight"
    volatility_lookback: int = 20

    def __post_init__(self) -> None:
        valid_sources = {"next_open", "next_close"}
        valid_schemes = {"equal_weight", "inverse_volatility"}
        if self.price_source not in valid_sources:
            raise ValueError(f"price_source must be one of {valid_sources}")
        if self.allocation_scheme not in valid_schemes:
            raise ValueError(f"allocation_scheme must be one of {valid_schemes}")
        if self.volume_limit <= 0:
            raise ValueError("volume_limit must be positive")
        if self.max_gross_leverage <= 0:
            raise ValueError("max_gross_leverage must be positive")
        if not 0 < self.max_asset_weight <= self.max_gross_leverage:
            raise ValueError("max_asset_weight must be positive and not exceed max_gross_leverage")
        if not 0 <= self.min_cash_buffer < 1:
            raise ValueError("min_cash_buffer must be in [0, 1)")
        if self.max_active_positions is not None and self.max_active_positions <= 0:
            raise ValueError("max_active_positions must be positive when provided")
        if self.volatility_lookback <= 0:
            raise ValueError("volatility_lookback must be positive")


@dataclass(slots=True)
class BacktestConfig:
    """Portfolio and simulation settings."""

    initial_cash: float = 1_000_000.0
    annual_trading_days: int = 252
    train_fraction: float = 0.6
    random_seed: int = 7
    benchmark_symbol: str | None = None

    def __post_init__(self) -> None:
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must lie strictly between 0 and 1")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")


@dataclass(slots=True)
class ResearchConfig:
    """High-level runtime settings for CLI execution."""

    output_dir: Path = Path("reports")
    data_dir: Path = Path("data")
    create_plots: bool = True
    save_csv: bool = True
    save_summary: bool = True
    include_cost_free_run: bool = True
    slippage_scenarios_bps: tuple[float, ...] = (0.0, 3.0, 10.0)
    sample_assets: tuple[str, ...] = ("SPY", "QQQ", "IWM")
    bootstrap_iterations: int = 500
    bootstrap_block_size: int = 5
    walk_forward_train_bars: int = 126
    walk_forward_test_bars: int = 63
    walk_forward_step_bars: int = 63
    regime_lookback: int = 63


DEFAULT_BACKTEST_CONFIG = BacktestConfig()
DEFAULT_EXECUTION_CONFIG = ExecutionConfig()
