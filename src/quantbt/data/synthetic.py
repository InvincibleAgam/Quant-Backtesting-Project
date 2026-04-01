"""Deterministic synthetic data generation for demos and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantbt.types import MarketData


def generate_sample_ohlcv(
    assets: tuple[str, ...] = ("SPY", "QQQ", "IWM"),
    periods: int = 504,
    seed: int = 7,
    start: str = "2020-01-01",
) -> MarketData:
    """Generate multi-asset OHLCV data with trend and mean-reverting components."""

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=periods, name="timestamp")
    frames: list[pd.DataFrame] = []
    for asset_idx, asset in enumerate(assets):
        drift = 0.0003 + asset_idx * 0.00005
        vol = 0.012 + asset_idx * 0.002
        shocks = rng.normal(drift, vol, size=periods)
        cycle = 0.003 * np.sin(np.linspace(0, 8 * np.pi, periods) + asset_idx)
        returns = shocks + cycle
        close = 100.0 * np.exp(np.cumsum(returns))
        open_ = np.concatenate([[close[0]], close[:-1]]) * (1.0 + rng.normal(0.0, 0.002, size=periods))
        high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.01, size=periods))
        low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.01, size=periods))
        volume = rng.integers(500_000, 3_000_000, size=periods)
        frame = pd.DataFrame(
            {
                "timestamp": dates,
                "asset": asset,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        ).set_index(["timestamp", "asset"])
        frames.append(frame)
    return MarketData(pd.concat(frames).sort_index())
