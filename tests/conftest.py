"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from quantbt.types import MarketData


@pytest.fixture()
def single_asset_market_data() -> MarketData:
    """Simple deterministic OHLCV data for unit tests."""

    timestamps = pd.bdate_range("2024-01-01", periods=8, name="timestamp")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "asset": "TEST",
            "open": [100, 101, 102, 103, 104, 105, 106, 107],
            "high": [101, 102, 103, 104, 105, 106, 107, 108],
            "low": [99, 100, 101, 102, 103, 104, 105, 106],
            "close": [100, 101, 102, 103, 120, 119, 118, 117],
            "volume": [10_000] * 8,
        }
    ).set_index(["timestamp", "asset"])
    return MarketData(frame)
