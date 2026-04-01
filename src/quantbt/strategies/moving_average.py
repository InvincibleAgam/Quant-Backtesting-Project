"""Moving average crossover strategy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantbt.strategies.base import BaseStrategy, stack_signals
from quantbt.types import MarketData


@dataclass(slots=True)
class MovingAverageCrossoverStrategy(BaseStrategy):
    """Trend-following strategy using fast and slow moving averages."""

    fast_window: int = 20
    slow_window: int = 100
    name: str = field(init=False, default="moving_average_crossover")

    def __post_init__(self) -> None:
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be smaller than slow_window")

    def generate_signals(self, market_data: MarketData) -> pd.DataFrame:
        signals_by_asset: dict[str, pd.Series] = {}
        for asset in market_data.assets:
            frame = market_data.asset_frame(asset)
            fast = frame["close"].rolling(self.fast_window).mean()
            slow = frame["close"].rolling(self.slow_window).mean()
            signal = pd.Series(0.0, index=frame.index)
            valid = slow.notna()
            signal.loc[valid] = np.where(fast.loc[valid] > slow.loc[valid], 1.0, -1.0)
            if self.long_only:
                signal = signal.clip(lower=0.0)
            signals_by_asset[asset] = signal
        return stack_signals(signals_by_asset)

    @classmethod
    def default_parameter_grid(cls) -> list[dict[str, int]]:
        return [
            {"fast_window": 10, "slow_window": 50},
            {"fast_window": 20, "slow_window": 100},
            {"fast_window": 30, "slow_window": 150},
            {"fast_window": 50, "slow_window": 200},
        ]
