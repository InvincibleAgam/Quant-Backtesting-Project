"""Mean reversion strategy using rolling z-scores."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quantbt.strategies.base import BaseStrategy, stack_signals
from quantbt.types import MarketData


@dataclass(slots=True)
class MeanReversionStrategy(BaseStrategy):
    """Bollinger-style mean reversion with stateful exits."""

    lookback: int = 20
    entry_z: float = 1.5
    exit_z: float = 0.5
    name: str = field(init=False, default="mean_reversion")

    def generate_signals(self, market_data: MarketData) -> pd.DataFrame:
        signals_by_asset: dict[str, pd.Series] = {}
        for asset in market_data.assets:
            frame = market_data.asset_frame(asset)
            rolling_mean = frame["close"].rolling(self.lookback).mean()
            rolling_std = frame["close"].rolling(self.lookback).std(ddof=0)
            zscore = (frame["close"] - rolling_mean) / rolling_std.replace(0.0, pd.NA)
            signal = pd.Series(0.0, index=frame.index)
            current = 0.0
            for timestamp in frame.index:
                z = zscore.loc[timestamp]
                if pd.isna(z):
                    current = 0.0
                elif z <= -self.entry_z:
                    current = 1.0
                elif z >= self.entry_z:
                    current = -1.0
                elif abs(float(z)) <= self.exit_z:
                    current = 0.0
                signal.loc[timestamp] = max(current, 0.0) if self.long_only else current
            signals_by_asset[asset] = signal
        return stack_signals(signals_by_asset)

    @classmethod
    def default_parameter_grid(cls) -> list[dict[str, float]]:
        return [
            {"lookback": 10, "entry_z": 1.0, "exit_z": 0.25},
            {"lookback": 20, "entry_z": 1.5, "exit_z": 0.5},
            {"lookback": 30, "entry_z": 2.0, "exit_z": 0.75},
        ]
