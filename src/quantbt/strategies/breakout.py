"""Momentum breakout strategy."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quantbt.strategies.base import BaseStrategy, stack_signals
from quantbt.types import MarketData


@dataclass(slots=True)
class MomentumBreakoutStrategy(BaseStrategy):
    """Price breakout strategy with lagged breakout bands to avoid leakage."""

    lookback: int = 50
    exit_lookback: int = 20
    name: str = field(init=False, default="momentum_breakout")

    def generate_signals(self, market_data: MarketData) -> pd.DataFrame:
        signals_by_asset: dict[str, pd.Series] = {}
        for asset in market_data.assets:
            frame = market_data.asset_frame(asset)
            breakout_high = frame["high"].shift(1).rolling(self.lookback).max()
            breakout_low = frame["low"].shift(1).rolling(self.lookback).min()
            exit_high = frame["high"].shift(1).rolling(self.exit_lookback).max()
            exit_low = frame["low"].shift(1).rolling(self.exit_lookback).min()
            signal = pd.Series(0.0, index=frame.index)
            current = 0.0
            for timestamp in frame.index:
                close = frame.at[timestamp, "close"]
                upper = breakout_high.loc[timestamp]
                lower = breakout_low.loc[timestamp]
                trailing_high = exit_high.loc[timestamp]
                trailing_low = exit_low.loc[timestamp]
                if pd.isna(upper) or pd.isna(lower):
                    current = 0.0
                elif close > upper:
                    current = 1.0
                elif close < lower:
                    current = -1.0
                elif current > 0 and close < trailing_low:
                    current = 0.0
                elif current < 0 and close > trailing_high:
                    current = 0.0
                signal.loc[timestamp] = max(current, 0.0) if self.long_only else current
            signals_by_asset[asset] = signal
        return stack_signals(signals_by_asset)

    @classmethod
    def default_parameter_grid(cls) -> list[dict[str, int]]:
        return [
            {"lookback": 20, "exit_lookback": 10},
            {"lookback": 50, "exit_lookback": 20},
            {"lookback": 100, "exit_lookback": 50},
        ]
