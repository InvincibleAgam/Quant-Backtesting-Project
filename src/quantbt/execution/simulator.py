"""Execution simulator with explicit cost and slippage assumptions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantbt.config import ExecutionConfig
from quantbt.types import Fill, Order


@dataclass(slots=True)
class ExecutionOutcome:
    """Executed fill and any residual quantity left unfilled."""

    fill: Fill | None
    unfilled_quantity: int = 0


class PositionSizer:
    """Convert strategy signals into target share counts."""

    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config

    def target_shares(
        self,
        signals: pd.Series,
        prices: pd.Series,
        equity: float,
        price_history: pd.DataFrame | None = None,
    ) -> dict[str, int]:
        """Allocate equal gross risk budget across active signals."""

        clean_signals = signals.fillna(0.0).copy()
        if not self.config.allow_short:
            clean_signals = clean_signals.clip(lower=0.0)
        active = clean_signals[clean_signals.abs() > self.config.rebalance_buffer]
        if active.empty:
            return {asset: 0 for asset in clean_signals.index}

        active = self._limit_active_positions(active, price_history)
        gross_budget = equity * self.config.max_gross_leverage * (1.0 - self.config.min_cash_buffer)
        weights = self._determine_weights(active, price_history)
        targets: dict[str, int] = {}
        for asset, signal in clean_signals.items():
            price = float(prices.get(asset, np.nan))
            if np.isnan(price) or price <= 0.0 or abs(signal) <= self.config.rebalance_buffer:
                targets[asset] = 0
                continue
            target_notional = gross_budget * weights.get(asset, 0.0) * float(np.sign(signal))
            shares = int(np.floor(abs(target_notional) / price))
            targets[asset] = int(np.sign(target_notional) * shares)
        return targets

    def _limit_active_positions(self, active: pd.Series, price_history: pd.DataFrame | None) -> pd.Series:
        """Limit the number of simultaneously active positions."""

        if self.config.max_active_positions is None or len(active) <= self.config.max_active_positions:
            return active
        ranking = active.abs().astype(float)
        if price_history is not None and not price_history.empty:
            asset_vol = self._recent_volatility(price_history, active.index)
            quality_score = ranking / asset_vol.replace(0.0, np.nan)
            ranking = quality_score.fillna(ranking)
        selected_assets = ranking.sort_values(ascending=False).head(self.config.max_active_positions).index
        return active.loc[selected_assets]

    def _determine_weights(self, active: pd.Series, price_history: pd.DataFrame | None) -> pd.Series:
        """Compute capped target weights for active signals."""

        if self.config.allocation_scheme == "inverse_volatility" and price_history is not None and not price_history.empty:
            asset_vol = self._recent_volatility(price_history, active.index)
            inverse_vol = 1.0 / asset_vol.replace(0.0, np.nan)
            raw_weights = inverse_vol.reindex(active.index).fillna(0.0)
            if raw_weights.sum() <= 0:
                raw_weights = pd.Series(1.0, index=active.index)
        else:
            raw_weights = pd.Series(1.0, index=active.index)
        normalized = raw_weights / raw_weights.sum()
        return self._apply_weight_cap(normalized)

    def _recent_volatility(self, price_history: pd.DataFrame, assets: pd.Index) -> pd.Series:
        """Estimate recent asset volatility from close price history."""

        returns = price_history.reindex(columns=list(assets)).pct_change().tail(self.config.volatility_lookback)
        vol = returns.std(ddof=0).replace(0.0, np.nan)
        return vol.fillna(vol.median()).fillna(1.0)

    def _apply_weight_cap(self, weights: pd.Series) -> pd.Series:
        """Cap single-name weights and redistribute residual capital."""

        capped = weights.astype(float).copy()
        max_weight = self.config.max_asset_weight / self.config.max_gross_leverage
        for _ in range(len(capped) + 1):
            overweight = capped[capped > max_weight]
            if overweight.empty:
                break
            excess = float((overweight - max_weight).sum())
            capped.loc[overweight.index] = max_weight
            underweight = capped[capped < max_weight]
            if excess <= 0 or underweight.empty:
                break
            redistribution = underweight / underweight.sum()
            capped.loc[underweight.index] = capped.loc[underweight.index] + redistribution * excess
        return capped.clip(lower=0.0, upper=max_weight)


class ExecutionSimulator:
    """Simulate market order fills at future bars."""

    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config

    def execute_order(self, order: Order, bar: pd.Series, timestamp: pd.Timestamp) -> ExecutionOutcome:
        """Simulate a market order fill using next bar open or close."""

        raw_price = float(bar["open"] if self.config.price_source == "next_open" else bar["close"])
        volume = max(float(bar["volume"]), 1.0)
        max_fill_quantity = max(1, int(volume * self.config.volume_limit))
        fill_quantity = int(np.sign(order.quantity) * min(abs(order.quantity), max_fill_quantity))
        unfilled = int(order.quantity - fill_quantity)
        if fill_quantity == 0:
            return ExecutionOutcome(fill=None, unfilled_quantity=order.quantity)

        volume_share = min(abs(fill_quantity) / volume, 1.0)
        slippage_bps = self.config.slippage_bps + volume_share * self.config.volume_share_slippage_bps
        half_spread_bps = self.config.spread_bps / 2.0

        execution_multiplier = 1.0
        if fill_quantity > 0:
            execution_multiplier += (slippage_bps + half_spread_bps) / 10_000.0
        else:
            execution_multiplier -= (slippage_bps + half_spread_bps) / 10_000.0

        fill_price = raw_price * execution_multiplier
        notional = fill_quantity * fill_price
        commission = abs(notional) * self.config.commission_bps / 10_000.0 + self.config.fixed_commission
        slippage_cost = abs(fill_quantity) * raw_price * slippage_bps / 10_000.0
        spread_cost = abs(fill_quantity) * raw_price * half_spread_bps / 10_000.0
        fill = Fill(
            timestamp=timestamp,
            asset=order.asset,
            quantity=fill_quantity,
            fill_price=fill_price,
            commission=commission,
            slippage_cost=slippage_cost,
            spread_cost=spread_cost,
            notional=notional,
            reason=order.reason,
        )
        return ExecutionOutcome(fill=fill, unfilled_quantity=unfilled)
