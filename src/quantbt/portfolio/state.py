"""Portfolio and position bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantbt.types import Fill, TradeRecord


@dataclass(slots=True)
class PositionState:
    """Aggregate position state for a single asset."""

    asset: str
    quantity: int = 0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    entry_timestamp: pd.Timestamp | None = None
    entry_bar_index: int | None = None

    def market_value(self, mark_price: float) -> float:
        return self.quantity * mark_price

    def unrealized_pnl(self, mark_price: float) -> float:
        if self.quantity == 0:
            return 0.0
        return self.quantity * (mark_price - self.avg_price)

    def apply_fill(self, fill: Fill, bar_index: int) -> list[TradeRecord]:
        """Update position state and return any closed trade records."""

        trades: list[TradeRecord] = []
        signed_qty = fill.quantity
        if self.quantity == 0:
            self.quantity = signed_qty
            self.avg_price = fill.fill_price
            self.entry_timestamp = fill.timestamp
            self.entry_bar_index = bar_index
            return trades

        current_direction = int(np.sign(self.quantity))
        fill_direction = int(np.sign(signed_qty))
        if current_direction == fill_direction:
            total_quantity = abs(self.quantity) + abs(signed_qty)
            self.avg_price = (
                abs(self.quantity) * self.avg_price + abs(signed_qty) * fill.fill_price
            ) / total_quantity
            self.quantity += signed_qty
            return trades

        closing_quantity = min(abs(self.quantity), abs(signed_qty))
        pnl = closing_quantity * (fill.fill_price - self.avg_price) * current_direction
        self.realized_pnl += pnl

        is_full_close = abs(signed_qty) >= abs(self.quantity)
        if self.entry_timestamp is not None and self.entry_bar_index is not None:
            entry_notional = max(abs(self.avg_price * closing_quantity), 1e-12)
            trades.append(
                TradeRecord(
                    asset=self.asset,
                    entry_timestamp=self.entry_timestamp,
                    exit_timestamp=fill.timestamp,
                    direction=current_direction,
                    entry_price=self.avg_price,
                    exit_price=fill.fill_price,
                    quantity=closing_quantity,
                    pnl=pnl,
                    return_pct=pnl / entry_notional,
                    holding_period_bars=max(bar_index - self.entry_bar_index, 1),
                )
            )

        self.quantity += signed_qty
        if self.quantity == 0:
            self.avg_price = 0.0
            self.entry_timestamp = None
            self.entry_bar_index = None
            return trades

        if is_full_close:
            self.avg_price = fill.fill_price
            self.entry_timestamp = fill.timestamp
            self.entry_bar_index = bar_index
        return trades


@dataclass(slots=True)
class PortfolioState:
    """Track cash, positions, fills, trades, and the equity curve."""

    initial_cash: float
    cash: float = field(init=False)
    positions: dict[str, PositionState] = field(default_factory=dict)
    fills: list[dict[str, object]] = field(default_factory=list)
    trades: list[dict[str, object]] = field(default_factory=list)
    equity_records: list[dict[str, float | pd.Timestamp]] = field(default_factory=list)
    last_equity: float | None = None

    def __post_init__(self) -> None:
        self.cash = self.initial_cash

    def get_position(self, asset: str) -> PositionState:
        """Fetch or create a position state."""

        if asset not in self.positions:
            self.positions[asset] = PositionState(asset=asset)
        return self.positions[asset]

    def apply_fill(self, fill: Fill, bar_index: int) -> None:
        """Apply a fill to cash and position state."""

        self.cash -= fill.notional
        self.cash -= fill.commission
        position = self.get_position(fill.asset)
        trades = position.apply_fill(fill, bar_index)
        self.fills.append(
            {
                "timestamp": fill.timestamp,
                "asset": fill.asset,
                "quantity": fill.quantity,
                "fill_price": fill.fill_price,
                "commission": fill.commission,
                "slippage_cost": fill.slippage_cost,
                "spread_cost": fill.spread_cost,
                "notional": fill.notional,
                "reason": fill.reason,
            }
        )
        for trade in trades:
            self.trades.append(
                {
                    "asset": trade.asset,
                    "entry_timestamp": trade.entry_timestamp,
                    "exit_timestamp": trade.exit_timestamp,
                    "direction": trade.direction,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "quantity": trade.quantity,
                    "pnl": trade.pnl,
                    "return_pct": trade.return_pct,
                    "holding_period_bars": trade.holding_period_bars,
                }
            )

    def position_quantities(self) -> dict[str, int]:
        """Return signed share quantities for all current positions."""

        return {asset: position.quantity for asset, position in self.positions.items()}

    def snapshot(self, timestamp: pd.Timestamp, bars: pd.DataFrame, turnover_notional: float) -> None:
        """Record an end-of-bar portfolio snapshot using bar close prices."""

        market_value = 0.0
        unrealized = 0.0
        realized = sum(position.realized_pnl for position in self.positions.values())
        gross_notional = 0.0
        net_notional = 0.0
        for asset, row in bars.iterrows():
            position = self.get_position(asset)
            close_price = float(row["close"])
            value = position.market_value(close_price)
            market_value += value
            unrealized += position.unrealized_pnl(close_price)
            gross_notional += abs(value)
            net_notional += value

        equity = self.cash + market_value
        returns = 0.0 if self.last_equity in (None, 0.0) else equity / self.last_equity - 1.0
        gross_exposure = 0.0 if equity == 0 else gross_notional / equity
        net_exposure = 0.0 if equity == 0 else net_notional / equity
        turnover = 0.0 if equity == 0 else turnover_notional / equity

        self.equity_records.append(
            {
                "timestamp": timestamp,
                "cash": self.cash,
                "market_value": market_value,
                "equity": equity,
                "realized_pnl": realized,
                "unrealized_pnl": unrealized,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "returns": returns,
                "turnover": turnover,
            }
        )
        self.last_equity = equity

    def equity_curve(self) -> pd.DataFrame:
        """Return the portfolio equity curve."""

        if not self.equity_records:
            return pd.DataFrame(columns=["equity", "returns"])
        curve = pd.DataFrame(self.equity_records).set_index("timestamp")
        curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1.0
        return curve

    def trades_frame(self) -> pd.DataFrame:
        """Return closed trades as a DataFrame."""

        return pd.DataFrame(self.trades)

    def fills_frame(self) -> pd.DataFrame:
        """Return raw fills as a DataFrame."""

        return pd.DataFrame(self.fills)
