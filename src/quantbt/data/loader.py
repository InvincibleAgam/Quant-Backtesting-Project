"""Historical market data loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from quantbt.types import MarketData


def _standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {column: column.lower().strip() for column in frame.columns}
    standardized = frame.rename(columns=rename_map)
    alias_map = {
        "adj close": "close",
        "datetime": "timestamp",
        "date": "timestamp",
    }
    standardized = standardized.rename(columns=alias_map)
    return standardized


def _validate_asset_frame(frame: pd.DataFrame, asset: str) -> pd.DataFrame:
    required_columns = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{asset} missing required columns: {missing}")
    validated = frame.loc[:, ["open", "high", "low", "close", "volume"]].copy()
    if validated.index.has_duplicates:
        raise ValueError(f"{asset} contains duplicate timestamps")
    if not validated.index.is_monotonic_increasing:
        validated = validated.sort_index()
    if validated.isna().any().any():
        raise ValueError(f"{asset} contains missing OHLCV values")
    return validated


def _combine_asset_frames(asset_frames: dict[str, pd.DataFrame]) -> MarketData:
    stacked_frames: list[pd.DataFrame] = []
    for asset, frame in asset_frames.items():
        local = frame.copy()
        local["asset"] = asset
        local.index.name = "timestamp"
        stacked_frames.append(local.reset_index().set_index(["timestamp", "asset"]))
    combined = pd.concat(stacked_frames).sort_index()
    return MarketData(combined)


class CSVDataLoader:
    """Load one or more CSV files into canonical multi-asset market data."""

    def load_file(
        self,
        path: str | Path,
        asset: str | None = None,
        timestamp_column: str = "timestamp",
    ) -> MarketData:
        """Load a single CSV file."""

        csv_path = Path(path)
        frame = pd.read_csv(csv_path)
        frame = _standardize_columns(frame)
        if timestamp_column not in frame.columns:
            timestamp_column = "timestamp"
        if timestamp_column not in frame.columns:
            raise ValueError(f"{csv_path} does not contain a timestamp/date column")
        inferred_asset = asset or csv_path.stem.upper()
        frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=False)
        frame = frame.set_index(timestamp_column)
        validated = _validate_asset_frame(frame, inferred_asset)
        return _combine_asset_frames({inferred_asset: validated})

    def load_directory(self, directory: str | Path, pattern: str = "*.csv") -> MarketData:
        """Load multiple asset CSVs from a directory."""

        data_dir = Path(directory)
        asset_frames: dict[str, pd.DataFrame] = {}
        for file_path in sorted(data_dir.glob(pattern)):
            loaded = self.load_file(file_path)
            asset = loaded.assets[0]
            asset_frames[asset] = loaded.asset_frame(asset)
        if not asset_frames:
            raise FileNotFoundError(f"no CSV files found in {data_dir}")
        return _combine_asset_frames(asset_frames)


class YFinanceDataLoader:
    """Download OHLCV data from Yahoo Finance."""

    def load(
        self,
        tickers: list[str],
        start: str,
        end: str,
        interval: str = "1d",
        auto_adjust: bool = False,
    ) -> MarketData:
        """Fetch market data for one or more tickers."""

        download = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
            group_by="ticker",
            threads=False,
        )
        asset_frames: dict[str, pd.DataFrame] = {}
        if isinstance(download.columns, pd.MultiIndex):
            for ticker in tickers:
                local = download[ticker].copy()
                local = _standardize_columns(local)
                asset_frames[ticker] = _validate_asset_frame(local, ticker)
        else:
            ticker = tickers[0]
            local = _standardize_columns(download.copy())
            asset_frames[ticker] = _validate_asset_frame(local, ticker)
        return _combine_asset_frames(asset_frames)
