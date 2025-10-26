from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from megler_monitor_poc import (  # type: ignore[import-untyped]
    collect_dnb,
    collect_hjem,
    save_csv,
    save_snapshot,
    sum_per_broker,
)

OUT_DIR = (Path(__file__).resolve().parent / "out").expanduser()


@dataclass
class RefreshResult:
    hjem_rows: int
    dnb_rows: int
    combined_rows: int
    agg_rows: int


def _ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(name: str) -> pd.DataFrame:
    """Return the CSV from `out/` if it exists, otherwise an empty frame."""
    _ensure_out_dir()
    path = OUT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_all_listings() -> pd.DataFrame:
    return load_csv("all_listings.csv")


def load_broker_totals() -> pd.DataFrame:
    return load_csv("agg_sum_per_broker.csv")


def format_compact_number(value: Optional[float]) -> str:
    """Format large values using Lovable-esque compact suffixes."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "–"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "–"

    abs_val = abs(numeric)
    if abs_val >= 1_000_000_000:
        return f"{numeric / 1_000_000_000:.3f}b"
    if abs_val >= 1_000_000:
        return f"{numeric / 1_000_000:.3f}m"
    if abs_val >= 1_000:
        return f"{numeric / 1_000:.1f}k"
    return f"{int(numeric)}"


def refresh_data(include_hjem: bool = True, include_dnb: bool = True) -> RefreshResult:
    """Re-fetch raw data and regenerate the aggregated CSV outputs."""
    _ensure_out_dir()

    data_frames: list[pd.DataFrame] = []

    hjem_df = pd.DataFrame()
    if include_hjem:
        hjem_df = collect_hjem()
        save_csv(hjem_df, "hjem_listings.csv")
        save_snapshot(hjem_df, "hjem_listings.csv")
        data_frames.append(hjem_df)

    dnb_df = pd.DataFrame()
    if include_dnb:
        dnb_df = collect_dnb()
        save_csv(dnb_df, "dnb_listings.csv")
        save_snapshot(dnb_df, "dnb_listings.csv")
        data_frames.append(dnb_df)

    combined = pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()
    save_csv(combined, "all_listings.csv")
    save_snapshot(combined, "all_listings.csv")

    agg = sum_per_broker(combined)
    save_csv(agg, "agg_sum_per_broker.csv")
    save_snapshot(agg, "agg_sum_per_broker.csv")

    return RefreshResult(
        hjem_rows=len(hjem_df),
        dnb_rows=len(dnb_df),
        combined_rows=len(combined),
        agg_rows=len(agg),
    )
