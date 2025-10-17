from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Optional, Tuple

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
POSTAL_FILE = DATA_DIR / "oslo_postal_districts.csv"
POSTAL_CODE_RE = re.compile(r"(\d{4})")

BROKER_KEY_SEP = "||"


def extract_postal_code(address: str | None) -> Optional[str]:
    """Return the last 4 digit token in an address string that looks like a postal code."""
    if not address:
        return None
    matches = list(POSTAL_CODE_RE.finditer(str(address)))
    if not matches:
        return None
    for match in reversed(matches):
        candidate = match.group(1)
        # Skip tokens that are immediately followed by more digits (e.g. apartment numbers)
        tail = str(address)[match.end():]
        if tail and tail[0].isdigit():
            continue
        return candidate
    return matches[-1].group(1)


@lru_cache(maxsize=1)
def load_oslo_postal_lookup() -> dict[str, str]:
    """Load the curated postal code → bydel mapping for Oslo."""
    if not POSTAL_FILE.exists():
        return {}
    df = pd.read_csv(POSTAL_FILE, dtype={"postal_code": str, "district": str})
    df["postal_code"] = df["postal_code"].str.zfill(4)
    df = df.dropna(subset=["postal_code", "district"])
    return dict(zip(df["postal_code"], df["district"]))


def most_common_value(series: pd.Series, fallback: str) -> str:
    if series is None or series.empty:
        return fallback
    cleaned = series.dropna()
    if cleaned.empty:
        return fallback
    modes = cleaned.mode()
    if modes.empty:
        return str(cleaned.iloc[0])
    return str(modes.iloc[0])


def _segment_summary_tuple(series: pd.Series) -> Tuple[str, Optional[str]]:
    if series is None or series.empty:
        return "–", None
    counts = series.fillna("(ukjent segment)").value_counts()
    if counts.empty:
        return "–", None
    dominant = counts.index[0]
    top_two = counts.head(2)
    parts = [f"{name} ({count})" for name, count in top_two.items()]
    return ", ".join(parts), dominant


def _location_summary_tuple(series: pd.Series) -> Tuple[str, Optional[str]]:
    if series is None or series.empty:
        return "–", None
    counts = series.fillna("(ukjent bydel)").value_counts()
    if counts.empty:
        return "–", None
    primary = counts.index[0]
    parts = [f"{name} ({count})" for name, count in counts.head(2).items()]
    return ", ".join(parts), primary


def build_broker_ranking(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "broker",
                "chain",
                "city",
                "district",
                "broker_role",
                "total_sales",
                "total_value",
                "avg_price",
                "latest_listing",
                "segment_summary",
                "dominant_segment",
                "location_summary",
                "primary_location",
                "broker_key",
                "high_volume",
            ]
        )

    group_cols = ["broker", "chain"]

    aggregated = (
        df.groupby(group_cols, dropna=False)
        .agg(
            total_sales=("listing_id", "count"),
            total_value=("price", "sum"),
            avg_price=("price", "mean"),
            city=("city", lambda s: most_common_value(s, "(ukjent by)")),
            district=("district", lambda s: most_common_value(s, "(ukjent bydel)")),
            broker_role=("broker_role", lambda s: most_common_value(s, "(ukjent rolle)")),
            latest_listing=("published_dt", "max"),
        )
        .reset_index()
    )

    segment_data = (
        df.groupby(group_cols, dropna=False)["property_type"]
        .apply(_segment_summary_tuple)
        .reset_index(name="segment_tuple")
    )
    if not segment_data.empty:
        segment_data[["segment_summary", "dominant_segment"]] = pd.DataFrame(
            segment_data["segment_tuple"].tolist(), index=segment_data.index
        )
        segment_data = segment_data.drop(columns=["segment_tuple"])
    else:
        segment_data = pd.DataFrame(columns=group_cols + ["segment_summary", "dominant_segment"])

    location_data = (
        df.groupby(group_cols, dropna=False)["district"]
        .apply(_location_summary_tuple)
        .reset_index(name="location_tuple")
    )
    if not location_data.empty:
        location_data[["location_summary", "primary_location"]] = pd.DataFrame(
            location_data["location_tuple"].tolist(), index=location_data.index
        )
        location_data = location_data.drop(columns=["location_tuple"])
    else:
        location_data = pd.DataFrame(columns=group_cols + ["location_summary", "primary_location"])

    ranking = aggregated.merge(segment_data, on=group_cols, how="left")
    ranking = ranking.merge(location_data, on=group_cols, how="left")

    ranking["total_value"] = ranking["total_value"].fillna(0.0)
    ranking["avg_price"] = ranking["avg_price"].fillna(0.0)
    ranking["segment_summary"] = ranking["segment_summary"].fillna("–")
    ranking["dominant_segment"] = ranking["dominant_segment"].fillna("(ukjent segment)")
    ranking["location_summary"] = ranking["location_summary"].fillna("–")
    ranking["primary_location"] = ranking["primary_location"].fillna("(ukjent bydel)")

    ranking["broker_key"] = ranking.apply(
        lambda row: f"{row['broker']}{BROKER_KEY_SEP}{row['chain']}", axis=1
    )

    if not ranking.empty:
        if len(ranking) >= 5:
            threshold = ranking["total_value"].quantile(0.9)
        else:
            threshold = ranking["total_value"].median()
        ranking["high_volume"] = ranking["total_value"] >= threshold
    else:
        ranking["high_volume"] = False

    return ranking
