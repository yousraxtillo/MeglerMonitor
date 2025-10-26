from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mm_utils import BROKER_KEY_SEP, build_broker_ranking

OUT_DIR = (Path(__file__).resolve().parent / "out").expanduser()
PRICE_BANDS = [0, 5_000_000, 10_000_000, 15_000_000, 20_000_000, float("inf")]
PRICE_BAND_LABELS = ["0–5 mill", "5–10 mill", "10–15 mill", "15–20 mill", "20+ mill"]
STATUS_LABELS = {
    0: "unknown",
    1: "coming",
    2: "available",
    3: "sold",
    4: "reserved",
    5: "inactive",
    99: "archived",
}
COMMISSION_RATE = 0.0125


@dataclass(frozen=True)
class FilterParams:
    city: Optional[str] = None
    districts: Tuple[str, ...] = ()
    chains: Tuple[str, ...] = ()
    chain_keyword: str = ""
    roles: Tuple[str, ...] = ()
    segments: Tuple[str, ...] = ()
    sources: Tuple[str, ...] = ()
    search: str = ""
    period: str = "Alle"
    min_sales: int = 0
    sort_by: str = "total_value"

    def as_dict(self) -> Dict[str, object]:
        return {
            "city": self.city,
            "districts": list(self.districts),
            "chains": list(self.chains),
            "chain_keyword": self.chain_keyword,
            "roles": list(self.roles),
            "segments": list(self.segments),
            "sources": list(self.sources),
            "search": self.search,
            "period": self.period,
            "min_sales": self.min_sales,
            "sort_by": self.sort_by,
        }

    def validated(self) -> "FilterParams":
        allowed_sort = {"total_value", "total_sales", "avg_price", "latest_listing"}
        sort = self.sort_by if self.sort_by in allowed_sort else "total_value"
        allowed_period = {"Alle", "Siste 30 dager", "Siste 12 mnd", "Dette året"}
        period = self.period if self.period in allowed_period else "Alle"
        return FilterParams(
            city=self.city,
            districts=self.districts,
            chains=self.chains,
            chain_keyword=self.chain_keyword or "",
            roles=self.roles,
            segments=self.segments,
            sources=self.sources,
            search=self.search or "",
            period=period,
            min_sales=max(0, int(self.min_sales)),
            sort_by=sort,
        )

_TOKEN_REPLACEMENTS = {
    "ø": "o",
    "å": "a",
    "æ": "ae",
    "é": "e",
    "ü": "u",
    "ö": "o",
    "ä": "a",
}

DISTRICT_KEYWORDS = {
    "gronland": "Grønland",
    "toy en": "Tøyen",
    "tøyen": "Tøyen",
    "bjorvika": "Bjørvika",
    "ensjo": "Ensjø",
    "grunerlokka": "Grünerløkka",
    "grünerløkka": "Grünerløkka",
    "sagene": "Sagene",
    "st hanshaugen": "St. Hanshaugen",
    "majorstuen": "Majorstuen",
    "frogner": "Frogner",
    "bjerke": "Bjerke",
    "loren": "Løren",
    "løren": "Løren",
    "hasle": "Hasle",
    "grefsen": "Grefsen",
    "lambertseter": "Lambertseter",
    "bjorndal": "Bjørndal",
    "holtet": "Holtet",
    "ulven": "Ulven",
    "ulvenbyen": "Ulven",
    "opp sal": "Oppsal",
    "oppsal": "Oppsal",
    "malerhaugen": "Malerhaugen",
    "eternittkollen": "Eternittkollen",
    "skoyen": "Skøyen",
    "stovner": "Stovner",
    "alna": "Alna",
    "nordstrand": "Nordstrand",
    "ostensjo": "Østensjø",
    "østensjø": "Østensjø",
    "sinsen": "Sinsen",
    "ryen": "Ryen",
    "byen": "Sentrum",
    "gamle oslo": "Gamle Oslo",
}

POSTAL_DISTRICT_BANDS = [
    (180, 199, "Grønland"),
    (200, 299, "Frogner"),
    (350, 369, "St. Hanshaugen"),
    (400, 489, "Sagene"),
    (500, 579, "Grünerløkka"),
    (580, 599, "Gamle Oslo"),
    (600, 699, "Oslo sør"),
    (700, 799, "Oslo vest"),
    (800, 899, "Oslo nord"),
    (900, 999, "Oslo nordøst"),
]

POSTAL_CODE_RE = re.compile(r"(\d{4})")


def _normalize_token(value: str | None) -> str:
    if not value:
        return ""
    lowered = value.lower()
    for src, dst in _TOKEN_REPLACEMENTS.items():
        lowered = lowered.replace(src, dst)
    lowered = re.sub(r"[^a-z0-9 ]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def extract_postal_code(address: str | None) -> Optional[str]:
    if not address:
        return None
    match = POSTAL_CODE_RE.search(str(address))
    return match.group(1) if match else None


def normalize_case(value: str | float | None) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    lowered = text.lower()
    if not text or lowered in {"none", "nan", "null"}:
        return None
    return text.title() if text.upper() == text else text


def clean_timestamp(value: str | float | None) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "nat", "null"}:
        return None
    return text


def infer_city_from_address(address: str | None) -> Optional[str]:
    if address is None or pd.isna(address):
        return None
    parts = [p.strip() for p in str(address).split(",") if p and p.strip()]
    if not parts:
        return None
    street = parts[0]
    postal_tokens = {p.strip() for p in parts[1:] if p.replace(" ", "").isdigit()}

    candidates = []
    if len(parts) >= 3:
        candidates.append(parts[2])
    if len(parts) > 3:
        candidates.extend(parts[3:])
    candidates.extend(parts[1:])

    seen: set[str] = set()
    for candidate in candidates:
        cand = candidate.strip()
        if not cand or cand in seen:
            continue
        seen.add(cand)
        if cand == street or cand in postal_tokens:
            continue
        if cand.lower() in {"norge"}:
            continue
        if cand.replace(" ", "").isdigit():
            continue
        normalized = normalize_case(cand)
        if normalized:
            return normalized
    return None


def infer_district(city: str | None, address: str | None, chain: str | None) -> Optional[str]:
    if not city or pd.isna(city):
        return None
    city_norm = _normalize_token(city)
    if city_norm != "oslo":
        return None

    tokens: set[str] = set()
    for raw in (address, chain):
        if not raw or pd.isna(raw):
            continue
        text = str(raw)
        tokens.add(_normalize_token(text))
        for part in re.split(r"[\s,;/\\-]", text):
            norm = _normalize_token(part)
            if norm:
                tokens.add(norm)

    for token in sorted(tokens, key=len, reverse=True):
        if token in DISTRICT_KEYWORDS:
            return DISTRICT_KEYWORDS[token]

    postal_code = extract_postal_code(address)
    if postal_code:
        try:
            postal_int = int(postal_code)
        except ValueError:
            postal_int = None
        if postal_int is not None:
            for low, high, label in POSTAL_DISTRICT_BANDS:
                if low <= postal_int <= high:
                    return label
    return None


def normalize_status(value) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.isdigit():
            return STATUS_LABELS.get(int(stripped), stripped)
        try:
            as_int = int(float(stripped))
            return STATUS_LABELS.get(as_int, stripped)
        except ValueError:
            return stripped
    try:
        as_int = int(value)
        return STATUS_LABELS.get(as_int, str(value))
    except (TypeError, ValueError):
        return str(value)


def to_dt_safe(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NaT": pd.NA})
    return pd.to_datetime(cleaned, errors="coerce", utc=True)


def prepare_dataframe(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    if out.empty:
        return out

    if "price" in out.columns:
        out["price"] = pd.to_numeric(out["price"], errors="coerce")
    if "property_type" in out.columns:
        out["property_type"] = out["property_type"].fillna("(ukjent boligtype)")
    else:
        out["property_type"] = "(ukjent boligtype)"
    if "broker_role" in out.columns:
        out["broker_role"] = out["broker_role"].fillna("(ukjent rolle)")
    else:
        out["broker_role"] = "(ukjent rolle)"

    if "price" in out.columns:
        price_band = pd.cut(
            out["price"],
            bins=PRICE_BANDS,
            labels=PRICE_BAND_LABELS,
            include_lowest=True,
            right=False,
        )
        price_band = price_band.cat.add_categories(["(ukjent prissjikt)"]).fillna("(ukjent prissjikt)")
        out["price_band"] = price_band.astype(str)
    else:
        out["price_band"] = "(ukjent prissjikt)"

    if "city" in out.columns:
        out["city"] = out["city"].apply(normalize_case)
        if "address" in out.columns:
            missing_city = out["city"].isna() | out["city"].astype(str).str.strip().eq("")
            out.loc[missing_city, "city"] = out.loc[missing_city, "address"].apply(infer_city_from_address)
        out["city"] = out["city"].apply(normalize_case)

    if "address" in out.columns:
        out["postal_code"] = out["address"].apply(extract_postal_code)
    else:
        out["postal_code"] = None

    if {"city", "address", "chain"}.issubset(out.columns):
        out["district"] = out.apply(
            lambda row: infer_district(row.get("city"), row.get("address"), row.get("chain")),
            axis=1,
        )
    else:
        out["district"] = None
    if "district" in out.columns:
        if "city" in out.columns:
            district_blank = out["district"].isna() | (out["district"].astype(str).str.strip() == "")
            out.loc[district_blank, "district"] = out.loc[district_blank, "city"].apply(normalize_case)
        out["district"] = out["district"].fillna("(ukjent bydel)")

    if "status" in out.columns:
        out["status"] = out["status"].apply(normalize_status)

    if "published" in out.columns:
        out["published"] = out["published"].apply(clean_timestamp)
        if "snapshot_at" in out.columns:
            mask = out["published"].isna()
            out.loc[mask, "published"] = out.loc[mask, "snapshot_at"].apply(clean_timestamp)
        if "last_seen_at" in out.columns:
            mask = out["published"].isna()
            out.loc[mask, "published"] = out.loc[mask, "last_seen_at"].apply(clean_timestamp)

    for col in ["broker", "chain", "city", "source", "title", "status"]:
        if col in out.columns:
            out[col] = out[col].fillna(f"(ukjent {col})")

    if "published" in out.columns:
        out["published_dt"] = to_dt_safe(out["published"])
    else:
        out["published_dt"] = pd.NaT

    return out


@lru_cache(maxsize=1)
def _load_current_df() -> pd.DataFrame:
    data_path = OUT_DIR / "all_listings.csv"
    if not data_path.exists():
        return pd.DataFrame()
    df_raw = pd.read_csv(data_path)
    return prepare_dataframe(df_raw)


def get_current_dataframe(refresh: bool = False) -> pd.DataFrame:
    if refresh:
        _load_current_df.cache_clear()
    return _load_current_df().copy()


@lru_cache(maxsize=1)
def _load_agg_df() -> pd.DataFrame:
    path = OUT_DIR / "agg_sum_per_broker.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def get_agg_dataframe(refresh: bool = False) -> pd.DataFrame:
    if refresh:
        _load_agg_df.cache_clear()
    return _load_agg_df().copy()


@lru_cache(maxsize=1)
def _load_dnb_df() -> pd.DataFrame:
    path = OUT_DIR / "dnb_listings.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        raw = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return prepare_dataframe(raw)


def get_dnb_dataframe(refresh: bool = False) -> pd.DataFrame:
    if refresh:
        _load_dnb_df.cache_clear()
    return _load_dnb_df().copy()


@lru_cache(maxsize=1)
def _load_hjem_df() -> pd.DataFrame:
    path = OUT_DIR / "hjem_listings.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        raw = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return prepare_dataframe(raw)


def get_hjem_dataframe(refresh: bool = False) -> pd.DataFrame:
    if refresh:
        _load_hjem_df.cache_clear()
    return _load_hjem_df().copy()


@lru_cache(maxsize=1)
def _load_profiles_df() -> pd.DataFrame:
    path = OUT_DIR / "broker_profiles.csv"
    if not path.exists():
        return pd.DataFrame(columns=["broker", "chain", "linkedin_url", "experience_years", "age"])
    df = pd.read_csv(path)
    for col in ["broker", "chain"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()
    if "linkedin_url" in df.columns:
        df["linkedin_url"] = df["linkedin_url"].fillna("").astype(str).str.strip()
    if "experience_years" in df.columns:
        df["experience_years"] = pd.to_numeric(df["experience_years"], errors="coerce")
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    return df


def get_broker_profiles(refresh: bool = False) -> pd.DataFrame:
    if refresh:
        _load_profiles_df.cache_clear()
    return _load_profiles_df().copy()


def get_filter_options(city: Optional[str] = None, refresh: bool = False) -> Dict[str, List[str]]:
    df = get_current_dataframe(refresh=refresh)
    if df.empty:
        return {"cities": [], "districts": [], "segments": [], "roles": [], "chains": [], "sources": []}
    scope = df[df["city"].eq(city)] if city else df
    districts_series = scope.get("district", pd.Series(dtype="object"))
    segments_series = df.get("property_type", pd.Series(dtype="object"))
    roles_series = df.get("broker_role", pd.Series(dtype="object"))
    chains_series = df.get("chain", pd.Series(dtype="object"))
    sources_series = df.get("source", pd.Series(dtype="object"))
    cities_series = df.get("city", pd.Series(dtype="object"))

    districts = sorted(
        {str(d) for d in districts_series.dropna() if d and d != "(ukjent bydel)"}
    )
    segments = sorted({str(s) for s in segments_series.dropna() if s})
    roles = sorted({str(r) for r in roles_series.dropna() if r})
    chains = sorted({str(c) for c in chains_series.dropna() if c})
    sources = sorted({str(s) for s in sources_series.dropna() if s})
    cities = sorted({str(c) for c in cities_series.dropna() if c})
    return {
        "cities": cities,
        "districts": districts,
        "segments": segments,
        "roles": roles,
        "chains": chains,
        "sources": sources,
    }


def enforce_min_sales(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    if threshold <= 0 or df is None or df.empty:
        return df
    required_cols = {"broker", "chain", "listing_id"}
    if not required_cols.issubset(df.columns):
        return df
    counts = df.groupby(["broker", "chain"], dropna=False)["listing_id"].count()
    qualifying = counts[counts >= threshold]
    if qualifying.empty:
        return df.iloc[0:0].copy()
    qualifying = qualifying.reset_index()[["broker", "chain"]]
    filtered = df.merge(qualifying.assign(_keep=True), on=["broker", "chain"], how="inner")
    return filtered.drop(columns="_keep")


def filter_dataframe(
    df: pd.DataFrame,
    city: str,
    districts: List[str],
    chains: List[str],
    chain_keyword: str,
    roles: List[str],
    segments: List[str],
    sources: List[str],
    search: str,
    period: str,
) -> pd.DataFrame:
    subset = df.copy()
    if city and "city" in subset.columns:
        subset = subset[subset["city"] == city]
    if districts and "district" in subset.columns:
        subset = subset[subset["district"].isin(districts)]
    if chains and "chain" in subset.columns:
        subset = subset[subset["chain"].isin(chains)]
    if chain_keyword.strip() and "chain" in subset.columns:
        kw = chain_keyword.strip().lower()
        subset = subset[subset["chain"].str.lower().str.contains(kw, na=False)]
    if roles and "broker_role" in subset.columns:
        subset = subset[subset["broker_role"].isin(roles)]
    if segments and "property_type" in subset.columns:
        subset = subset[subset["property_type"].isin(segments)]
    if sources and "source" in subset.columns:
        subset = subset[subset["source"].isin(sources)]
    if search.strip():
        s = search.strip().lower()
        mask = False
        if "broker" in subset.columns:
            mask = mask | subset["broker"].str.lower().str.contains(s, na=False)
        if "chain" in subset.columns:
            mask = mask | subset["chain"].str.lower().str.contains(s, na=False)
        if "title" in subset.columns:
            mask = mask | subset["title"].str.lower().str.contains(s, na=False)
        subset = subset[mask]
    if "published_dt" in subset.columns and not subset.empty:
        now_utc = pd.Timestamp.utcnow()
        if period == "Siste 30 dager":
            start = now_utc - pd.Timedelta(days=30)
            subset = subset[subset["published_dt"] >= start]
        elif period == "Siste 12 mnd":
            start = now_utc - pd.Timedelta(days=365)
            subset = subset[subset["published_dt"] >= start]
        elif period == "Dette året":
            oslo_now = now_utc.tz_convert("Europe/Oslo")
            start = pd.Timestamp(year=oslo_now.year, month=1, day=1, tz="Europe/Oslo").tz_convert("UTC")
            subset = subset[subset["published_dt"] >= start]
    return subset


def split_windows_12m(df: pd.DataFrame, col: str = "published_dt") -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty or col not in df.columns:
        return df.iloc[0:0].copy(), df.iloc[0:0].copy()
    end = pd.Timestamp.utcnow()
    start_now = end - pd.Timedelta(days=365)
    start_prev = start_now - pd.Timedelta(days=365)
    now = df[(df[col] >= start_now) & (df[col] <= end)].copy()
    prev = df[(df[col] >= start_prev) & (df[col] < start_now)].copy()
    return now, prev


def pct_change(current: float, previous: float) -> Optional[float]:
    if previous > 0:
        return ((current - previous) / previous) * 100.0
    if current == 0:
        return 0.0
    return None


def _brokers_per_chain(df: pd.DataFrame) -> dict[str, int]:
    if df is None or df.empty or {"chain", "broker"}.isdisjoint(df.columns):
        return {}
    data = df.copy()
    data["chain"] = data["chain"].fillna("(ukjent)")
    data["broker"] = data["broker"].fillna("(ukjent)")
    counts = data.groupby("chain")["broker"].nunique()
    return {str(chain): int(count) for chain, count in counts.items()}


def _commission_tables(df: pd.DataFrame, top_n: int = 10) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if df is None or df.empty:
        return [], []
    required = {"price", "listing_id"}
    if not required.issubset(df.columns):
        return [], []

    brokers = []
    offices = []
    per_chain = _brokers_per_chain(df)

    if {"broker", "chain"}.issubset(df.columns):
        broker_grouped = (
            df.groupby(["broker", "chain"], dropna=False)
            .agg(total_value=("price", "sum"), listing_count=("listing_id", "count"))
            .reset_index()
        )
        broker_grouped["commission"] = broker_grouped["total_value"] * COMMISSION_RATE
        broker_grouped["commission_avg"] = broker_grouped.apply(
            lambda row: row["commission"] / row["listing_count"] if row["listing_count"] else 0.0,
            axis=1,
        )
        broker_grouped = broker_grouped.sort_values("commission", ascending=False).head(top_n)
        for _, row in broker_grouped.iterrows():
            chain_name = str(row.get("chain") or "(ukjent)")
            brokers.append(
                {
                    "broker": row.get("broker"),
                    "chain": chain_name,
                    "listing_count": int(row.get("listing_count", 0)),
                    "total_value": float(row.get("total_value", 0.0)),
                    "commission": float(row.get("commission", 0.0)),
                    "commission_avg": float(row.get("commission_avg", 0.0)),
                    "chain_broker_count": per_chain.get(chain_name, 0),
                }
            )

    if "chain" in df.columns:
        office_grouped = (
            df.groupby(["chain"], dropna=False)
            .agg(total_value=("price", "sum"), listing_count=("listing_id", "count"))
            .reset_index()
        )
        office_grouped["commission"] = office_grouped["total_value"] * COMMISSION_RATE
        office_grouped["commission_avg"] = office_grouped.apply(
            lambda row: row["commission"] / row["listing_count"] if row["listing_count"] else 0.0,
            axis=1,
        )
        office_grouped = office_grouped.sort_values("commission", ascending=False).head(top_n)
        for _, row in office_grouped.iterrows():
            chain_name = str(row.get("chain") or "(ukjent)")
            offices.append(
                {
                    "office": chain_name,
                    "chain": chain_name,
                    "listing_count": int(row.get("listing_count", 0)),
                    "total_value": float(row.get("total_value", 0.0)),
                    "commission": float(row.get("commission", 0.0)),
                    "commission_avg": float(row.get("commission_avg", 0.0)),
                    "chain_broker_count": per_chain.get(chain_name, 0),
                }
            )

    return brokers, offices


def _window_split(df: pd.DataFrame, now_days: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty or "published_dt" not in df.columns:
        empty = df.iloc[0:0].copy() if df is not None else pd.DataFrame()
        return empty, empty
    end = pd.Timestamp.utcnow()
    start_now = end - pd.Timedelta(days=now_days)
    start_prev = start_now - pd.Timedelta(days=now_days)
    now_slice = df[(df["published_dt"] >= start_now) & (df["published_dt"] <= end)].copy()
    prev_slice = df[(df["published_dt"] >= start_prev) & (df["published_dt"] < start_now)].copy()
    return now_slice, prev_slice


def _portfolio_deltas(now_df: pd.DataFrame, prev_df: pd.DataFrame) -> pd.DataFrame:
    required = {"broker", "chain", "price"}
    if now_df is None and prev_df is None:
        return pd.DataFrame(columns=["broker", "chain", "value_now", "value_prev", "delta_value", "delta_pct"])
    now_df = now_df if now_df is not None else pd.DataFrame(columns=list(required))
    prev_df = prev_df if prev_df is not None else pd.DataFrame(columns=list(required))
    for frame in (now_df, prev_df):
        for col in required - set(frame.columns):
            frame[col] = None

    def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["broker", "chain", "value"])
        grouped = (
            frame.groupby(["broker", "chain"], dropna=False)["price"]
            .sum()
            .reset_index(name="value")
        )
        return grouped

    now_values = aggregate(now_df)
    prev_values = aggregate(prev_df)
    merged = pd.merge(
        now_values,
        prev_values,
        on=["broker", "chain"],
        how="outer",
        suffixes=("_now", "_prev"),
    ).fillna(0.0)
    merged["delta_value"] = merged["value_now"] - merged["value_prev"]
    merged["delta_pct"] = np.where(
        merged["value_prev"] > 0,
        (merged["delta_value"] / merged["value_prev"]) * 100.0,
        np.where(merged["value_now"] > 0, 100.0, 0.0),
    )
    return merged


def get_ranking_dataframe(filtered: pd.DataFrame, params: FilterParams) -> pd.DataFrame:
    ranking = build_broker_ranking(filtered)
    if ranking.empty:
        return ranking
    metric = params.sort_by
    ascending = False
    if metric == "latest_listing":
        ascending = False
    ranking = ranking.sort_values(metric, ascending=ascending).reset_index(drop=True)
    ranking["rank"] = ranking.index + 1
    return ranking


def compute_recent_growth(df_in: pd.DataFrame, days: int = 90) -> Tuple[int, int]:
    if df_in is None or df_in.empty or "published_dt" not in df_in.columns:
        return 0, 0
    now = pd.Timestamp.utcnow()
    current_start = now - pd.Timedelta(days=days)
    prev_start = current_start - pd.Timedelta(days=days)
    current = df_in[df_in["published_dt"] >= current_start]
    prev = df_in[(df_in["published_dt"] >= prev_start) & (df_in["published_dt"] < current_start)]
    return int(len(current)), int(len(prev))


def ranking_response(params: FilterParams, refresh: bool = False) -> Dict[str, object]:
    params = params.validated()
    df = get_current_dataframe(refresh=refresh)
    filtered = filter_dataframe(
        df,
        params.city,
        list(params.districts),
        list(params.chains),
        params.chain_keyword,
        list(params.roles),
        list(params.segments),
        list(params.sources),
        params.search,
        params.period,
    )
    filtered = enforce_min_sales(filtered, params.min_sales)
    ranking = get_ranking_dataframe(filtered, params)

    now12, prev12 = split_windows_12m(filtered)
    total_sales_now = int(len(now12))
    total_sales_prev = int(len(prev12))
    total_value_now = float(np.nan_to_num(now12.get("price", []), copy=False).sum())
    total_value_prev = float(np.nan_to_num(prev12.get("price", []), copy=False).sum())
    commission_now = total_value_now * COMMISSION_RATE
    commission_prev = total_value_prev * COMMISSION_RATE
    active_brokers_now = int(now12["broker"].nunique()) if not now12.empty else 0
    active_brokers_prev = int(prev12["broker"].nunique()) if not prev12.empty else 0

    top_brokers, top_offices = _commission_tables(filtered)
    now_window, prev_window = _window_split(filtered, now_days=30)
    deltas = _portfolio_deltas(now_window, prev_window)
    gainers = (
        deltas.sort_values("delta_value", ascending=False).head(5)
        if not deltas.empty
        else pd.DataFrame(columns=["broker", "chain", "delta_value", "delta_pct", "value_now"])
    )
    losers = (
        deltas.sort_values("delta_value", ascending=True).head(5)
        if not deltas.empty
        else pd.DataFrame(columns=["broker", "chain", "delta_value", "delta_pct", "value_now"])
    )

    gainers_payload = [
        {
            "broker": row.get("broker"),
            "chain": row.get("chain"),
            "delta_value": float(row.get("delta_value", 0.0)),
            "delta_pct": float(row.get("delta_pct", 0.0)),
            "value_now": float(row.get("value_now", 0.0)),
        }
        for _, row in gainers.iterrows()
    ]
    losers_payload = [
        {
            "broker": row.get("broker"),
            "chain": row.get("chain"),
            "delta_value": float(row.get("delta_value", 0.0)),
            "delta_pct": float(row.get("delta_pct", 0.0)),
            "value_now": float(row.get("value_now", 0.0)),
        }
        for _, row in losers.iterrows()
    ]

    items: List[Dict[str, object]] = []
    for _, row in ranking.iterrows():
        latest = row.get("latest_listing")
        if pd.notna(latest):
            latest_val = latest.isoformat() if isinstance(latest, pd.Timestamp) else str(latest)
        else:
            latest_val = None
        items.append(
            {
                "rank": int(row.get("rank", 0)),
                "broker_key": row.get("broker_key"),
                "broker": row.get("broker"),
                "chain": row.get("chain"),
                "city": row.get("city"),
                "primary_location": row.get("primary_location"),
                "dominant_segment": row.get("dominant_segment"),
                "total_sales": int(row.get("total_sales", 0)),
                "total_value": float(row.get("total_value", 0.0)),
                "avg_price": float(row.get("avg_price", 0.0)),
                "high_volume": bool(row.get("high_volume", False)),
                "segment_summary": row.get("segment_summary"),
                "location_summary": row.get("location_summary"),
                "broker_role": row.get("broker_role"),
                "latest_listing": latest_val,
            }
        )

    return {
        "filters": params.as_dict(),
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "total_brokers": int(len(ranking)),
        "items": items,
        "kpi": {
            "listings_now": total_sales_now,
            "listings_previous": total_sales_prev,
            "listings_pct_change": pct_change(total_sales_now, total_sales_prev),
            "total_value_now": total_value_now,
            "total_value_previous": total_value_prev,
            "total_value_pct_change": pct_change(total_value_now, total_value_prev),
            "commission_now": commission_now,
            "commission_previous": commission_prev,
            "commission_pct_change": pct_change(commission_now, commission_prev),
            "active_brokers_now": active_brokers_now,
            "active_brokers_previous": active_brokers_prev,
            "active_brokers_pct_change": pct_change(active_brokers_now, active_brokers_prev),
        },
        "top_commission_brokers": top_brokers,
        "top_commission_offices": top_offices,
        "portfolio_gainers": gainers_payload,
        "portfolio_losers": losers_payload,
    }


def broker_detail_response(broker_key: str, params: FilterParams, refresh: bool = False) -> Optional[Dict[str, object]]:
    params = params.validated()
    df = get_current_dataframe(refresh=refresh)
    filtered = filter_dataframe(
        df,
        params.city,
        list(params.districts),
        list(params.chains),
        params.chain_keyword,
        list(params.roles),
        list(params.segments),
        list(params.sources),
        params.search,
        params.period,
    )
    filtered = enforce_min_sales(filtered, params.min_sales)
    ranking = get_ranking_dataframe(filtered, params)
    if ranking.empty or broker_key not in ranking["broker_key"].tolist():
        return None

    row = ranking.set_index("broker_key").loc[broker_key]
    broker_name = row["broker"]
    chain_name = row["chain"]

    broker_subset = filtered[(filtered["broker"] == broker_name) & (filtered["chain"] == chain_name)]

    segment_counts = (
        broker_subset["property_type"]
        .fillna("(ukjent)")
        .value_counts()
        .reset_index(name="count")
        .rename(columns={"index": "segment"})
        if not broker_subset.empty
        else pd.DataFrame(columns=["segment", "count"])
    )
    location_counts = (
        broker_subset["district"]
        .fillna("(ukjent)")
        .value_counts()
        .reset_index(name="count")
        .rename(columns={"index": "district"})
        if not broker_subset.empty
        else pd.DataFrame(columns=["district", "count"])
    )

    prices = broker_subset["price"].dropna()
    price_stats = {
        "min": float(prices.min()) if not prices.empty else None,
        "median": float(prices.median()) if not prices.empty else None,
        "mean": float(prices.mean()) if not prices.empty else None,
        "max": float(prices.max()) if not prices.empty else None,
    }

    timeline: List[Dict[str, object]] = []
    if "published_dt" in broker_subset.columns and not broker_subset.empty:
        series = broker_subset.dropna(subset=["published_dt"]).set_index("published_dt")
        if not series.empty:
            try:
                series = series.tz_convert("Europe/Oslo")
            except TypeError:
                series = series.tz_localize("UTC").tz_convert("Europe/Oslo")
            monthly = series.resample("M").agg(
                listings=("listing_id", "count"),
                value=("price", "sum"),
            )
            for stamp, row_month in monthly.iterrows():
                timeline.append(
                    {
                        "month": stamp.strftime("%Y-%m"),
                        "listings": int(row_month.get("listings", 0)),
                        "value": float(row_month.get("value", 0.0)),
                    }
                )

    recent_now, recent_prev = compute_recent_growth(broker_subset, days=90)

    peers = ranking[
        (ranking["broker_key"] != broker_key)
        & (ranking["primary_location"] == row.get("primary_location"))
        & (ranking["dominant_segment"] == row.get("dominant_segment"))
    ].sort_values("total_value", ascending=False).head(5)
    peer_payload = [
        {
            "broker_key": peer["broker_key"],
            "broker": peer["broker"],
            "chain": peer["chain"],
            "total_sales": int(peer["total_sales"]),
            "total_value": float(peer["total_value"]),
            "avg_price": float(peer["avg_price"]),
        }
        for _, peer in peers.iterrows()
    ]

    recommendations = ranking[ranking["broker_key"] != broker_key].copy()
    recommendations["segment_match"] = recommendations["dominant_segment"] == row.get("dominant_segment")
    recommendations["district_match"] = recommendations["primary_location"] == row.get("primary_location")
    recommendations["score"] = (
        recommendations["segment_match"].astype(int) * 2 + recommendations["district_match"].astype(int)
    )
    rec_top = recommendations.sort_values(["score", "total_value"], ascending=[False, False]).head(3)
    rec_payload = [
        {
            "broker_key": rec["broker_key"],
            "broker": rec["broker"],
            "chain": rec["chain"],
            "dominant_segment": rec["dominant_segment"],
            "primary_location": rec["primary_location"],
            "total_value": float(rec["total_value"]),
            "score": int(rec["score"]),
        }
        for _, rec in rec_top.iterrows()
    ]

    profiles = get_broker_profiles(refresh=refresh)
    profile_info = None
    if not profiles.empty:
        matches = profiles[
            (profiles["broker"].str.lower() == broker_name.lower())
            & (profiles["chain"].str.lower() == chain_name.lower())
        ]
        if matches.empty:
            matches = profiles[profiles["broker"].str.lower() == broker_name.lower()]
        if not matches.empty:
            profile_row = matches.iloc[0]
            profile_info = {
                "linkedin_url": profile_row.get("linkedin_url") or None,
                "experience_years": float(profile_row.get("experience_years"))
                if pd.notna(profile_row.get("experience_years"))
                else None,
                "age": int(profile_row.get("age")) if pd.notna(profile_row.get("age")) else None,
            }

    return {
        "broker_key": broker_key,
        "broker": broker_name,
        "chain": chain_name,
        "broker_role": row.get("broker_role"),
        "city": row.get("city"),
        "primary_location": row.get("primary_location"),
        "dominant_segment": row.get("dominant_segment"),
        "high_volume": bool(row.get("high_volume", False)),
        "segment_summary": row.get("segment_summary"),
        "location_summary": row.get("location_summary"),
        "metrics": {
            "total_sales": int(row.get("total_sales", 0)),
            "total_value": float(row.get("total_value", 0.0)),
            "avg_price": float(row.get("avg_price", 0.0)),
            "recent_sales": {"current": recent_now, "previous": recent_prev},
            "commission_estimate": float(row.get("total_value", 0.0)) * COMMISSION_RATE,
        },
        "segments": [
            {"segment": seg, "count": int(cnt)} for seg, cnt in segment_counts.values.tolist()
        ],
        "locations": [
            {"district": dist, "count": int(cnt)} for dist, cnt in location_counts.values.tolist()
        ],
        "price_stats": price_stats,
        "timeline": timeline,
        "peers": peer_payload,
        "recommendations": rec_payload,
        "profile": profile_info,
        "filters": params.as_dict(),
    }


def overview_response(refresh: bool = False) -> Dict[str, object]:
    df = get_current_dataframe(refresh=refresh)
    agg_df = get_agg_dataframe(refresh=refresh)
    dnb_df = get_dnb_dataframe(refresh=refresh)
    hjem_df = get_hjem_dataframe(refresh=refresh)
    profiles_df = get_broker_profiles(refresh=refresh)
    params = FilterParams().validated()
    ranking = get_ranking_dataframe(df, params)

    now12, prev12 = split_windows_12m(df)
    total_sales_now = int(len(df))
    total_sales_prev = int(len(prev12))
    total_value_now = float(np.nan_to_num(df.get("price", []), copy=False).sum())
    total_value_prev = float(np.nan_to_num(prev12.get("price", []), copy=False).sum())
    commission_now = total_value_now * COMMISSION_RATE
    commission_prev = total_value_prev * COMMISSION_RATE
    active_brokers_now = int(df["broker"].nunique()) if not df.empty else 0
    active_brokers_prev = int(prev12["broker"].nunique()) if not prev12.empty else 0

    timeline: List[Dict[str, object]] = []
    if not df.empty and "published_dt" in df.columns:
        published = df.dropna(subset=["published_dt"]).set_index("published_dt")
        if not published.empty:
            try:
                published = published.tz_convert("Europe/Oslo")
            except TypeError:
                published = published.tz_localize("UTC").tz_convert("Europe/Oslo")
            monthly = published.resample("M").agg(
                listings=("listing_id", "count"),
                value=("price", "sum"),
            )
            for stamp, row_month in monthly.iterrows():
                timeline.append(
                    {
                        "month": stamp.strftime("%Y-%m"),
                        "listings": int(row_month.get("listings", 0)),
                        "value": float(row_month.get("value", 0.0)),
                    }
                )

    def top_counts(series: pd.Series, limit: int = 6) -> List[Dict[str, object]]:
        if series is None or series.empty:
            return []
        counts = series.fillna("(ukjent)").value_counts().head(limit)
        total = int(counts.sum()) if not counts.empty else 0
        return [
            {
                "label": name,
                "count": int(count),
                "share": (int(count) / total) if total else 0.0,
            }
            for name, count in counts.items()
        ]

    segments = top_counts(df.get("property_type", pd.Series(dtype="object")))
    locations = top_counts(df.get("district", pd.Series(dtype="object")))
    chains = top_counts(df.get("chain", pd.Series(dtype="object")))

    high_volume = int(ranking["high_volume"].sum()) if not ranking.empty else 0
    datasets_info = {
        "all_listings": int(len(df)),
        "agg_sum_per_broker": int(len(agg_df)),
        "dnb_listings": int(len(dnb_df)),
        "hjem_listings": int(len(hjem_df)),
        "broker_profiles": int(len(profiles_df)),
    }

    return {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "broker_count": int(ranking["broker"].nunique()) if not ranking.empty else 0,
        "high_volume": high_volume,
        "kpi": {
            "listings_now": total_sales_now,
            "listings_previous": total_sales_prev,
            "listings_pct_change": pct_change(total_sales_now, total_sales_prev),
            "total_value_now": total_value_now,
            "total_value_previous": total_value_prev,
            "total_value_pct_change": pct_change(total_value_now, total_value_prev),
            "commission_now": commission_now,
            "commission_previous": commission_prev,
            "commission_pct_change": pct_change(commission_now, commission_prev),
            "active_brokers_now": active_brokers_now,
            "active_brokers_previous": active_brokers_prev,
            "active_brokers_pct_change": pct_change(active_brokers_now, active_brokers_prev),
        },
        "datasets": datasets_info,
        "timeline": timeline,
        "segments": segments,
        "locations": locations,
        "chains": chains,
    }
