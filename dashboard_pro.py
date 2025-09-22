# dashboard_pro.py
from pathlib import Path
import math
from typing import Tuple
import pandas as pd
import numpy as np
import streamlit as st

# -------------------- App setup --------------------
st.set_page_config(page_title="MeglerMonitor", layout="wide")
OUT = (Path(__file__).resolve().parent / "out").expanduser()
OUT.mkdir(parents=True, exist_ok=True)

# -------------------- Helpers --------------------
@st.cache_data
def load_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def fmt_nok(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and (pd.isna(x) or math.isnan(float(x)))):
        return "–"
    try:
        return f"{int(round(float(x))):,} kr".replace(",", " ").replace("\xa0", " ")
    except Exception:
        return "–"

def fmt_compact_nok(x: float | int | None) -> str:
    """12,4 mrd / 1,2 mill / 123 000 (for KPI row, matches screenshot vibe)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "–"
    v = float(x)
    if abs(v) >= 1_000_000_000:  # billion
        return f"{v/1_000_000_000:.1f}".replace(".", ",") + " mrd"
    if abs(v) >= 1_000_000:      # million
        return f"{v/1_000_000:.1f}".replace(".", ",") + " mill"
    return f"{int(round(v)):,}".replace(",", " ")

def fmt_delta(value: float | None, pct: float | None) -> str:
    if value is None and pct is None:
        return ""
    parts = []
    if value is not None and not pd.isna(value):
        sign = "+" if value >= 0 else "–"
        parts.append(f"{sign}{fmt_nok(abs(value))}")
    if pct is not None and not pd.isna(pct):
        parts.append(f"({pct:+.1f}%)")
    return " ".join(parts)


STATUS_LABELS = {
    0: "unknown",
    1: "coming",
    2: "available",
    3: "sold",
    4: "reserved",
    5: "inactive",
    99: "archived",
}


def normalize_case(value: str | float | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    lowered = text.lower()
    if not text or lowered in {"none", "nan", "null"}:
        return None
    return text.title() if text.upper() == text else text


def clean_timestamp(value: str | float | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "nat", "null"}:
        return None
    return text


def infer_city_from_address(address: str | None) -> str | None:
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

    seen = set()
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


def normalize_status(value) -> str | None:
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


def initials(name: str | None) -> str:
    if not name: return "?"
    bits = [b for b in str(name).strip().split() if b]
    if len(bits) == 1: return bits[0][:2].upper()
    return (bits[0][0] + bits[-1][0]).upper()

def kpi_card(label: str, value: str, sublabel: str = "–", positive=True):
    color_class = "mm-kpi-value-green" if positive else "mm-kpi-value-red"
    sub_class = "mm-kpi-sub-green" if positive else "mm-kpi-sub-red"
    st.markdown(
        f"""
        <div class="mm-card">
          <div class="mm-kpi-label">{label}</div>
          <div class="{color_class}">{value}</div>
          <div class="{sub_class}">{sublabel}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def to_dt_safe(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)

def split_windows_12m(df: pd.DataFrame, col="published_dt") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """NOW=last 12 months, PREV=the 12 months before that."""
    if col not in df.columns:
        return df.iloc[0:0].copy(), df.iloc[0:0].copy()
    end = pd.Timestamp.utcnow()
    start_now = end - pd.Timedelta(days=365)
    start_prev = start_now - pd.Timedelta(days=365)
    now = df[(df[col] >= start_now) & (df[col] <= end)].copy()
    prev = df[(df[col] >= start_prev) & (df[col] < start_now)].copy()
    return now, prev

def sales_count(df_in: pd.DataFrame) -> int:
    """Return antall aktive annonser i datasettet."""
    if df_in.empty:
        return 0
    return int(len(df_in))

# -------------------- CSS --------------------
st.markdown("""
<style>
body { background: #0d1015; }
.block-container { padding-top: 18px; padding-bottom: 40px; }

/* KPI styling */
.mm-kpi-row { display:grid; grid-template-columns: repeat(4, 1fr); gap:16px; margin-bottom: 8px; }
.mm-card { background:#12151c; border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:16px; }
.mm-kpi-label { color:#aeb4be; font-size:12px; margin-bottom:4px; }
.mm-kpi-value-green { font-weight:800; font-size:28px; color:#35d07f; }
.mm-kpi-value-red { font-weight:800; font-size:28px; color:#ff5c5c; }
.mm-kpi-sub-green { color:#35d07f; font-size:12px; margin-top:4px; }
.mm-kpi-sub-red { color:#ff5c5c; font-size:12px; margin-top:4px; }

/* Sections / rows (unchanged from your previous styling) */
.mm-title { font-weight: 700; font-size: 18px; margin-bottom: 8px; }
.mm-subtle { color: #aeb4be; font-size: 12px; margin-bottom: 8px; }
.mm-row { display:flex; align-items:center; gap:12px; padding:16px; border-radius:14px; margin-bottom:8px; }
.mm-row.vokser { background:rgba(47,158,68,.08); }
.mm-row.faller { background:rgba(190,27,27,.08); }
.mm-rank { width:28px; text-align:center; color:#9aa1ab; font-weight:600; }
.mm-avatar { width:40px; height:40px; border-radius:999px; background:#1d232b; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:18px; }
.mm-avatar.vokser { background:#2f9e44; color:#fff; }
.mm-avatar.faller { background:#be1b1b; color:#fff; }
.mm-main { flex:1; min-width:0; }
.mm-name { font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:32vw; }
.mm-sub { font-size:12px; color:#9aa1ab; }
.mm-right { text-align:right; }
.mm-value { font-weight:700; font-size:18px; }
.mm-chip { display:inline-flex; gap:6px; align-items:center; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; margin-top:4px; }
.mm-chip.up { background:rgba(47,158,68,.12); color:#2f9e44; border:1px solid rgba(47,158,68,.3); font-weight:700; }
.mm-chip.down { background:rgba(190,27,27,.10); color:#be1b1b; border:1px solid rgba(190,27,27,.25); font-weight:700; }
.mm-section-title { font-weight:700; font-size:18px; margin: 8px 0 12px 0; }
.mm-table { width:100%; border-collapse:separate; border-spacing:0 8px; }
.mm-th, .mm-td { padding:10px 8px; }
.mm-th { color:#aeb4be; font-size:13px; font-weight:600; background:transparent; }
.mm-td { font-size:15px; background:#161a20; border-radius:10px; }
.mm-office-icon { width:28px; height:28px; border-radius:999px; background:#1d232b; display:inline-flex; align-items:center; justify-content:center; margin-right:8px; color:#7fd188; font-weight:700; font-size:18px; }
</style>
""", unsafe_allow_html=True)

# -------------------- Load & clean data --------------------
df = load_csv(OUT / "all_listings.csv")
if df.empty:
    st.warning("Ingen data i `out/`. Kjør: `python -u megler_monitor_poc.py` først.")
    st.stop()

# Numeric/text cleanup
if "price" in df.columns:
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

if "city" in df.columns:
    df["city"] = df["city"].apply(normalize_case)
    if "address" in df.columns:
        missing_city = df["city"].isna() | df["city"].astype(str).str.strip().eq("")
        df.loc[missing_city, "city"] = df.loc[missing_city, "address"].apply(infer_city_from_address)
    df["city"] = df["city"].apply(normalize_case)

if "status" in df.columns:
    df["status"] = df["status"].apply(normalize_status)

if "published" in df.columns:
    df["published"] = df["published"].apply(clean_timestamp)
    if "snapshot_at" in df.columns:
        mask = df["published"].isna()
        df.loc[mask, "published"] = df.loc[mask, "snapshot_at"].apply(clean_timestamp)
    if "last_seen_at" in df.columns:
        mask = df["published"].isna()
        df.loc[mask, "published"] = df.loc[mask, "last_seen_at"].apply(clean_timestamp)

for col in ["broker", "chain", "city", "source", "title", "status"]:
    if col in df.columns:
        df[col] = df[col].fillna(f"(ukjent {col})")

# Dates
if "published" in df.columns:
    df["published_dt"] = to_dt_safe(df["published"])
else:
    df["published_dt"] = pd.NaT

# -------------------- Header --------------------
st.title("MeglerMonitor")
snapshot_ts = pd.Timestamp.now(tz="Europe/Oslo").strftime("%Y-%m-%d %H:%M")

# -------------------- Filters --------------------
with st.expander("Filtre", expanded=True):
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1])

    cities = ["(Alle)"] + sorted(df["city"].dropna().unique().tolist())
    sel_city = c1.selectbox("By", cities, index=0)

    chains = sorted(df["chain"].dropna().unique().tolist())
    sel_chains = c2.multiselect("Kjede/kontor", chains, default=[])

    sources = sorted(df["source"].dropna().unique().tolist())
    sel_sources = c3.multiselect("Kilde", sources, default=sources)

    search = c4.text_input("Søk megler/kontor", "", placeholder="Søk…")

    c5, _ = st.columns([1.2, 3])
    period = c5.selectbox(
        "Tidsperiode (brukes i listene under)",
        ["Alle", "Siste 30 dager", "Siste 12 mnd", "Dette året"],
        index=0
    )

flt = df.copy()
if sel_city != "(Alle)":
    flt = flt[flt["city"] == sel_city]
if sel_chains:
    flt = flt[flt["chain"].isin(sel_chains)]
if sel_sources:
    flt = flt[flt["source"].isin(sel_sources)]
if search.strip():
    s = search.strip().lower()
    flt = flt[
        flt["broker"].str.lower().str.contains(s, na=False) |
        flt["chain"].str.lower().str.contains(s, na=False) |
        flt["title"].str.lower().str.contains(s, na=False)
    ]

# Optional time filter for the tables
if "published_dt" in flt.columns:
    now_utc = pd.Timestamp.utcnow()
    if period == "Siste 30 dager":
        start = now_utc - pd.Timedelta(days=30)
        flt = flt[flt["published_dt"] >= start]
    elif period == "Siste 12 mnd":
        start = now_utc - pd.Timedelta(days=365)
        flt = flt[flt["published_dt"] >= start]
    elif period == "Dette året":
        start = pd.Timestamp(year=now_utc.tz_convert("Europe/Oslo").year, month=1, day=1, tz="Europe/Oslo").tz_convert("UTC")
        flt = flt[flt["published_dt"] >= start]

# -------------------- KPI row (last 12m vs previous 12m) --------------------
# KPI should always be computed on the CURRENT filter selection but with fixed windows (12m vs prev 12m)
now12, prev12 = split_windows_12m(flt, col="published_dt")

# Aktive annonser (siste 12 mnd)
listings_now = sales_count(now12)
listings_prev = sales_count(prev12)
listings_delta_val = listings_now - listings_prev
listings_delta_pct = ((listings_now - listings_prev) / listings_prev * 100) if listings_prev > 0 else (100.0 if listings_now > 0 else 0.0)

# Total verdi av aktive boliger
omset_now = float(np.nan_to_num(now12["price"]).sum())
omset_prev = float(np.nan_to_num(prev12["price"]).sum())
omset_delta_val = omset_now - omset_prev
omset_delta_pct = (omset_delta_val / omset_prev * 100) if omset_prev > 0 else (100.0 if omset_now > 0 else 0.0)

# Meglere med aktive annonser
active_now = int(now12["broker"].nunique()) if not now12.empty else 0
active_prev = int(prev12["broker"].nunique()) if not prev12.empty else 0
active_delta_val = active_now - active_prev
active_delta_pct = (active_delta_val / active_prev * 100) if active_prev > 0 else (100.0 if active_now > 0 else 0.0)

# Gj.snitt dager i markedet – placeholder
days_now = "–"
days_delta_text = "–"

st.markdown('<div class="mm-kpi-row">', unsafe_allow_html=True)
kpi_card("Aktive annonser (siste 12 mnd)",
         f"{listings_now:,}".replace(",", " "),
         f"{listings_delta_pct:+.1f}% vs forrige periode",
         positive=(listings_delta_val >= 0))
kpi_card("Total verdi av aktive boliger",
         fmt_compact_nok(omset_now),
         f"{omset_delta_pct:+.1f}% vs forrige periode",
         positive=(omset_delta_val >= 0))
kpi_card("Meglere med aktive annonser",
         str(active_now),
         f"{active_delta_pct:+.1f}% vs forrige periode",
         positive=(active_delta_val >= 0))
kpi_card("Gj.snitt dager på markedet",
         days_now,
         days_delta_text,
         positive=True)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------- TOPP 5 MEGLERE / KONTOR --------------------
CARD_STYLE = "width:100%;max-width:900px;margin:auto;"
colL, colR = st.columns([1, 1], gap="large")
brokers_total = int(flt["broker"].nunique()) if "broker" in flt.columns else 0
brokers_per_chain = (
    flt.groupby("chain")["broker"].nunique() if {"chain", "broker"}.issubset(flt.columns)
    else pd.Series(dtype="int64")
)

with colL:
    st.markdown(f'<div class="mm-card" style="{CARD_STYLE}">', unsafe_allow_html=True)
    st.markdown('<div class="mm-title">Flest aktive boliger – meglere</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mm-subtle">Antall meglere i utvalget: {brokers_total}</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;gap:16px;padding:0 8px 8px 8px;color:#aeb4be;font-size:13px;">
      <div style="width:40px;">#</div>
      <div style="flex:2;">Megler</div>
      <div style="flex:2;">Kontor</div>
      <div style="width:120px;text-align:center;">Meglere i kjeden</div>
      <div style="width:80px;text-align:center;">Aktive boliger</div>
      <div style="width:160px;text-align:right;">Samlet verdi</div>
    </div>
    """, unsafe_allow_html=True)

    brokers_now = (
        flt.groupby(["broker", "chain"], dropna=False)
        .agg(total_value=("price", "sum"), n=("listing_id", "count"))
        .reset_index()
        .sort_values("total_value", ascending=False)
        .head(10)
    )
    container_style = "max-height:540px;overflow:auto;padding-right:4px;"
    st.markdown(f'<div style="{container_style}">', unsafe_allow_html=True)
    for i, row in brokers_now.reset_index(drop=True).iterrows():
        rank = i + 1
        name = row["broker"]
        chain = row["chain"]
        count = int(row["n"])
        total = fmt_nok(row["total_value"])
        broker_count = int(brokers_per_chain.get(chain, 0)) if not brokers_per_chain.empty else 0
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:16px;background:#161a20;border-radius:12px;padding:12px 8px;margin-bottom:8px;">
          <div style="width:40px;color:#9aa1ab;font-weight:600;">#{rank}</div>
          <div style="flex:2;display:flex;align-items:center;">
            <span class="mm-avatar" style="margin-right:8px;">{initials(name)}</span>
            <span style="font-weight:600; color:#fff;">{name}</span>
          </div>
          <div style="flex:2;color:#aeb4be;">{chain}</div>
          <div style="width:120px;text-align:center;">
            <span style="background:#23262B;color:#9aa1ab;font-weight:700;padding:2px 12px;border-radius:999px;">{broker_count}</span>
          </div>
          <div style="width:80px;text-align:center;">
            <span style="background:#23262B;color:#2f9e44;font-weight:700;padding:2px 12px;border-radius:999px;">{count}</span>
          </div>
          <div style="width:160px;text-align:right;">
            <span style="background:#23262B;color:#ffd700;font-weight:700;padding:2px 12px;border-radius:999px;">{total}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)  # close scroll container
    st.markdown("</div>", unsafe_allow_html=True)  # close card

with colR:
    st.markdown(f'<div class="mm-card" style="{CARD_STYLE}">', unsafe_allow_html=True)
    st.markdown('<div class="mm-title">Flest aktive boliger – kontorer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mm-subtle">Antall meglere i utvalget: {brokers_total}</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;gap:16px;padding:0 8px 8px 8px;color:#aeb4be;font-size:13px;">
      <div style="width:40px;">#</div>
      <div style="flex:2;">Kontor</div>
      <div style="flex:2;">Kjede</div>
      <div style="width:120px;text-align:center;">Meglere i kjeden</div>
      <div style="width:80px;text-align:center;">Aktive boliger</div>
      <div style="width:160px;text-align:right;">Samlet verdi</div>
    </div>
    """, unsafe_allow_html=True)

    offices_now = (
        flt.groupby(["chain"], dropna=False)
        .agg(total_value=("price", "sum"), n=("listing_id", "count"))
        .reset_index()
        .sort_values("total_value", ascending=False)
        .head(10)
    )
    st.markdown(f'<div style="{container_style}">', unsafe_allow_html=True)
    for i, row in offices_now.reset_index(drop=True).iterrows():
        rank = i + 1
        office = row["chain"]
        chain = row["chain"]
        count = int(row["n"])
        total = fmt_nok(row["total_value"])
        broker_count = int(brokers_per_chain.get(chain, 0)) if not brokers_per_chain.empty else 0
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:16px;background:#161a20;border-radius:12px;padding:12px 8px;margin-bottom:8px;">
          <div style="width:40px;color:#9aa1ab;font-weight:600;">#{rank}</div>
          <div style="flex:2;display:flex;align-items:center;">
            <span class="mm-office-icon">🏢</span>
            <span style="font-weight:600; color:#fff;">{office}</span>
          </div>
          <div style="flex:2;color:#aeb4be;">{chain}</div>
          <div style="width:120px;text-align:center;">
            <span style="background:#23262B;color:#9aa1ab;font-weight:700;padding:2px 12px;border-radius:999px;">{broker_count}</span>
          </div>
          <div style="width:80px;text-align:center;">
            <span style="background:#23262B;color:#2f9e44;font-weight:700;padding:2px 12px;border-radius:999px;">{count}</span>
          </div>
          <div style="width:160px;text-align:right;">
            <span style="background:#23262B;color:#ffd700;font-weight:700;padding:2px 12px;border-radius:999px;">{total}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# -------------------- RANGERING: Vokser vs Faller (30d vs prev 30d) --------------------
st.markdown('<div class="mm-section-title">Endring i aktive boliger</div>', unsafe_allow_html=True)
cGrow, cFall = st.columns(2)

c8, _ = st.columns([1.2, 3])
portfolio_options = {
    "Siste 30 dager": 30,
    "Siste 60 dager": 60,
    "Siste 90 dager": 90,
    "Siste 180 dager": 180,
}
portfolio_label = c8.selectbox(
    "Vindu for endring i aktive boliger",
    list(portfolio_options.keys()),
    index=0,
    help="Bestem hvor langt tilbake vi ser når vi sammenligner aktive boliger (nå-vindu vs forrige vindu)."
)
portfolio_window_days = portfolio_options[portfolio_label]

def window_split(df: pd.DataFrame, now_days=30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "published_dt" not in df.columns:
        return df.iloc[0:0].copy(), df.iloc[0:0].copy()
    end = pd.Timestamp.utcnow()
    start_now = end - pd.Timedelta(days=now_days)
    start_prev = start_now - pd.Timedelta(days=now_days)
    now = df[(df["published_dt"] >= start_now) & (df["published_dt"] <= end)].copy()
    prev = df[(df["published_dt"] >= start_prev) & (df["published_dt"] < start_now)].copy()
    return now, prev

now_win, prev_win = window_split(flt, now_days=portfolio_window_days)

def deltas_per_broker(now_df: pd.DataFrame, prev_df: pd.DataFrame) -> pd.DataFrame:
    def agg(x: pd.DataFrame):
        return (x.groupby(["broker", "chain"], dropna=False)["price"]
                  .sum().reset_index(name="value"))
    a = agg(now_df) if len(now_df) else pd.DataFrame(columns=["broker","chain","value"])
    b = agg(prev_df) if len(prev_df) else pd.DataFrame(columns=["broker","chain","value"])
    merged = pd.merge(a, b, on=["broker","chain"], how="outer", suffixes=("_now","_prev")).fillna(0)
    merged["delta_value"] = merged["value_now"] - merged["value_prev"]
    merged["delta_pct"] = np.where(
        merged["value_prev"] > 0,
        (merged["delta_value"] / merged["value_prev"]) * 100.0,
        np.where(merged["value_now"] > 0, 100.0, 0.0)
    )
    return merged

deltas = deltas_per_broker(now_win, prev_win)

with cGrow:
    st.markdown('<div class="mm-card">', unsafe_allow_html=True)
    st.markdown('<div class="mm-title">Meglere med flere aktive boliger</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:#aeb4be;font-size:13px;margin-bottom:10px;">Størst økning i aktive boliger ({portfolio_window_days} dager vs forrige {portfolio_window_days} dager)</div>', unsafe_allow_html=True)
    grow = deltas.sort_values("delta_value", ascending=False).head(5)
    for i, row in grow.reset_index(drop=True).iterrows():
        rank = i+1
        name = row["broker"]
        chain = row["chain"]
        total_now = row["value_now"]
        delta_text = fmt_delta(row["delta_value"], row["delta_pct"])
        current_text = f"Aktive nå: {fmt_nok(total_now)}"
        st.markdown(f"""
        <div class="mm-row vokser">
          <div class="mm-rank">#{rank}</div>
          <div class="mm-avatar vokser">{initials(name)}</div>
          <div class="mm-main">
            <div class="mm-name">{name}</div>
            <div class="mm-sub">{chain}</div>
          </div>
          <div class="mm-right">
            <div class="mm-value">{delta_text}</div>
            <div class="mm-sub" style="margin-top:4px;">{current_text}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with cFall:
    st.markdown('<div class="mm-card">', unsafe_allow_html=True)
    st.markdown('<div class="mm-title">Meglere med færre aktive boliger</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:#aeb4be;font-size:13px;margin-bottom:10px;">Størst nedgang i aktive boliger ({portfolio_window_days} dager vs forrige {portfolio_window_days} dager)</div>', unsafe_allow_html=True)
    fall = deltas.sort_values("delta_value", ascending=True).head(5)
    for i, row in fall.reset_index(drop=True).iterrows():
        rank = i+1
        name = row["broker"]
        chain = row["chain"]
        total_now = row["value_now"]
        delta_text = fmt_delta(row["delta_value"], row["delta_pct"])
        current_text = f"Aktive nå: {fmt_nok(total_now)}"
        st.markdown(f"""
        <div class="mm-row faller">
          <div class="mm-rank">#{rank}</div>
          <div class="mm-avatar faller">{initials(name)}</div>
          <div class="mm-main">
            <div class="mm-name">{name}</div>
            <div class="mm-sub">{chain}</div>
          </div>
          <div class="mm-right">
            <div class="mm-value">{delta_text}</div>
            <div class="mm-sub" style="margin-top:4px;">{current_text}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Raw table --------------------
st.write("")
st.subheader("Alle aktive annonser")
st.caption(f"Data hentet {snapshot_ts} (Europe/Oslo). Tallene viser aktive annonser som er ute i markedet nå.")
st.markdown("Denne tabellen viser hver enkelt annonse med megler, kjede, adresse og prisantydning slik den er publisert hos Hjem.no og DNB.")
df_display = flt.drop(columns=["listing_id"], errors="ignore").copy()
if "published_dt" in df_display.columns:
    df_display["published_local"] = df_display["published_dt"].dt.tz_convert("Europe/Oslo").dt.strftime("%Y-%m-%d %H:%M")
    cols = df_display.columns.tolist()
    if "published_local" in cols:
        cols.insert(0, cols.pop(cols.index("published_local")))
        df_display = df_display[cols]

df_display = df_display.sort_values("price", ascending=False).reset_index(drop=True)
df_display.index = df_display.index + 1

st.dataframe(df_display, use_container_width=True, height=520)

st.download_button(
    label="Last ned filtrert data",
    data=flt.to_csv(index=False).encode("utf-8"),
    file_name="meglermonitor_filtered.csv",
    mime="text/csv"
)
