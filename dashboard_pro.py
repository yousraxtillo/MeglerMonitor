from pathlib import Path
import math
import re
from collections import Counter
from typing import Tuple
import pandas as pd
import numpy as np
import streamlit as st


# Skrive liste slik at vi vet hvordan vi utfører oppgaven. Vi ser på hva som
#  må bli gjort og fullførerr det sakte. Så vi skriver ikke alle todo i amtalen men vu legger
#  til 1 og en slik at det ikkke oppstår noe feil

#App setup
st.set_page_config(page_title="MeglerMonitor", layout="wide")
OUT = (Path(__file__).resolve().parent / "out").expanduser()
OUT.mkdir(parents=True, exist_ok=True)

if "phase2_view_mode" not in st.session_state:
    st.session_state["phase2_view_mode"] = "list"
if "phase2_selected" not in st.session_state:
    st.session_state["phase2_selected"] = None

#Helpers
@st.cache_data
def load_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data
def load_snapshot_csv(filename: str, which: str = "earliest") -> tuple[pd.DataFrame | None, str | None]:
    raw_dir = OUT / "raw"
    if not raw_dir.exists():
        return None, None

    candidates = sorted(raw_dir.glob(f"*_{filename}"))
    if not candidates:
        return None, None

    if which == "earliest":
        target = candidates[0]
    elif which == "latest":
        target = candidates[-1]
    else:
        raise ValueError("which must be 'earliest' or 'latest'")

    try:
        df = pd.read_csv(target)
    except Exception:
        return None, None

    stamp = target.name[:-len(filename)].rstrip("_")
    return df, stamp


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


def fmt_compact_nok_with_kr(x: float | int | None) -> str:
    base = fmt_compact_nok(x)
    return base if base == "–" else base + " kr"

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


POSTAL_CODE_RE = re.compile(r"(\d{4})")

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
    "akerselva": "Akerselva",
    "aker brygge": "Aker Brygge",
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
    "skøyen": "Skøyen",
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


def _normalize_token(value: str | None) -> str:
    if not value:
        return ""
    lowered = value.lower()
    for src, dst in _TOKEN_REPLACEMENTS.items():
        lowered = lowered.replace(src, dst)
    lowered = re.sub(r"[^a-z0-9 ]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def extract_postal_code(address: str | None) -> str | None:
    if not address or pd.isna(address):
        return None
    match = POSTAL_CODE_RE.search(str(address))
    return match.group(1) if match else None


def infer_district(city: str | None, address: str | None, chain: str | None) -> str | None:
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


def _segment_summary_tuple(series: pd.Series) -> tuple[str, str | None]:
    if series is None or series.empty:
        return "–", None
    counts = series.fillna("(ukjent segment)").value_counts()
    if counts.empty:
        return "–", None
    dominant = counts.index[0]
    top_two = counts.head(2)
    parts = [f"{name} ({count})" for name, count in top_two.items()]
    return ", ".join(parts), dominant


def _location_summary_tuple(series: pd.Series) -> tuple[str, str | None]:
    if series is None or series.empty:
        return "–", None
    counts = series.fillna("(ukjent bydel)").value_counts()
    if counts.empty:
        return "–", None
    primary = counts.index[0]
    parts = [f"{name} ({count})" for name, count in counts.head(2).items()]
    return ", ".join(parts), primary


BROKER_KEY_SEP = "||"


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
        threshold = (
            ranking["total_value"].quantile(0.9)
            if len(ranking) >= 5
            else ranking["total_value"].median()
        )
        ranking["high_volume"] = ranking["total_value"] >= threshold
    else:
        ranking["high_volume"] = False

    return ranking


def render_phase2_list(ranking: pd.DataFrame) -> None:
    st.markdown("<div class=\"mm-section-title\">Rangert liste over meglere</div>", unsafe_allow_html=True)

    top_n_default = st.session_state.setdefault("phase2_top_n", 12)
    top_n = st.selectbox(
        "Vis topp",
        options=[6, 9, 12, 24],
        index=[6, 9, 12, 24].index(top_n_default) if top_n_default in {6, 9, 12, 24} else 2,
        help="Velg hvor mange meglere som vises i oversikten.",
    )
    st.session_state["phase2_top_n"] = top_n

    subset = ranking.head(top_n)
    if subset.empty:
        st.info("Ingen meglere matcher filtrene.")
        return

    columns_per_row = 3
    rows = (len(subset) + columns_per_row - 1) // columns_per_row
    for row_idx in range(rows):
        cols = st.columns(columns_per_row)
        for col_idx in range(columns_per_row):
            item_index = row_idx * columns_per_row + col_idx
            if item_index >= len(subset):
                cols[col_idx].empty()
                continue
            row = subset.iloc[item_index]
            rank = int(row.get("rank", item_index + 1))
            total_value = fmt_compact_nok_with_kr(row.get("total_value"))
            avg_price = fmt_compact_nok_with_kr(row.get("avg_price"))
            card_html = f"""
            <div class="mm-card" style="padding:16px; min-height:200px; display:flex;flex-direction:column;justify-content:space-between;">
              <div>
                <div class="mm-subtle" style="margin-bottom:6px;">#{rank}</div>
                <div style="font-weight:700;font-size:18px;color:#fff;">{row['broker']}</div>
                <div class="mm-subtle" style="margin-top:2px;">{row.get('chain','')}</div>
                {'<div class="mm-chip up" style="margin-top:6px;">Høyvolum</div>' if row.get('high_volume') else ''}
                <div class="mm-subtle" style="margin-top:10px;">{row.get('city','')} · {row.get('primary_location','')}</div>
                <div class="mm-subtle" style="margin-top:6px;">{row.get('segment_summary','–')}</div>
              </div>
              <div style="margin-top:12px;">
                <div style="display:flex;justify-content:space-between;font-size:13px;color:#aeb4be;">
                  <span>Salg</span><span>{int(row.get('total_sales',0))}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:13px;color:#aeb4be;">
                  <span>Samlet verdi</span><span>{total_value}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:13px;color:#aeb4be;">
                  <span>Snittpris</span><span>{avg_price}</span>
                </div>
              </div>
            </div>
            """
            cols[col_idx].markdown(card_html, unsafe_allow_html=True)
            button_key = f"phase2_open_{row['broker_key']}_card"
            if cols[col_idx].button("Åpne meglerkort", key=button_key, use_container_width=True):
                st.session_state["phase2_view_mode"] = "profile"
                st.session_state["phase2_selected"] = row["broker_key"]
                st.rerun()

    with st.expander("Se hele listen i tabellform"):
        display_df = ranking[["rank", "broker", "chain", "primary_location", "dominant_segment", "total_sales", "total_value", "avg_price"]].copy()
        display_df["total_value"] = display_df["total_value"].apply(fmt_compact_nok_with_kr)
        display_df["avg_price"] = display_df["avg_price"].apply(fmt_compact_nok_with_kr)
        display_df = display_df.rename(columns={
            "rank": "#",
            "broker": "Megler",
            "chain": "Kontor",
            "primary_location": "Bydel",
            "dominant_segment": "Segment",
            "total_sales": "Salg",
            "total_value": "Samlet verdi",
            "avg_price": "Snittpris",
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def _growth_last_periods(df_in: pd.DataFrame, days: int = 90) -> tuple[int, int]:
    if df_in is None or df_in.empty or "published_dt" not in df_in.columns:
        return 0, 0
    now = pd.Timestamp.utcnow()
    current_start = now - pd.Timedelta(days=days)
    prev_start = current_start - pd.Timedelta(days=days)
    current = df_in[df_in["published_dt"] >= current_start]
    prev = df_in[(df_in["published_dt"] >= prev_start) & (df_in["published_dt"] < current_start)]
    return int(len(current)), int(len(prev))


def render_phase2_profile(selected_key: str,
                          ranking: pd.DataFrame,
                          df_filtered: pd.DataFrame,
                          baseline_filtered: pd.DataFrame | None) -> None:
    if not selected_key:
        st.info("Ingen megler valgt.")
        return

    try:
        row = ranking.set_index("broker_key").loc[selected_key]
    except KeyError:
        st.warning("Valgt megler finnes ikke lenger i utvalget.")
        st.session_state["phase2_view_mode"] = "list"
        st.session_state["phase2_selected"] = None
        st.rerun()
        return
    broker_name = row["broker"]
    chain_name = row["chain"]

    back_col, _ = st.columns([0.3, 2.7])
    if back_col.button("← Tilbake til liste", key="phase2_back"):
        st.session_state["phase2_view_mode"] = "list"
        st.session_state["phase2_selected"] = None
        st.rerun()

    st.markdown("<div class=\"mm-section-title\">Meglerkort</div>", unsafe_allow_html=True)
    header_md = f"**{broker_name}**<br/>{chain_name}"
    st.markdown(header_md, unsafe_allow_html=True)
    meta_line = f"{row.get('broker_role', '(ukjent rolle)')} · {row.get('city', '')} — {row.get('primary_location', '')}"
    st.caption(meta_line)
    if row.get("high_volume"):
        st.success("Denne megleren er i høyvolum-segmentet for gjeldende filtrering.")

    broker_subset = df_filtered[(df_filtered["broker"] == broker_name) & (df_filtered["chain"] == chain_name)]
    baseline_subset = (
        baseline_filtered[(baseline_filtered["broker"] == broker_name) & (baseline_filtered["chain"] == chain_name)]
        if baseline_filtered is not None else pd.DataFrame()
    )

    segment_counts = (
        broker_subset["property_type"].value_counts().rename_axis("Segment").reset_index(name="Antall")
        if not broker_subset.empty else pd.DataFrame(columns=["Segment", "Antall"])
    )
    location_counts = (
        broker_subset["district"].value_counts().rename_axis("Bydel").reset_index(name="Antall")
        if not broker_subset.empty else pd.DataFrame(columns=["Bydel", "Antall"])
    )

    recent_now, recent_prev = _growth_last_periods(broker_subset, days=90)
    _, recent_delta_text = pct_change_label(recent_now, recent_prev, suffix="siste 90 dager vs foregående 90")

    total_sales = int(row.get("total_sales", 0))
    total_value = fmt_compact_nok_with_kr(row.get("total_value"))
    avg_price = fmt_compact_nok_with_kr(row.get("avg_price"))
    baseline_count = int(len(baseline_subset)) if not baseline_subset.empty else 0
    delta_vs_baseline = total_sales - baseline_count

    st.markdown("**Segmenter & lokasjoner**")
    overview_seg, overview_loc = st.columns(2)

    if segment_counts.empty:
        overview_seg.info("Ingen segmentdata i utvalget.")
    else:
        seg_total = int(segment_counts["Antall"].sum())
        top_segment = segment_counts.iloc[0]
        seg_share = (top_segment["Antall"] / seg_total * 100) if seg_total else 0
        seg_toplist = ", ".join(
            f"{row['Segment']} ({row['Antall']})" for _, row in segment_counts.head(3).iterrows()
        )
        overview_seg.markdown(
            f"""
            <div class="mm-card" style="padding:14px;">
              <div class="mm-title">Dominerende segment</div>
              <div style="font-size:18px;font-weight:700;color:#fff;">{top_segment['Segment']}</div>
              <div class="mm-subtle">{top_segment['Antall']} salg · {seg_share:.0f}% av porteføljen</div>
              <div class="mm-subtle" style="margin-top:6px;">Toppliste: {seg_toplist}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if location_counts.empty:
        overview_loc.info("Ingen bydelsdata tilgjengelig.")
    else:
        loc_total = int(location_counts["Antall"].sum())
        top_loc = location_counts.iloc[0]
        loc_share = (top_loc["Antall"] / loc_total * 100) if loc_total else 0
        loc_toplist = ", ".join(
            f"{row['Bydel']} ({row['Antall']})" for _, row in location_counts.head(3).iterrows()
        )
        overview_loc.markdown(
            f"""
            <div class="mm-card" style="padding:14px;">
              <div class="mm-title">Største lokasjon</div>
              <div style="font-size:18px;font-weight:700;color:#fff;">{top_loc['Bydel']}</div>
              <div class="mm-subtle">{top_loc['Antall']} salg · {loc_share:.0f}% av porteføljen</div>
              <div class="mm-subtle" style="margin-top:6px;">Toppliste: {loc_toplist}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Antall salg (utvalg)", total_sales, delta_vs_baseline)
    k2.metric("Samlet verdi", total_value)
    k3.metric("Snittpris", avg_price)
    k4.metric("Utvikling", f"{recent_now} salg", recent_delta_text)

    c_seg, c_loc = st.columns(2)
    if segment_counts.empty:
        c_seg.info("Ingen segmentdata i utvalget.")
    else:
        c_seg.markdown("**Segmentfordeling**")
        c_seg.dataframe(segment_counts, use_container_width=True)

    if location_counts.empty:
        c_loc.info("Ingen bydelsdata tilgjengelig.")
    else:
        c_loc.markdown("**Lokasjoner**")
        c_loc.dataframe(location_counts, use_container_width=True)

    price_col, trend_col = st.columns([1.1, 1.9])
    prices = broker_subset["price"].dropna()
    if prices.empty:
        price_col.info("Prisdata mangler for utvalget.")
    else:
        price_stats = pd.DataFrame(
            {
                "Nøkkel": ["Min", "Median", "Snitt", "Max"],
                "Verdi": [fmt_nok(prices.min()), fmt_nok(prices.median()), fmt_nok(prices.mean()), fmt_nok(prices.max())],
            }
        )
        price_col.markdown("**Prisnivåer**")
        price_col.dataframe(price_stats, use_container_width=True)

    if "published_dt" in broker_subset.columns and not broker_subset.empty:
        trend_col.markdown("**Salg over tid**")
        timeline = (
            broker_subset.dropna(subset=["published_dt"])
            .set_index("published_dt")
            .resample("M")["listing_id"].count()
        )
        if not timeline.empty:
            timeline.index = timeline.index.tz_convert("Europe/Oslo")
            trend_col.line_chart(timeline)
        else:
            trend_col.info("Ikke nok data for tidsserie.")
    else:
        trend_col.info("Ingen tidsstempler tilgjengelig for historikk.")

    st.markdown("**Peer comparison**")
    peers = ranking[
        (ranking["broker_key"] != selected_key)
        & (ranking["primary_location"] == row.get("primary_location"))
        & (ranking["dominant_segment"] == row.get("dominant_segment"))
    ].copy()
    if peers.empty:
        st.info("Ingen relevante peers funnet innen samme segment/bydel.")
    else:
        peer_display = peers.sort_values("total_value", ascending=False).head(5)
        peer_display = peer_display[["broker", "chain", "total_sales", "total_value", "avg_price"]]
        peer_display["total_value"] = peer_display["total_value"].apply(fmt_compact_nok_with_kr)
        peer_display["avg_price"] = peer_display["avg_price"].apply(fmt_compact_nok_with_kr)
        peer_display = peer_display.rename(columns={
            "broker": "Megler",
            "chain": "Kontor",
            "total_sales": "Salg",
            "total_value": "Verdi",
            "avg_price": "Snittpris",
        })
        st.dataframe(peer_display, use_container_width=True)

    st.markdown("**Du vil også like**")
    recommendations = ranking[ranking["broker_key"] != selected_key].copy()
    if recommendations.empty:
        st.info("Ingen andre meglere i rangeringen.")
    else:
        recommendations["segment_match"] = recommendations["dominant_segment"] == row.get("dominant_segment")
        recommendations["district_match"] = recommendations["primary_location"] == row.get("primary_location")
        recommendations["score"] = (
            recommendations["segment_match"].astype(int) * 2 + recommendations["district_match"].astype(int)
        )
        rec_top = recommendations.sort_values([
            "score", "total_value"
        ], ascending=[False, False]).head(3)
        if rec_top.empty:
            st.info("Ingen forslag tilgjengelig for nå.")
        else:
            for _, rec in rec_top.iterrows():
                st.markdown(
                    f"- **{rec['broker']}** ({rec['chain']}) – {rec['dominant_segment']} · {rec['primary_location']}"
                )

    st.markdown("**Profilinformasjon**")
    st.info("LinkedIn-integrasjon og erfaringsdata er ikke hentet ennå. Legg til `out/broker_profiles.csv` med kolonnene `broker`, `chain`, `linkedin_url`, `experience_years`, `age` for å berike kortet.")


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
COMMISSION_LABEL = f"{COMMISSION_RATE * 100:.2f}%"

MIN_LISTINGS_FOR_AVG = 5

PRICE_BANDS = [0, 5_000_000, 10_000_000, 15_000_000, 20_000_000, float('inf')]
PRICE_BAND_LABELS = ["0–5 mill", "5–10 mill", "10–15 mill", "15–20 mill", "20+ mill"]

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

#konverterer dato
def to_dt_safe(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype='datetime64[ns, UTC]')
    cleaned = s.astype(str).str.strip()
    cleaned = cleaned.replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA, 'NaT': pd.NA})
    return pd.to_datetime(cleaned, errors='coerce', utc=True, format='ISO8601')


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

#klargjør df
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
        price_band = pd.cut(out["price"], bins=PRICE_BANDS, labels=PRICE_BAND_LABELS, include_lowest=True, right=False)
        price_band = price_band.cat.add_categories(["(ukjent prissjikt)"]).fillna("(ukjent prissjikt)")
        out["price_band"] = price_band
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


#appliserer filtre
def apply_filters(df: pd.DataFrame | None,
                  city: str,
                  districts: list[str],
                  chains: list[str],
                  chain_keyword: str,
                  roles: list[str],
                  segments: list[str],
                  sources: list[str],
                  search: str,
                  period: str) -> pd.DataFrame:
    if df is None or df.empty:
        if isinstance(df, pd.DataFrame):
            return df.iloc[0:0].copy()
        return pd.DataFrame()

    subset = df.copy()

    if city != "(Alle)" and "city" in subset.columns:
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

    if "published_dt" in subset.columns:
        now_utc = pd.Timestamp.utcnow()
        if period == "Siste 30 dager":
            start = now_utc - pd.Timedelta(days=30)
            subset = subset[subset["published_dt"] >= start]
        elif period == "Siste 12 mnd":
            start = now_utc - pd.Timedelta(days=365)
            subset = subset[subset["published_dt"] >= start]
        elif period == "Dette året":
            start = pd.Timestamp(year=now_utc.tz_convert("Europe/Oslo").year, month=1, day=1, tz="Europe/Oslo").tz_convert("UTC")
            subset = subset[subset["published_dt"] >= start]

    return subset


def compute_change(current: float, baseline: float) -> tuple[float, float]:
    delta = current - baseline
    if baseline > 0:
        pct = (delta / baseline) * 100.0
    elif current > 0:
        pct = 100.0
    else:
        pct = 0.0
    return delta, pct


#tekst for pct
def pct_change_label(current: float, previous: float, suffix: str = "vs forrige periode") -> tuple[float | None, str]:
    if previous > 0:
        pct = ((current - previous) / previous) * 100.0
        return pct, f"{pct:+.1f}% {suffix}"
    if current == 0:
        return 0.0, f"0.0% {suffix}"
    return None, "Ingen data forrige periode"


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

#CSS
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
@media (max-width: 1200px) {
  .mm-kpi-row { grid-template-columns: repeat(2, 1fr); }
  .mm-card { max-width: 100%; }
}
@media (max-width: 768px) {
  .mm-kpi-row { grid-template-columns: 1fr; }
  .mm-row { flex-direction: column; align-items: flex-start; }
  .mm-row .mm-right { text-align: left; margin-top: 8px; }
  .mm-row .mm-rank { width: auto; margin-bottom: 6px; }
  .mm-card { padding: 12px; }
  .mm-section-title { font-size: 16px; }
  .mm-kpi-value-green, .mm-kpi-value-red { font-size: 24px; }
}
@media (max-width: 600px) {
  .mm-table { font-size: 13px; }
  .mm-th, .mm-td { padding: 8px 6px; }
}
</style>
""", unsafe_allow_html=True)

# -------------------- Load & clean data --------------------
df_raw = load_csv(OUT / "all_listings.csv")
if df_raw.empty:
    st.warning("Ingen data i `out/`. Kjør: `python -u megler_monitor_poc.py` først.")
    st.stop()

df = prepare_dataframe(df_raw)

baseline_raw, baseline_stamp = load_snapshot_csv("all_listings.csv", which="earliest")
if baseline_raw is not None and not baseline_raw.empty:
    baseline_df = prepare_dataframe(baseline_raw)
else:
    baseline_df = pd.DataFrame()
baseline_label = baseline_stamp or "første snapshot"

#Header
st.title("MeglerMonitor")
snapshot_ts = pd.Timestamp.now(tz="Europe/Oslo").strftime("%Y-%m-%d %H:%M")

#Filters
with st.expander("Filtre", expanded=True):
    st.caption("Bruk filtrene for å se hvem som selger mest i det området og segmentet du er opptatt av.")

    row1_col1, row1_col2 = st.columns([1.2, 1.2])
    cities = ["(Alle)"] + sorted(df["city"].dropna().unique().tolist())
    sel_city = row1_col1.selectbox("By", cities, index=0)

    district_series = df.get("district")
    if district_series is None:
        district_series = pd.Series(dtype="object")
    if sel_city != "(Alle)" and "district" in df.columns:
        base_mask = df["city"].eq(sel_city)
        district_series = df.loc[base_mask, "district"]
    district_clean = district_series.dropna() if isinstance(district_series, pd.Series) else pd.Series(dtype="object")
    district_options = sorted({d for d in district_clean if d and d != "(ukjent bydel)"})
    sel_districts = row1_col2.multiselect("Bydel", district_options, default=[])

    row2_col1, row2_col2 = st.columns([1.2, 1.2])
    segment_options = sorted(df["property_type"].dropna().unique().tolist()) if "property_type" in df.columns else []
    sel_segments = row2_col1.multiselect(
        "Boligsegment",
        segment_options,
        default=segment_options,
        help="Velg hvilke boligtyper som skal inngå i oversikten."
    )

    role_options = sorted(df["broker_role"].dropna().unique().tolist()) if "broker_role" in df.columns else []
    default_roles = role_options if role_options else []
    sel_roles = row2_col2.multiselect(
        "Meglerrolle/tittel",
        role_options,
        default=default_roles,
        help="Skiller f.eks. mellom meglere og fullmektiger."
    ) if role_options else []

    row3_col1, row3_col2, row3_col3 = st.columns([1.1, 1.1, 1.2])
    chains = sorted(df["chain"].dropna().unique().tolist())
    sel_chains = row3_col1.multiselect("Kjede/kontor", chains, default=[])
    chain_keyword = row3_col2.text_input("Kjedesøk", "", placeholder="F.eks. Nordvik")

    sources = sorted(df["source"].dropna().unique().tolist())
    sel_sources = row3_col3.multiselect("Datakilde", sources, default=sources)

    row4_col1, row4_col2, row4_col3 = st.columns([1.1, 1.1, 1.2])
    period = row4_col1.selectbox(
        "Tidsperiode",
        ["Alle", "Siste 30 dager", "Siste 12 mnd", "Dette året"],
        index=0
    )

    if {"broker", "listing_id"}.issubset(df.columns) and not df.empty:
        broker_counts = df.groupby("broker")["listing_id"].count()
        max_sales_available = int(broker_counts.max()) if not broker_counts.empty else 1
    else:
        max_sales_available = 1
    slider_max = max(max_sales_available, 5)
    min_sales = row4_col2.slider(
        "Min. antall salg",
        min_value=0,
        max_value=slider_max,
        value=0,
        help="Skjuler meglere med færre salg enn valgt terskel."
    )

    sort_metric_label = row4_col3.selectbox(
        "Sorter etter",
        ("Samlet verdi", "Antall salg", "Snittpris"),
        index=0,
        help="Bestem hvilket nøkkeltall meglerlisten skal rangeres etter."
    )

    search = st.text_input("Søk megler eller kontor", "", placeholder="Søk…")

sort_metric_map = {
    "Samlet verdi": "total_value",
    "Antall salg": "total_sales",
    "Snittpris": "avg_price",
}
sort_metric = sort_metric_map.get(sort_metric_label, "total_value")

flt = apply_filters(df, sel_city, sel_districts, sel_chains, chain_keyword, sel_roles, sel_segments, sel_sources, search, period)
baseline_flt = apply_filters(baseline_df, sel_city, sel_districts, sel_chains, chain_keyword, sel_roles, sel_segments, sel_sources, search, period)
flt = enforce_min_sales(flt, min_sales)
baseline_flt = enforce_min_sales(baseline_flt, min_sales)
baseline_has_data = not baseline_flt.empty

#KPI row (last 12m vs previous 12m
#KPI should always be computed on the CURRENT filter selection but with fixed windows (12m vs prev 12m)
now12, prev12 = split_windows_12m(flt, col="published_dt")

#Aktive annonser (siste 12 mnd)
listings_now = sales_count(now12)
listings_prev = sales_count(prev12)
listings_delta_val = listings_now - listings_prev
_, listings_delta_label = pct_change_label(listings_now, listings_prev)

#Total verdi av aktive boliger
omset_now = float(np.nan_to_num(now12["price"]).sum())
omset_prev = float(np.nan_to_num(prev12["price"]).sum())
omset_delta_val = omset_now - omset_prev
_, omset_delta_label = pct_change_label(omset_now, omset_prev)

commission_now = omset_now * COMMISSION_RATE
commission_prev = omset_prev * COMMISSION_RATE
commission_delta_val = commission_now - commission_prev
_, commission_delta_label = pct_change_label(commission_now, commission_prev, suffix=f"vs forrige periode ({COMMISSION_LABEL})")

active_now = int(now12["broker"].nunique()) if not now12.empty else 0
#Meglere med aktive annonser
active_prev = int(prev12["broker"].nunique()) if not prev12.empty else 0
active_delta_val = active_now - active_prev
_, active_delta_label = pct_change_label(active_now, active_prev)

#Gj.snitt dager i markedet – placeholder
days_now = "–"
days_delta_text = "–"

st.markdown('<div class="mm-kpi-row">', unsafe_allow_html=True)
kpi_card("Aktive annonser (siste 12 mnd)",
         f"{listings_now:,}".replace(",", " "),
         listings_delta_label,
         positive=(listings_delta_val >= 0))
kpi_card("Total verdi av aktive boliger",
         fmt_compact_nok(omset_now),
         omset_delta_label,
         positive=(omset_delta_val >= 0))
kpi_card("Meglere med aktive annonser",
         str(active_now),
         active_delta_label,
         positive=(active_delta_val >= 0))
kpi_card("Estimert provisjonsgrunnlag",
         fmt_compact_nok(commission_now),
         commission_delta_label,
         positive=(commission_delta_val >= 0))
st.markdown('</div>', unsafe_allow_html=True)
if baseline_has_data:
    st.markdown(f'<div class="mm-subtle">Første snapshot lagret: {baseline_label}</div>', unsafe_allow_html=True)
#TOPP 5 MEGLERE/KONTOR
CARD_STYLE = "width:100%;max-width:900px;margin:auto;"
colL, colR = st.columns([1, 1], gap="large")
brokers_total = int(flt["broker"].nunique()) if "broker" in flt.columns else 0
brokers_per_chain = (
    flt.groupby("chain")["broker"].nunique() if {"chain", "broker"}.issubset(flt.columns)
    else pd.Series(dtype="int64")
)

if {"broker", "chain", "price", "listing_id"}.issubset(flt.columns) and not flt.empty:
    brokers_grouped = (
        flt.groupby(["broker", "chain"], dropna=False)
        .agg(total_value=("price", "sum"), n=("listing_id", "count"))
        .reset_index()
    )
    brokers_grouped["commission_base"] = brokers_grouped["total_value"] * COMMISSION_RATE
    brokers_grouped["commission_avg"] = np.where(
        brokers_grouped["n"] > 0,
        brokers_grouped["commission_base"] / brokers_grouped["n"],
        0.0
    )
else:
    brokers_grouped = pd.DataFrame(columns=["broker", "chain", "total_value", "n", "commission_base", "commission_avg"])

if {"chain", "price", "listing_id"}.issubset(flt.columns) and not flt.empty:
    offices_grouped = (
        flt.groupby(["chain"], dropna=False)
        .agg(total_value=("price", "sum"), n=("listing_id", "count"))
        .reset_index()
    )
    offices_grouped["commission_base"] = offices_grouped["total_value"] * COMMISSION_RATE
    offices_grouped["commission_avg"] = np.where(
        offices_grouped["n"] > 0,
        offices_grouped["commission_base"] / offices_grouped["n"],
        0.0
    )
else:
    offices_grouped = pd.DataFrame(columns=["chain", "total_value", "n", "commission_base", "commission_avg"])

with colL:
    st.markdown(f'<div class="mm-card" style="{CARD_STYLE}">', unsafe_allow_html=True)
    st.markdown('<div class="mm-title">Størst provisjonsgrunnlag – meglere</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mm-subtle">Antall meglere i utvalget: {brokers_total} – sortert på estimert provisjon ({COMMISSION_LABEL})</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex;gap:16px;padding:0 8px 8px 8px;color:#aeb4be;font-size:13px;">
      <div style="width:40px;">#</div>
      <div style="flex:2;">Megler</div>
      <div style="flex:2;">Kontor</div>
      <div style="width:120px;text-align:center;">Meglere i kjeden</div>
      <div style="width:80px;text-align:center;">Aktive boliger</div>
      <div style="width:140px;text-align:right;">Samlet verdi</div>
      <div style="width:140px;text-align:right;">Provisjon ({COMMISSION_LABEL})</div>
      <div style="width:160px;text-align:right;">Snitt provisjon pr. bolig</div>
    </div>
    """, unsafe_allow_html=True)

    brokers_now = brokers_grouped.sort_values("commission_base", ascending=False).head(10)
    container_style = "max-height:540px;overflow:auto;padding-right:4px;"
    st.markdown(f'<div style="{container_style}">', unsafe_allow_html=True)
    for i, row in brokers_now.reset_index(drop=True).iterrows():
        rank = i + 1
        name = row["broker"]
        chain = row["chain"]
        count = int(row["n"])
        total = fmt_compact_nok_with_kr(row["total_value"])
        commission = fmt_compact_nok_with_kr(row["commission_base"])
        avg_raw = row.get("commission_avg")
        commission_avg = fmt_compact_nok_with_kr(avg_raw)
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
          <div style="width:140px;text-align:right;">
            <span style="background:#23262B;color:#ffd700;font-weight:700;padding:2px 12px;border-radius:999px;">{total}</span>
          </div>
          <div style="width:140px;text-align:right;">
            <span style="background:#23262B;color:#35d07f;font-weight:700;padding:2px 12px;border-radius:999px;">{commission}</span>
          </div>
          <div style="width:160px;text-align:right;">
            <span style="background:#23262B;color:#89a7ff;font-weight:700;padding:2px 12px;border-radius:999px;">{commission_avg}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)  # close scroll container
    st.markdown("</div>", unsafe_allow_html=True)  # close card

with colR:
    st.markdown(f'<div class="mm-card" style="{CARD_STYLE}">', unsafe_allow_html=True)
    st.markdown('<div class="mm-title">Størst provisjonsgrunnlag – kontorer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mm-subtle">Antall meglere i utvalget: {brokers_total} – sortert på estimert provisjon ({COMMISSION_LABEL})</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex;gap:16px;padding:0 8px 8px 8px;color:#aeb4be;font-size:13px;">
      <div style="width:40px;">#</div>
      <div style="flex:2;">Kontor</div>
      <div style="flex:2;">Kjede</div>
      <div style="width:120px;text-align:center;">Meglere i kjeden</div>
      <div style="width:80px;text-align:center;">Aktive boliger</div>
      <div style="width:140px;text-align:right;">Samlet verdi</div>
      <div style="width:140px;text-align:right;">Provisjon ({COMMISSION_LABEL})</div>
      <div style="width:160px;text-align:right;">Snitt provisjon pr. bolig</div>
    </div>
    """, unsafe_allow_html=True)

    offices_now = offices_grouped.sort_values("commission_base", ascending=False).head(10)
    st.markdown(f'<div style="{container_style}">', unsafe_allow_html=True)
    for i, row in offices_now.reset_index(drop=True).iterrows():
        rank = i + 1
        office = row["chain"]
        chain = row["chain"]
        count = int(row["n"])
        total = fmt_compact_nok_with_kr(row["total_value"])
        commission = fmt_compact_nok_with_kr(row["commission_base"])
        avg_raw = row.get("commission_avg")
        commission_avg = fmt_compact_nok_with_kr(avg_raw)
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
          <div style="width:140px;text-align:right;">
            <span style="background:#23262B;color:#ffd700;font-weight:700;padding:2px 12px;border-radius:999px;">{total}</span>
          </div>
          <div style="width:140px;text-align:right;">
            <span style="background:#23262B;color:#35d07f;font-weight:700;padding:2px 12px;border-radius:999px;">{commission}</span>
          </div>
          <div style="width:160px;text-align:right;">
            <span style="background:#23262B;color:#89a7ff;font-weight:700;padding:2px 12px;border-radius:999px;">{commission_avg}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Phase 2 broker ranking --------------------
st.markdown("<hr style='opacity:0.15;'>", unsafe_allow_html=True)
st.markdown("<div class=\"mm-section-title\">Fase 2 – Megleroversikt</div>", unsafe_allow_html=True)

phase2_ranking = build_broker_ranking(flt)
if min_sales > 0 and not phase2_ranking.empty:
    phase2_ranking = phase2_ranking[phase2_ranking["total_sales"] >= min_sales]

if not phase2_ranking.empty:
    phase2_sorted = phase2_ranking.sort_values(sort_metric, ascending=False).reset_index(drop=True)
    phase2_sorted["rank"] = phase2_sorted.index + 1
else:
    phase2_sorted = phase2_ranking

selected_key = st.session_state.get("phase2_selected")
valid_keys = set(phase2_sorted["broker_key"].tolist()) if "broker_key" in phase2_sorted else set()
if not valid_keys or selected_key not in valid_keys:
    st.session_state["phase2_view_mode"] = "list"
    st.session_state["phase2_selected"] = None
    selected_key = None

st.caption(f"Meglere i utvalget: {len(phase2_sorted)} (etter filtrering og terskler)")

if phase2_sorted.empty:
    st.info("Ingen meglere matcher filtrene. Juster bydel, segment eller min. antall salg.")
else:
    if st.session_state.get("phase2_view_mode") == "profile" and st.session_state.get("phase2_selected"):
        render_phase2_profile(st.session_state["phase2_selected"], phase2_sorted, flt, baseline_flt)
    else:
        render_phase2_list(phase2_sorted)

#summerer type
type_summary = (
    flt.groupby("property_type", dropna=False)
      .agg(antall=("listing_id", "count"), total=("price", "sum"), snitt=("price", "mean"))
      .reset_index()
)
if not type_summary.empty:
    type_summary["property_type"] = type_summary["property_type"].fillna("(ukjent boligtype)")
    type_summary = type_summary.sort_values("antall", ascending=False)

#summerer pris
price_summary = (
    flt.groupby("price_band", dropna=False)
      .agg(antall=("listing_id", "count"), total=("price", "sum"), snitt=("price", "mean"))
      .reset_index()
)
if not price_summary.empty:
    price_summary["price_band"] = price_summary["price_band"].astype(str)
    order = PRICE_BAND_LABELS + ["(ukjent prissjikt)"]
    price_summary["__order"] = price_summary["price_band"].apply(lambda x: order.index(x) if x in order else len(order))
    price_summary = price_summary.sort_values("__order").drop(columns="__order")

#summerer rolle
role_summary = (
    flt.groupby("broker_role", dropna=False)
      .agg(antall=("listing_id", "count"), total=("price", "sum"))
      .reset_index()
)
if not role_summary.empty:
    role_summary["broker_role"] = role_summary["broker_role"].fillna("(ukjent rolle)")
    role_summary = role_summary.sort_values("antall", ascending=False)

st.markdown('<div class="mm-section-title">Boligtyper og prisfordeling</div>', unsafe_allow_html=True)
type_col, price_col = st.columns([1, 1], gap="large")
with type_col:
    st.markdown('<div class="mm-card">', unsafe_allow_html=True)
    st.markdown('<div class="mm-title">Fordeling per boligtype</div>', unsafe_allow_html=True)
    if not type_summary.empty:
        type_display = type_summary.assign(
            Boligtype=lambda df: df["property_type"],
            Antall=lambda df: df["antall"],
            **{"Total verdi": type_summary["total"].apply(fmt_compact_nok_with_kr),
               "Snittpris": type_summary["snitt"].apply(fmt_compact_nok_with_kr)}
        )[
            ["Boligtype", "Antall", "Total verdi", "Snittpris"]
        ]
        st.dataframe(type_display, use_container_width=True, hide_index=True)
    else:
        st.info('Ingen boligtyper i utvalget.')
    st.markdown('</div>', unsafe_allow_html=True)

with price_col:
    st.markdown('<div class="mm-card">', unsafe_allow_html=True)
    st.markdown('<div class="mm-title">Fordeling per prissjikt</div>', unsafe_allow_html=True)
    if not price_summary.empty:
        price_display = price_summary.assign(
            Prissjikt=lambda df: df["price_band"],
            Antall=lambda df: df["antall"],
            **{"Total verdi": price_summary["total"].apply(fmt_compact_nok_with_kr),
               "Snittpris": price_summary["snitt"].apply(fmt_compact_nok_with_kr)}
        )[
            ["Prissjikt", "Antall", "Total verdi", "Snittpris"]
        ]
        st.dataframe(price_display, use_container_width=True, hide_index=True)
    else:
        st.info('Ingen prissjikt i utvalget.')
    st.markdown('</div>', unsafe_allow_html=True)

role_card = st.container()
with role_card:
    st.markdown('<div class="mm-card">', unsafe_allow_html=True)
    st.markdown('<div class="mm-title">Meglerroller</div>', unsafe_allow_html=True)
    if not role_summary.empty:
        role_display = role_summary.assign(
            Rolle=lambda df: df["broker_role"],
            Antall=lambda df: df["antall"],
            **{"Total verdi": role_summary["total"].apply(fmt_compact_nok_with_kr)}
        )[
            ["Rolle", "Antall", "Total verdi"]
        ]
        st.dataframe(role_display, use_container_width=True, hide_index=True)
    else:
        st.info('Ingen meglerroller i utvalget.')
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="mm-section-title">Høyest snittprovisjon</div>', unsafe_allow_html=True)
avg_colL, avg_colR = st.columns([1, 1], gap="large")

with avg_colL:
    st.markdown(f'<div class="mm-card" style="{CARD_STYLE}">', unsafe_allow_html=True)
    st.markdown('<div class="mm-title">Høyest snittprovisjon – meglere</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mm-subtle">Minst {MIN_LISTINGS_FOR_AVG} aktive boliger. Basert på estimert provisjon ({COMMISSION_LABEL}).</div>', unsafe_allow_html=True)
    avg_brokers = brokers_grouped[brokers_grouped["n"] >= MIN_LISTINGS_FOR_AVG].copy()
    if avg_brokers.empty:
        st.info(f"Ingen meglere med minst {MIN_LISTINGS_FOR_AVG} aktive boliger i utvalget.")
    else:
        st.markdown(f"""
        <div style="display:flex;gap:16px;padding:0 8px 8px 8px;color:#aeb4be;font-size:13px;">
          <div style="width:40px;">#</div>
          <div style="flex:2;">Megler</div>
          <div style="flex:2;">Kontor</div>
          <div style="width:120px;text-align:center;">Meglere i kjeden</div>
          <div style="width:80px;text-align:center;">Aktive boliger</div>
          <div style="width:150px;text-align:right;">Snitt provisjon</div>
          <div style="width:150px;text-align:right;">Total provisjon</div>
        </div>
        """, unsafe_allow_html=True)
        avg_container_style = "max-height:540px;overflow:auto;padding-right:4px;"
        st.markdown(f'<div style="{avg_container_style}">', unsafe_allow_html=True)
        top_avg_brokers = avg_brokers.sort_values("commission_avg", ascending=False).head(5)
        for i, row in top_avg_brokers.reset_index(drop=True).iterrows():
            rank = i + 1
            name = row["broker"]
            chain = row["chain"]
            count = int(row["n"])
            avg_text = fmt_compact_nok_with_kr(row.get("commission_avg"))
            total_text = fmt_compact_nok_with_kr(row.get("commission_base"))
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
              <div style="width:150px;text-align:right;">
                <span style="background:#23262B;color:#89a7ff;font-weight:700;padding:2px 12px;border-radius:999px;">{avg_text}</span>
              </div>
              <div style="width:150px;text-align:right;">
                <span style="background:#23262B;color:#35d07f;font-weight:700;padding:2px 12px;border-radius:999px;">{total_text}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with avg_colR:
    st.markdown(f'<div class="mm-card" style="{CARD_STYLE}">', unsafe_allow_html=True)
    st.markdown('<div class="mm-title">Høyest snittprovisjon – kontorer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mm-subtle">Minst {MIN_LISTINGS_FOR_AVG} aktive boliger. Basert på estimert provisjon ({COMMISSION_LABEL}).</div>', unsafe_allow_html=True)
    avg_offices = offices_grouped[offices_grouped["n"] >= MIN_LISTINGS_FOR_AVG].copy()
    if avg_offices.empty:
        st.info(f"Ingen kontorer med minst {MIN_LISTINGS_FOR_AVG} aktive boliger i utvalget.")
    else:
        st.markdown(f"""
        <div style="display:flex;gap:16px;padding:0 8px 8px 8px;color:#aeb4be;font-size:13px;">
          <div style="width:40px;">#</div>
          <div style="flex:2;">Kontor</div>
          <div style="width:120px;text-align:center;">Meglere i kjeden</div>
          <div style="width:80px;text-align:center;">Aktive boliger</div>
          <div style="width:150px;text-align:right;">Snitt provisjon</div>
          <div style="width:150px;text-align:right;">Total provisjon</div>
        </div>
        """, unsafe_allow_html=True)
        avg_container_style = "max-height:540px;overflow:auto;padding-right:4px;"
        st.markdown(f'<div style="{avg_container_style}">', unsafe_allow_html=True)
        top_avg_offices = avg_offices.sort_values("commission_avg", ascending=False).head(5)
        for i, row in top_avg_offices.reset_index(drop=True).iterrows():
            rank = i + 1
            office = row["chain"]
            count = int(row["n"])
            avg_text = fmt_compact_nok_with_kr(row.get("commission_avg"))
            total_text = fmt_compact_nok_with_kr(row.get("commission_base"))
            broker_count = int(brokers_per_chain.get(office, 0)) if not brokers_per_chain.empty else 0
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:16px;background:#161a20;border-radius:12px;padding:12px 8px;margin-bottom:8px;">
              <div style="width:40px;color:#9aa1ab;font-weight:600;">#{rank}</div>
              <div style="flex:2;display:flex;align-items:center;">
                <span class="mm-office-icon">🏢</span>
                <span style="font-weight:600; color:#fff;">{office}</span>
              </div>
              <div style="width:120px;text-align:center;">
                <span style="background:#23262B;color:#9aa1ab;font-weight:700;padding:2px 12px;border-radius:999px;">{broker_count}</span>
              </div>
              <div style="width:80px;text-align:center;">
                <span style="background:#23262B;color:#2f9e44;font-weight:700;padding:2px 12px;border-radius:999px;">{count}</span>
              </div>
              <div style="width:150px;text-align:right;">
                <span style="background:#23262B;color:#89a7ff;font-weight:700;padding:2px 12px;border-radius:999px;">{avg_text}</span>
              </div>
              <div style="width:150px;text-align:right;">
                <span style="background:#23262B;color:#35d07f;font-weight:700;padding:2px 12px;border-radius:999px;">{total_text}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")

#RANGERING: Vokser vs Faller (30d vs prev 30d)
st.markdown('<div class="mm-section-title">Endring i aktive boliger</div>', unsafe_allow_html=True)
cGrow, cFall = st.columns(2)

c8, _ = st.columns([1.2, 3])
portfolio_options = {
    "Siste 30 dager": 30,
    "Siste 60 dager": 60,
    "Siste 90 dager": 90,
    "Siste 180 dager": 180,
}
if baseline_has_data:
    portfolio_options["Siden første snapshot"] = None

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

if portfolio_window_days is None:
    now_win = flt.copy()
    prev_win = baseline_flt.copy()
    grow_desc = f"Størst økning i aktive boliger siden første snapshot ({baseline_label})"
    fall_desc = f"Størst nedgang i aktive boliger siden første snapshot ({baseline_label})"
else:
    now_win, prev_win = window_split(flt, now_days=portfolio_window_days)
    grow_desc = f"Størst økning i aktive boliger ({portfolio_window_days} dager vs forrige {portfolio_window_days} dager)"
    fall_desc = f"Størst nedgang i aktive boliger ({portfolio_window_days} dager vs forrige {portfolio_window_days} dager)"

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
    st.markdown(f'<div style="color:#aeb4be;font-size:13px;margin-bottom:10px;">{grow_desc}</div>', unsafe_allow_html=True)
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
    st.markdown(f'<div style="color:#aeb4be;font-size:13px;margin-bottom:10px;">{fall_desc}</div>', unsafe_allow_html=True)
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

#Raw table
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
