from __future__ import annotations

import pandas as pd
import streamlit as st

from poc_data import (
    RefreshResult,
    format_compact_number,
    load_all_listings,
    load_broker_totals,
    refresh_data,
)

st.set_page_config(page_title="MeglerMonitor POC", layout="wide")
st.title("MeglerMonitor – Proof of Concept")
st.caption("En rask visning av meglerdata fra DNB Eiendom og Hjem.no.")


def _render_refresh_button() -> None:
    status_box = st.empty()
    if st.button("Oppdater data nå", use_container_width=True):
        with st.spinner("Henter ferske data fra kildene..."):
            try:
                result: RefreshResult = refresh_data()
            except Exception as exc:  # noqa: BLE001 - vis feilen for manuell feilsøking
                status_box.error(f"Oppdatering feilet: {exc}")
            else:
                message = (
                    f"Oppdatert {format_compact_number(result.combined_rows)} annonser "
                    f"fra {format_compact_number(result.hjem_rows)} Hjem.no og "
                    f"{format_compact_number(result.dnb_rows)} DNB-poster. "
                    f"Aggregert {format_compact_number(result.agg_rows)} meglerrader."
                )
                status_box.success(message)


def _compute_summary_metrics(agg_df: pd.DataFrame, listings_df: pd.DataFrame) -> None:
    total_value = agg_df["sum_price"].sum()
    total_listings = len(listings_df)
    unique_brokers = agg_df["broker"].nunique(dropna=True)

    col_total, col_listings, col_brokers = st.columns(3)
    col_total.metric("Samlet verdi ute til salgs", format_compact_number(total_value) + " kr")
    col_listings.metric("Aktive annonser", format_compact_number(total_listings))
    col_brokers.metric("Meglere i datasettet", format_compact_number(unique_brokers))


def _render_top_lists(agg_df: pd.DataFrame) -> None:
    working = agg_df.copy()
    working["broker"] = working["broker"].fillna("(Ukjent)")
    working["chain"] = working["chain"].fillna("(Ukjent)")
    working["sum_price_fmt"] = working["sum_price"].apply(format_compact_number)
    working["commission_base_fmt"] = working["commission_base"].apply(format_compact_number)
    working["listing_count_fmt"] = working["listing_count"].apply(format_compact_number)

    cols_to_show = [
        "broker",
        "chain",
        "sum_price_fmt",
        "listing_count_fmt",
        "commission_base_fmt",
    ]
    renamed = working.sort_values("sum_price", ascending=False)[cols_to_show].rename(
        columns={
            "broker": "Megler",
            "chain": "Meglerkjede",
            "sum_price_fmt": "Total verdi",
            "listing_count_fmt": "Antall boliger",
            "commission_base_fmt": "Estimert provisjon",
        }
    )

    st.subheader("Toppmeglere etter totalt volum")
    st.dataframe(
        renamed.head(30),
        use_container_width=True,
        hide_index=True,
    )

    chain_summary = (
        agg_df.copy()
        .fillna({"chain": "(Ukjent)"})
        .groupby("chain", dropna=False)
        .agg(
            total_value=("sum_price", "sum"),
            avg_price=("sum_price", "mean"),
            brokers=("broker", "nunique"),
            listings=("listing_count", "sum"),
        )
        .reset_index()
        .sort_values("total_value", ascending=False)
    )

    chain_summary["total_value_fmt"] = chain_summary["total_value"].apply(format_compact_number)
    chain_summary["listings_fmt"] = chain_summary["listings"].apply(format_compact_number)
    chain_summary["brokers_fmt"] = chain_summary["brokers"].apply(format_compact_number)

    chain_display = chain_summary[
        ["chain", "total_value_fmt", "listings_fmt", "brokers_fmt"]
    ].rename(
        columns={
            "chain": "Meglerkjede",
            "total_value_fmt": "Total verdi",
            "listings_fmt": "Antall boliger",
            "brokers_fmt": "Antall meglere",
        }
    )

    st.subheader("Meglerkjeder")
    st.dataframe(chain_display.head(15), use_container_width=True, hide_index=True)


def main() -> None:
    _render_refresh_button()

    listings_df = load_all_listings()
    agg_df = load_broker_totals()

    if agg_df.empty:
        st.info("Ingen aggregator-data ennå. Kjør en oppdatering for å hente første datasett.")
        return

    _compute_summary_metrics(agg_df, listings_df)

    top_row = agg_df.sort_values("sum_price", ascending=False).head(1)
    if not top_row.empty:
        row = top_row.iloc[0]
        st.success(
            f"Størst volum akkurat nå: **{row.get('broker', '(Ukjent)')}** "
            f"hos **{row.get('chain', '(Ukjent)')}** – "
            f"{format_compact_number(row.get('sum_price'))} kr fordelt på "
            f"{format_compact_number(row.get('listing_count'))} boliger."
        )

    _render_top_lists(agg_df)


if __name__ == "__main__":
    main()
