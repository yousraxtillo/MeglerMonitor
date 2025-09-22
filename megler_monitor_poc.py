import os, time, random
from datetime import datetime, timezone
from typing import Iterable, Optional, Tuple
import requests, pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import datetime as dt  # NY
import logging

logging.basicConfig(level=logging.INFO)

START_2024_TS = int(dt.datetime(2024, 1, 1).timestamp())
NOW_TS        = int(dt.datetime.now().timestamp())

def snapshot_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ---------- Utils ----------
OUT_DIR = "out"; os.makedirs(OUT_DIR, exist_ok=True)
UA = "MeglerMonitor/POC (+contact: yousra)"
def now_iso(): return datetime.now(timezone.utc).isoformat()
def jitter(a=0.3,b=0.9): time.sleep(random.uniform(a,b))
def save_csv(df, name): 
    p = os.path.join(OUT_DIR,name); df.to_csv(p, index=False)
    print(f"[ok] {name} ({len(df)} rader) lagret.")

def to_int(x):
    try: return int(float(str(x).replace("\u00a0","").replace(" ","").replace(",",".")))
    except: return None


def _normalize_case(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.title() if text.upper() == text else text


def _extract_dnb_location_fields(locs: Iterable) -> Tuple[list, Optional[str]]:
    values = []
    city = None
    street = None
    postal = None
    municipality = None

    for loc in locs or []:
        if isinstance(loc, dict):
            value = loc.get("value")
            loc_type = (loc.get("type") or "").upper()
        else:
            value = loc
            loc_type = ""
        if not value:
            continue
        value_str = str(value).strip()
        if not value_str:
            continue
        values.append(value_str)

        if loc_type in {"STREET", "ADDRESS"} and not street:
            street = value_str
        elif loc_type in {"ZIPCODE", "POSTALCODE"} and not postal:
            postal = value_str
        elif loc_type in {"CITY", "POSTALPLACE"} and not city:
            city = _normalize_case(value_str)
        elif loc_type in {"MUNICIPALITY", "AREA"} and not municipality:
            municipality = value_str

    if not city:
        candidates = []
        if municipality:
            candidates.append(municipality)
        if len(values) >= 3:
            candidates.append(values[2])
        candidates.extend(values)
        for candidate in candidates:
            cand = str(candidate).strip()
            if not cand or cand.lower() == "norge":
                continue
            if cand == street or cand == postal:
                continue
            if cand.replace(" ", "").isdigit():
                continue
            city = _normalize_case(cand)
            if city:
                break

    return values, city


DNB_STATUS_MAP = {
    0: "unknown",
    1: "coming",
    2: "available",
    3: "sold",
    4: "reserved",
    5: "inactive",
    99: "archived",
}


def _map_dnb_status(value):
    if value is None:
        return None
    try:
        num = int(value)
        return DNB_STATUS_MAP.get(num, value)
    except (TypeError, ValueError):
        return value


DOTNET_EPOCH_TICKS = 621355968000000000


def _ticks_to_iso8601(ticks: Optional[int | str]) -> Optional[str]:
    try:
        ticks_int = int(ticks)
    except (TypeError, ValueError):
        return None
    if ticks_int <= 0:
        return None
    unix_seconds = (ticks_int - DOTNET_EPOCH_TICKS) / 10_000_000
    if unix_seconds < 0:
        return None
    return datetime.fromtimestamp(unix_seconds, tz=timezone.utc).isoformat()


def _first_non_null(values: Iterable[Optional[str]]) -> Optional[str]:
    for value in values:
        if value:
            return value
    return None


def _extract_dnb_published(doc: dict, fallback: str) -> str:
    for_sale = doc.get("forSaleDate")
    created = doc.get("created")

    showings = doc.get("showings") or []
    showing_dates = sorted(
        [s.get("start") for s in showings if s and s.get("start")],
        key=lambda x: x
    )

    media = doc.get("media") or []
    media_dates = sorted(
        filter(None, (_ticks_to_iso8601(m.get("lastModified")) for m in media)),
        key=lambda x: x
    )

    primary = _first_non_null([
        for_sale,
        created,
        showing_dates[0] if showing_dates else None,
        media_dates[0] if media_dates else None,
    ])

    return primary or fallback

# robust session
session = requests.Session()
adapter = HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.6, status_forcelist=[429,500,502,503,504]))
session.mount("https://", adapter); session.mount("http://", adapter)

# ---------- HJEM.NO ----------
HJEM_URL = "https://apigw.hjem.no/search-backend/api/v4/property/search"
HJEM_HEADERS = {"User-Agent":UA,"Accept":"application/json","Content-Type":"application/json","Referer":"https://hjem.no/"}
HJEM_BASE_PAYLOAD = {"listing_type":"residential_sale","order":"desc","page":1,"size":50,"view":"list"}

def fetch_hjem_page(p):
    try:
        r = session.post(HJEM_URL, headers=HJEM_HEADERS, json=p, timeout=25)
        logging.info(f"[HJEM] page={p.get('page')} status={r.status_code}")
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        logging.error(f"Failed to fetch Hjem.no page {p.get('page')}: {e}")
        return {"data": []}

def parse_hjem_ad(ad):
    agency = ad.get("agency") or {}
    address = ad.get("address") or {}
    price = (ad.get("prices") or {}).get("asking_price",{}).get("amount")
    agents = [c.get("name") for c in ad.get("contacts",[]) if c.get("type")=="agent" and c.get("name")] or [None]
    rows=[]
    snap = snapshot_ts()
    for agent in agents:
        rows.append({
            "source":"Hjem.no",
            "listing_id": ad.get("id"),
            "title": ad.get("title"),
            "address": address.get("display_name"),
            "city": address.get("postal_place"),
            "chain": agency.get("name"),
            "broker": agent,
            "price": to_int(price),
            "status": ad.get("status"),
            "published": ad.get("publish_date"),
            "snapshot_at": snap,          # 👈 nytt felt
            "last_seen_at": now_iso()
        })
    return rows


def collect_hjem():
    """Fetches and parses property listings from Hjem.no."""
    rows = []
    page = 1
    size = HJEM_BASE_PAYLOAD["size"]

    base = dict(
        HJEM_BASE_PAYLOAD,
        publish_date_min=START_2024_TS,
        publish_date_max=NOW_TS
    )
    print(f"[HJEM] daterange min={base['publish_date_min']} max={base['publish_date_max']}")

    while True:
        payload = dict(base, page=page)
        data = fetch_hjem_page(payload)
        ads = data.get("data", [])
        if not ads:
            break
        for ad in ads:
            rows += parse_hjem_ad(ad)
        if len(ads) < size:
            break
        page += 1
        jitter()
    return pd.DataFrame(rows)


# ---------- DNB EIENDOM ----------
DNB_URL = "https://dnbeiendom.no/api/v1/cognitivesearch/properties"
DNB_HEADERS = {"User-Agent":UA,"Accept":"application/json","Content-Type":"application/json","Referer":"https://dnbeiendom.no/"}
# Fra DevTools (du fant payloaden): legg til "brokers" i select for å få meglernavn
DNB_BASE_PAYLOAD = {
    "facets": [],
    "filter": "(status eq 2 and projectRelation eq 3 or (projectRelation eq 1 and status ne 99)) and status ne 3 and status ne null and not (projectRelation eq 1 and status eq 99) and not (projectRelation eq 2 and status eq 99)",
    "orderBy": ["forSaleDate desc","created desc"],
    "select": [
        "id","size","area","units","areas","noOfBedRooms","propertyBaseType","propertyTypeId",
        "ownership","heading","showings","assignmentNum","forSaleDate","created","status",
        "locations","media","price","brokers"
    ],
    "skip": 0,
    "top": 24
}

def fetch_dnb_page(skip, top):
    payload = dict(DNB_BASE_PAYLOAD, skip=skip, top=top)
    r = session.post(DNB_URL, headers=DNB_HEADERS, json=payload, timeout=25)
    print(f"[DNB] skip={skip} top={top} status={r.status_code}")
    r.raise_for_status(); return r.json()

def parse_dnb_ad(doc):
    price_obj = doc.get("price") or {}
    price = price_obj.get("salePrice") or price_obj.get("askingPrice") or price_obj.get("totalPrice")
    loc_values, city = _extract_dnb_location_fields(doc.get("locations"))
    address = ", ".join(loc_values) if loc_values else None
    brokers = [b.get("name") for b in (doc.get("brokers") or []) if b.get("name")] or [None]
    rows=[]
    snap = snapshot_ts()
    published = _extract_dnb_published(doc, snap)
    for br in brokers:
        rows.append({
            "source":"DNB",
            "listing_id": doc.get("id"),
            "title": doc.get("heading"),
            "address": address,
            "city": city,
            "chain": "DNB Eiendom",
            "broker": br,
            "price": to_int(price),
            "status": _map_dnb_status(doc.get("status")),
            "published": published,
            "snapshot_at": snap,          # 👈 nytt felt
            "last_seen_at": now_iso()
        })
    return rows

def collect_dnb():
    rows = []
    top = DNB_BASE_PAYLOAD["top"]
    skip = 0
    total = None

    while True:
        data = fetch_dnb_page(skip, top)
        docs = data.get("documents", [])
        if not docs:
            break
        for doc in docs:
            rows += parse_dnb_ad(doc)

        skip += top
        total = data.get("totalCount") or total

        if total and skip >= total:
            break
        if len(docs) < top:
            break

        jitter()

    return pd.DataFrame(rows)

# ---------- Aggregation ----------
def sum_per_broker(df):
    if df.empty: return df
    df = df.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    return (df.groupby(["source","chain","broker"], dropna=False)["price"]
              .sum().reset_index().rename(columns={"price":"sum_price"})
              .sort_values("sum_price", ascending=False))

# ---------- Main ----------
if __name__ == "__main__":
    print("[RUN] MeglerMonitor POC – Hjem.no + DNB")

    hjem_df = collect_hjem();      save_csv(hjem_df, "hjem_listings.csv")
    dnb_df  = collect_dnb();       save_csv(dnb_df,  "dnb_listings.csv")

    both = pd.concat([hjem_df, dnb_df], ignore_index=True)
    save_csv(both, "all_listings.csv")

    agg = sum_per_broker(both);    save_csv(agg, "agg_sum_per_broker.csv")

    print("[DONE]")
