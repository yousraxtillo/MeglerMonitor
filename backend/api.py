from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from data_service import (
    FilterParams,
    broker_detail_response,
    get_filter_options,
    overview_response,
    ranking_response,
)

app = FastAPI(
    title="MeglerMonitor API",
    version="0.1.0",
    description="API som leverer data til MeglerMonitor-dashboardet.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/filters")
def filters(
    city: str | None = Query(default=None),
    refresh: bool = Query(default=False),
) -> dict[str, list[str]]:
    return get_filter_options(city=city, refresh=refresh)


@app.get("/overview")
def overview(refresh: bool = Query(default=False)) -> dict[str, object]:
    return overview_response(refresh=refresh)


def build_params(
    city: str | None,
    districts: list[str] | None,
    chains: list[str] | None,
    chain_keyword: str | None,
    roles: list[str] | None,
    segments: list[str] | None,
    sources: list[str] | None,
    search: str | None,
    period: str,
    min_sales: int,
    sort_by: str,
) -> FilterParams:
    return FilterParams(
        city=city,
        districts=tuple(districts or ()),
        chains=tuple(chains or ()),
        chain_keyword=chain_keyword or "",
        roles=tuple(roles or ()),
        segments=tuple(segments or ()),
        sources=tuple(sources or ()),
        search=search or "",
        period=period,
        min_sales=min_sales,
        sort_by=sort_by,
    )


@app.get("/brokers")
def brokers(
    city: str | None = Query(default=None),
    districts: list[str] | None = Query(default=None),
    chains: list[str] | None = Query(default=None),
    chain_keyword: str | None = Query(default=None),
    roles: list[str] | None = Query(default=None),
    segments: list[str] | None = Query(default=None),
    sources: list[str] | None = Query(default=None),
    search: str | None = Query(default=None),
    period: str = Query(default="Alle"),
    min_sales: int = Query(default=0, ge=0),
    sort_by: str = Query(default="total_value"),
    refresh: bool = Query(default=False),
) -> dict[str, object]:
    params = build_params(
        city,
        districts,
        chains,
        chain_keyword,
        roles,
        segments,
        sources,
        search,
        period,
        min_sales,
        sort_by,
    )
    try:
        return ranking_response(params, refresh=refresh)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/brokers/{broker_key}")
def broker_detail(
    broker_key: str,
    city: str | None = Query(default=None),
    districts: list[str] | None = Query(default=None),
    chains: list[str] | None = Query(default=None),
    chain_keyword: str | None = Query(default=None),
    roles: list[str] | None = Query(default=None),
    segments: list[str] | None = Query(default=None),
    sources: list[str] | None = Query(default=None),
    search: str | None = Query(default=None),
    period: str = Query(default="Alle"),
    min_sales: int = Query(default=0, ge=0),
    sort_by: str = Query(default="total_value"),
    refresh: bool = Query(default=False),
) -> dict[str, object]:
    params = build_params(
        city,
        districts,
        chains,
        chain_keyword,
        roles,
        segments,
        sources,
        search,
        period,
        min_sales,
        sort_by,
    )
    detail = broker_detail_response(broker_key, params, refresh=refresh)
    if detail is None:
        raise HTTPException(status_code=404, detail="Megler finnes ikke for gjeldende filtrering.")
    return detail
