"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import styles from "./page.module.css";

type FilterCollection = {
  cities: string[];
  districts: string[];
  segments: string[];
  roles: string[];
  chains: string[];
  sources: string[];
};

type FilterState = {
  city: string;
  districts: string[];
  chains: string[];
  chainKeyword: string;
  roles: string[];
  segments: string[];
  sources: string[];
  search: string;
  period: string;
  minSales: number;
  sortBy: string;
};

type FilterChip =
  | { key: "city"; label: string }
  | { key: "district"; value: string; label: string }
  | { key: "segment"; value: string; label: string }
  | { key: "role"; value: string; label: string }
  | { key: "chain"; value: string; label: string }
  | { key: "source"; value: string; label: string }
  | { key: "minSales"; label: string }
  | { key: "search"; label: string };

type BrokerItem = {
  rank: number;
  broker_key: string;
  broker: string;
  chain: string;
  city: string;
  primary_location: string;
  dominant_segment: string;
  total_sales: number;
  total_value: number;
  avg_price: number;
  high_volume: boolean;
  segment_summary: string;
  location_summary: string;
  broker_role: string;
  latest_listing?: string | null;
};

type CommissionBrokerRow = {
  broker: string | null;
  chain: string | null;
  listing_count: number;
  total_value: number;
  commission: number;
  commission_avg: number;
  chain_broker_count: number;
};

type CommissionOfficeRow = {
  office: string | null;
  chain: string | null;
  listing_count: number;
  total_value: number;
  commission: number;
  commission_avg: number;
  chain_broker_count: number;
};

type BrokerSortKey = "commission" | "total_value" | "listing_count" | "commission_avg";
type OfficeSortKey = "commission" | "total_value" | "listing_count" | "chain_broker_count";
type TopLimitOption = number | "all";

type PortfolioDelta = {
  broker: string | null;
  chain: string | null;
  delta_value: number;
  delta_pct: number;
  value_now: number;
};

type RankingResponse = {
  filters: Record<string, unknown>;
  generated_at: string;
  total_brokers: number;
  items: BrokerItem[];
  top_commission_brokers: CommissionBrokerRow[];
  top_commission_offices: CommissionOfficeRow[];
  portfolio_gainers: PortfolioDelta[];
  portfolio_losers: PortfolioDelta[];
  kpi: {
    listings_now: number;
    listings_previous: number;
    listings_pct_change: number | null;
    total_value_now: number;
    total_value_previous: number;
    total_value_pct_change: number | null;
    commission_now: number;
    commission_previous: number;
    commission_pct_change: number | null;
    active_brokers_now: number;
    active_brokers_previous: number;
    active_brokers_pct_change: number | null;
  };
};

type OverviewDistribution = {
  label: string;
  count: number;
  share: number;
};

type OverviewKpi = {
  listings_now: number;
  listings_previous: number;
  listings_pct_change: number | null;
  total_value_now: number;
  total_value_previous: number;
  total_value_pct_change: number | null;
  commission_now: number;
  commission_previous: number;
  commission_pct_change: number | null;
  active_brokers_now: number;
  active_brokers_previous: number;
  active_brokers_pct_change: number | null;
};

type OverviewDatasets = {
  all_listings: number;
  agg_sum_per_broker: number;
  dnb_listings: number;
  hjem_listings: number;
  broker_profiles: number;
};

type OverviewTimelinePoint = {
  month: string;
  listings: number;
  value: number;
};

type OverviewResponse = {
  generated_at: string;
  broker_count: number;
  high_volume: number;
  kpi: OverviewKpi;
  datasets: OverviewDatasets;
  timeline: OverviewTimelinePoint[];
  segments: OverviewDistribution[];
  locations: OverviewDistribution[];
  chains: OverviewDistribution[];
};

type BrokerDetail = {
  broker_key: string;
  broker: string;
  chain: string;
  broker_role?: string;
  city?: string;
  primary_location?: string;
  dominant_segment?: string;
  high_volume: boolean;
  segment_summary?: string;
  location_summary?: string;
  metrics: {
    total_sales: number;
    total_value: number;
    avg_price: number;
    recent_sales: { current: number; previous: number };
    commission_estimate: number;
  };
  segments: { segment: string; count: number }[];
  locations: { district: string; count: number }[];
  price_stats: { min: number | null; median: number | null; mean: number | null; max: number | null };
  timeline: { month: string; listings: number; value: number }[];
  peers: { broker_key: string; broker: string; chain: string; total_sales: number; total_value: number; avg_price: number }[];
  recommendations: { broker_key: string; broker: string; chain: string; dominant_segment: string; primary_location: string; total_value: number; score: number }[];
  profile: { linkedin_url: string | null; experience_years: number | null; age: number | null } | null;
  filters: Record<string, unknown>;
};

const API_BASE = process.env.NEXT_PUBLIC_MM_API ?? "http://localhost:8000";

const DEFAULT_FILTERS: FilterState = {
  city: "",
  districts: [],
  chains: [],
  chainKeyword: "",
  roles: [],
  segments: [],
  sources: [],
  search: "",
  period: "Alle",
  minSales: 0,
  sortBy: "total_value",
};

const periodOptions = ["Alle", "Siste 30 dager", "Siste 12 mnd", "Dette året"];
const sortOptions = [
  { value: "total_value", label: "Samlet verdi" },
  { value: "total_sales", label: "Antall salg" },
  { value: "avg_price", label: "Snittpris" },
  { value: "latest_listing", label: "Siste publisert" },
];
const topOptions = [12, 24, 48, 96];
const topLimitOptions: TopLimitOption[] = [...topOptions, "all"];
const topBrokerSortOptions: { value: BrokerSortKey; label: string }[] = [
  { value: "commission", label: "Estimert provisjon" },
  { value: "total_value", label: "Samlet verdi" },
  { value: "listing_count", label: "Antall boliger" },
  { value: "commission_avg", label: "Snitt provisjon" },
];
const topOfficeSortOptions: { value: OfficeSortKey; label: string }[] = [
  { value: "commission", label: "Estimert provisjon" },
  { value: "total_value", label: "Samlet verdi" },
  { value: "listing_count", label: "Antall boliger" },
  { value: "chain_broker_count", label: "Antall meglere" },
];

const currencyFormatter = new Intl.NumberFormat("nb-NO", {
  style: "currency",
  currency: "NOK",
  maximumFractionDigits: 0,
});

const compactFormatter = new Intl.NumberFormat("nb-NO", {
  notation: "compact",
  compactDisplay: "short",
  maximumFractionDigits: 1,
});

function formatCurrency(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "–";
  }
  return currencyFormatter.format(value).replace(/\u00a0/g, " ");
}

function formatCompact(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "–";
  }
  return compactFormatter.format(value).replace(/\u00a0/g, " ");
}

function formatCompactWithKr(value: number | null | undefined) {
  const base = formatCompact(value);
  return base === "–" ? base : `${base} kr`;
}

function formatSignedCurrency(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "–";
  }
  if (value === 0) {
    return formatCurrency(0);
  }
  const sign = value >= 0 ? "+" : "−";
  const formatted = formatCurrency(Math.abs(value));
  return `${sign}${formatted}`;
}

function formatSignedPercent(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "–";
  }
  if (value === 0) {
    return "0.0 %";
  }
  const sign = value > 0 ? "+" : value < 0 ? "−" : "";
  return `${sign}${Math.abs(value).toFixed(1)} %`;
}

function pctLabel(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "Ingen data";
  }
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(1)} %`;
}

function formatPercentValue(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "–";
  }
  return `${Math.round(value * 100)}%`;
}

function cloneFilters(value: FilterState): FilterState {
  return {
    ...value,
    districts: [...value.districts],
    chains: [...value.chains],
    roles: [...value.roles],
    segments: [...value.segments],
    sources: [...value.sources],
  };
}

function buildQuery(params: FilterState): URLSearchParams {
  const query = new URLSearchParams();
  if (params.city) query.append("city", params.city);
  params.districts.forEach((item) => query.append("districts", item));
  params.chains.forEach((item) => query.append("chains", item));
  if (params.chainKeyword) query.append("chain_keyword", params.chainKeyword);
  params.roles.forEach((item) => query.append("roles", item));
  params.segments.forEach((item) => query.append("segments", item));
  params.sources.forEach((item) => query.append("sources", item));
  if (params.search) query.append("search", params.search);
  query.append("period", params.period);
  query.append("min_sales", String(params.minSales));
  query.append("sort_by", params.sortBy);
  return query;
}

export default function Home() {
  const [filterOptions, setFilterOptions] = useState<FilterCollection>({
    cities: [],
    districts: [],
    segments: [],
    roles: [],
    chains: [],
    sources: [],
  });
  const [pendingFilters, setPendingFilters] =
    useState<FilterState>(DEFAULT_FILTERS);
  const [appliedFilters, setAppliedFilters] =
    useState<FilterState>(DEFAULT_FILTERS);
  const [ranking, setRanking] = useState<RankingResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [overviewLoading, setOverviewLoading] = useState<boolean>(true);
  const [showOverview, setShowOverview] = useState<boolean>(true);
  const [hasApplied, setHasApplied] = useState<boolean>(false);
  const [districtQuery, setDistrictQuery] = useState<string>("");
  const [segmentQuery, setSegmentQuery] = useState<string>("");
  const [roleQuery, setRoleQuery] = useState<string>("");
  const [chainQuery, setChainQuery] = useState<string>("");
  const [sourceQuery, setSourceQuery] = useState<string>("");
  const [selectedBrokerKey, setSelectedBrokerKey] = useState<string | null>(null);
  const [brokerDetail, setBrokerDetail] = useState<BrokerDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState<boolean>(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [brokerSearch, setBrokerSearch] = useState<string>("");
  const [commissionSearch, setCommissionSearch] = useState<string>("");
  const [topBrokerSort, setTopBrokerSort] = useState<BrokerSortKey>("commission");
  const [topOfficeSort, setTopOfficeSort] = useState<OfficeSortKey>("commission");
  const [topBrokerLimit, setTopBrokerLimit] = useState<TopLimitOption>(topOptions[0]);
  const [topOfficeLimit, setTopOfficeLimit] = useState<TopLimitOption>(topOptions[0]);
  const [showAllSegments, setShowAllSegments] = useState<boolean>(false);
  const [showAllLocations, setShowAllLocations] = useState<boolean>(false);
  const [showAllPeers, setShowAllPeers] = useState<boolean>(false);
  const [showAllRecommendations, setShowAllRecommendations] = useState<boolean>(false);

  const rankedItems = useMemo(() => ranking?.items ?? [], [ranking]);
  const topCommissionBrokers = useMemo(
    () => ranking?.top_commission_brokers ?? [],
    [ranking]
  );
  const topCommissionOffices = useMemo(
    () => ranking?.top_commission_offices ?? [],
    [ranking]
  );
  const portfolioGainers = useMemo(
    () => ranking?.portfolio_gainers ?? [],
    [ranking]
  );
  const portfolioLosers = useMemo(
    () => ranking?.portfolio_losers ?? [],
    [ranking]
  );
  const openBrokerDetail = useCallback((brokerKey: string) => {
    setSelectedBrokerKey(brokerKey);
    setBrokerDetail(null);
    setDetailError(null);
  }, []);

  const brokerKeyLookup = useMemo(() => {
    const map = new Map<string, string>();
    rankedItems.forEach((item) => {
      const brokerKey = item.broker_key;
      const broker = (item.broker ?? "").toLowerCase();
      const chain = (item.chain ?? "").toLowerCase();
      if (brokerKey) {
        map.set(`${broker}||${chain}`, brokerKey);
      }
    });
    return map;
  }, [rankedItems]);

  const resolveBrokerKey = (broker?: string | null, chain?: string | null) => {
    if (!broker) return null;
    const key = `${broker.toLowerCase()}||${(chain ?? "").toLowerCase()}`;
    return brokerKeyLookup.get(key) ?? null;
  };

  const handleQuickSearch = useCallback(() => {
    const query = brokerSearch.trim().toLowerCase();
    if (!query || rankedItems.length === 0) {
      return;
    }
    const match = rankedItems.find((item) => {
      const broker = (item.broker ?? "").toLowerCase();
      const chain = (item.chain ?? "").toLowerCase();
      const location = (item.primary_location ?? "").toLowerCase();
      return (
        broker.includes(query) ||
        chain.includes(query) ||
        location.includes(query)
      );
    });
    if (match?.broker_key) {
      openBrokerDetail(match.broker_key);
    }
  }, [brokerSearch, rankedItems, openBrokerDetail]);

  const brokerSortFnMap: Record<BrokerSortKey, (row: CommissionBrokerRow) => number> = useMemo(
    () => ({
      commission: (row) => row.commission ?? 0,
      total_value: (row) => row.total_value ?? 0,
      listing_count: (row) => row.listing_count ?? 0,
      commission_avg: (row) => row.commission_avg ?? 0,
    }),
    []
  );

  const officeSortFnMap: Record<OfficeSortKey, (row: CommissionOfficeRow) => number> = useMemo(
    () => ({
      commission: (row) => row.commission ?? 0,
      total_value: (row) => row.total_value ?? 0,
      listing_count: (row) => row.listing_count ?? 0,
      chain_broker_count: (row) => row.chain_broker_count ?? 0,
    }),
    []
  );

  const sortedTopCommissionBrokers = useMemo(() => {
    const copy = [...topCommissionBrokers];
    copy.sort((a, b) => brokerSortFnMap[topBrokerSort](b) - brokerSortFnMap[topBrokerSort](a));
    return copy;
  }, [topCommissionBrokers, brokerSortFnMap, topBrokerSort]);

  const sortedTopCommissionOffices = useMemo(() => {
    const copy = [...topCommissionOffices];
    copy.sort((a, b) => officeSortFnMap[topOfficeSort](b) - officeSortFnMap[topOfficeSort](a));
    return copy;
  }, [topCommissionOffices, officeSortFnMap, topOfficeSort]);

  const commissionSearchTerm = commissionSearch.trim().toLowerCase();

  const displayedTopCommissionBrokers = useMemo(() => {
    const filtered = sortedTopCommissionBrokers.filter((row) => {
      if (!commissionSearchTerm) return true;
      const name = (row.broker ?? "").toLowerCase();
      const chain = (row.chain ?? "").toLowerCase();
      return name.includes(commissionSearchTerm) || chain.includes(commissionSearchTerm);
    });
    const limit = topBrokerLimit === "all" ? filtered.length : Number(topBrokerLimit) || filtered.length;
    return filtered.slice(0, limit);
  }, [sortedTopCommissionBrokers, commissionSearchTerm, topBrokerLimit]);

  const displayedTopCommissionOffices = useMemo(() => {
    const filtered = sortedTopCommissionOffices.filter((row) => {
      if (!commissionSearchTerm) return true;
      const office = (row.office ?? "").toLowerCase();
      const chain = (row.chain ?? "").toLowerCase();
      return office.includes(commissionSearchTerm) || chain.includes(commissionSearchTerm);
    });
    const limit = topOfficeLimit === "all" ? filtered.length : Number(topOfficeLimit) || filtered.length;
    return filtered.slice(0, limit);
  }, [sortedTopCommissionOffices, commissionSearchTerm, topOfficeLimit]);

  const brokerSegments = brokerDetail?.segments ?? [];
  const brokerLocations = brokerDetail?.locations ?? [];
  const brokerPeers = brokerDetail?.peers ?? [];
  const brokerRecommendations = brokerDetail?.recommendations ?? [];
  const displayedSegments = showAllSegments ? brokerSegments : brokerSegments.slice(0, 8);
  const displayedLocations = showAllLocations ? brokerLocations : brokerLocations.slice(0, 10);
  const displayedPeers = showAllPeers ? brokerPeers : brokerPeers.slice(0, 6);
  const displayedRecommendations = showAllRecommendations
    ? brokerRecommendations
    : brokerRecommendations.slice(0, 6);
  const segmentsToggleable = brokerSegments.length > 8;
  const locationsToggleable = brokerLocations.length > 10;
  const peersToggleable = brokerPeers.length > 6;
  const recommendationsToggleable = brokerRecommendations.length > 6;

  const loadFilters = useCallback(async (city?: string) => {
    try {
      const query = city ? `?city=${encodeURIComponent(city)}` : "";
      const res = await fetch(`${API_BASE}/filters${query}`);
      if (!res.ok) {
        throw new Error("Kunne ikke hente filtre");
      }
      const data = await res.json();
      setFilterOptions({
        cities: data.cities ?? [],
        districts: data.districts ?? [],
        segments: data.segments ?? [],
        roles: data.roles ?? [],
        chains: data.chains ?? [],
        sources: data.sources ?? [],
      });
    } catch (err) {
      console.error(err);
      setError(
        err instanceof Error
          ? err.message
          : "Ukjent feil ved henting av filtervalg",
      );
    }
  }, []);

  const fetchOverview = useCallback(async () => {
    setOverviewLoading(true);
    try {
      const res = await fetch(`${API_BASE}/overview`);
      if (!res.ok) {
        throw new Error("Kunne ikke hente overordnet dashboard");
      }
      const data: OverviewResponse = await res.json();
      setOverview(data);
    } catch (err) {
      console.error(err);
    } finally {
      setOverviewLoading(false);
    }
  }, []);

  const fetchRanking = useCallback(async (params: FilterState) => {
    setLoading(true);
    setError(null);
    try {
      const query = buildQuery(params);
      const url = `${API_BASE}/brokers?${query.toString()}`;
      const res = await fetch(url);
      if (!res.ok) {
        const message = await res.text();
        throw new Error(message || "Kunne ikke hente meglerdata");
      }
      const data: RankingResponse = await res.json();
      setRanking(data);
    } catch (err) {
      console.error(err);
      setError(
        err instanceof Error ? err.message : "Ukjent feil ved henting av data",
      );
      setRanking(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadFilters().catch((err) => {
      console.error(err);
    });
    fetchOverview().catch((err) => console.error(err));
  }, [loadFilters, fetchOverview]);

  useEffect(() => {
    if (!hasApplied) return;
    fetchRanking(appliedFilters).catch((err) => console.error(err));
  }, [appliedFilters, hasApplied, fetchRanking]);

  useEffect(() => {
    if (!selectedBrokerKey) {
      setBrokerDetail(null);
      setDetailError(null);
      return;
    }
    if (!hasApplied) return;
    setDetailLoading(true);
    setDetailError(null);
    setBrokerDetail(null);
    const query = buildQuery(appliedFilters);
    const url = `${API_BASE}/brokers/${encodeURIComponent(selectedBrokerKey)}?${query.toString()}`;
    fetch(url)
      .then(async (res) => {
        if (!res.ok) {
          const message = await res.text();
          throw new Error(message || "Kunne ikke hente meglerkort");
        }
        return res.json();
      })
      .then((data: BrokerDetail) => {
        setBrokerDetail(data);
      })
      .catch((err) => {
        console.error(err);
        setDetailError(err instanceof Error ? err.message : "Ukjent feil ved henting av meglerkort");
      })
      .finally(() => setDetailLoading(false));
  }, [selectedBrokerKey, appliedFilters, hasApplied]);

  useEffect(() => {
    if (!brokerDetail) return;
    setShowAllSegments(false);
    setShowAllLocations(false);
    setShowAllPeers(false);
    setShowAllRecommendations(false);
  }, [brokerDetail]);

  const applyFilters = () => {
    const next = cloneFilters(pendingFilters);
    setHasApplied(true);
    setSelectedBrokerKey(null);
    setBrokerDetail(null);
    setDetailError(null);
    setRanking(null);
    setAppliedFilters(next);
  };

  const closeBrokerDetail = () => {
    setSelectedBrokerKey(null);
    setDetailLoading(false);
    setDetailError(null);
  };

  const activeFilterChips = useMemo<FilterChip[]>(() => {
    const source = appliedFilters;
    const chips: FilterChip[] = [];
    if (source.city) chips.push({ key: "city", label: source.city });
    source.districts.forEach((district) =>
      chips.push({ key: "district", value: district, label: district }),
    );
    source.segments.forEach((segment) =>
      chips.push({ key: "segment", value: segment, label: segment }),
    );
    source.roles.forEach((role) =>
      chips.push({ key: "role", value: role, label: role }),
    );
    source.chains.forEach((chain) =>
      chips.push({ key: "chain", value: chain, label: chain }),
    );
    source.sources.forEach((src) =>
      chips.push({ key: "source", value: src, label: src }),
    );
    if (source.minSales > 0)
      chips.push({
        key: "minSales",
        label: `Min. salg ≥ ${source.minSales}`,
      });
    if (source.search)
      chips.push({
        key: "search",
        label: `Søk: ${source.search}`,
      });
    return chips;
  }, [appliedFilters]);

  const handleRemoveChip = (chip: FilterChip) => {
    setSelectedBrokerKey(null);
    setBrokerDetail(null);
    setDetailError(null);
    const current = cloneFilters(appliedFilters);
    switch (chip.key) {
      case "city":
        current.city = "";
        current.districts = [];
        setDistrictQuery("");
        break;
      case "district":
        current.districts = current.districts.filter((d) => d !== chip.value);
        break;
      case "segment":
        current.segments = current.segments.filter((s) => s !== chip.value);
        setSegmentQuery("");
        break;
      case "role":
        current.roles = current.roles.filter((r) => r !== chip.value);
        setRoleQuery("");
        break;
      case "chain":
        current.chains = current.chains.filter((c) => c !== chip.value);
        setChainQuery("");
        break;
      case "source":
        current.sources = current.sources.filter((c) => c !== chip.value);
        setSourceQuery("");
        break;
      case "minSales":
        current.minSales = 0;
        break;
      case "search":
        current.search = "";
        break;
      default:
        break;
    }
    setPendingFilters(current);
    setRanking(null);
    setAppliedFilters(cloneFilters(current));
  };

  const handleMultiSelectChange = (
    field: keyof FilterState,
    values: string[],
  ) => {
    if (values.includes("__all")) {
      setPendingFilters((prev) => ({
        ...prev,
        [field]: [],
      }));
      return;
    }
    setPendingFilters((prev) => ({
      ...prev,
      [field]: values,
    }));
  };

  const overviewCards = useMemo(() => {
    if (!overview) return [];
    return [
      {
        title: "Totalt antall annonser",
        value: overview.kpi.listings_now.toLocaleString("nb-NO"),
        delta: pctLabel(overview.kpi.listings_pct_change),
      },
      {
        title: "Samlet verdi",
        value: formatCompact(overview.kpi.total_value_now),
        delta: pctLabel(overview.kpi.total_value_pct_change),
      },
      {
        title: "Estimert provisjon",
        value: formatCompact(overview.kpi.commission_now),
        delta: pctLabel(overview.kpi.commission_pct_change),
      },
      {
        title: "Aktive meglere",
        value: overview.kpi.active_brokers_now.toLocaleString("nb-NO"),
        delta: pctLabel(overview.kpi.active_brokers_pct_change),
      },
    ];
  }, [overview]);

  const datasetItems = useMemo(() => {
    if (!overview?.datasets) return [];
    return [
      {
        key: "all",
        label: "all_listings.csv",
        hint: "Samlet datasett",
        value: overview.datasets.all_listings,
      },
      {
        key: "dnb",
        label: "dnb_listings.csv",
        hint: "Direkte fra DNB Eiendom",
        value: overview.datasets.dnb_listings,
      },
      {
        key: "hjem",
        label: "hjem_listings.csv",
        hint: "Aktive annonser fra Hjem.no",
        value: overview.datasets.hjem_listings,
      },
      {
        key: "agg",
        label: "agg_sum_per_broker.csv",
        hint: "Aggregert per megler",
        value: overview.datasets.agg_sum_per_broker,
      },
      {
        key: "profiles",
        label: "broker_profiles.csv",
        hint: "Berikede profilfelter",
        value: overview.datasets.broker_profiles,
      },
    ];
  }, [overview]);

  const timelineData = useMemo(() => {
    if (!overview || overview.timeline.length === 0) {
      return [];
    }
    const trimmed = overview.timeline.slice(-12);
    const maxListings = Math.max(...trimmed.map((point) => point.listings), 1);
    const maxValue = Math.max(...trimmed.map((point) => point.value), 1);
    return trimmed.map((point) => ({
      ...point,
      listingsHeight: Math.max(6, (point.listings / maxListings) * 100),
      valueHeight: Math.max(6, (point.value / maxValue) * 100),
    }));
  }, [overview]);

  const kpiCards = useMemo(() => {
    if (!ranking) return [];
    return [
      {
        title: "Aktive annonser (12 mnd)",
        value: ranking.kpi.listings_now.toLocaleString("nb-NO"),
        delta: pctLabel(ranking.kpi.listings_pct_change),
      },
      {
        title: "Samlet verdi",
        value: formatCompact(ranking.kpi.total_value_now),
        delta: pctLabel(ranking.kpi.total_value_pct_change),
      },
      {
        title: "Estimert provisjon",
        value: formatCompact(ranking.kpi.commission_now),
        delta: pctLabel(ranking.kpi.commission_pct_change),
      },
      {
        title: "Aktive meglere",
        value: ranking.kpi.active_brokers_now.toLocaleString("nb-NO"),
        delta: pctLabel(ranking.kpi.active_brokers_pct_change),
      },
    ];
  }, [ranking]);

  const derivedSummary = useMemo(() => {
    if (!ranking) return null;
    const highVolumeCount = ranking.items.filter((item) => item.high_volume).length;

    const countTop = (values: (string | null | undefined)[]) => {
      const counts = new Map<string, number>();
      values.forEach((value) => {
        if (!value || value === "(ukjent segment)" || value === "(ukjent bydel)") {
          return;
        }
        counts.set(value, (counts.get(value) ?? 0) + 1);
      });
      return Array.from(counts.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3);
    };

    const segmentTop = countTop(ranking.items.map((item) => item.dominant_segment));
    const locationTop = countTop(ranking.items.map((item) => item.primary_location));
    const chainTop = countTop(ranking.items.map((item) => item.chain));

    return {
      highVolumeCount,
      segmentTop,
      locationTop,
      chainTop,
    };
  }, [ranking]);

  const currentSort = appliedFilters.sortBy;

  return (
    <>
    <div className={styles.shell}>
      <aside className={styles.sidebar}>
        <div className={styles.sidebarHeader}>
          <h2>Filtrer</h2>
          <p>Finjuster utsnittet før du analyserer meglerne.</p>
        </div>

        <div className={styles.filterGroup}>
          <label htmlFor="city" className={styles.label}>
            By
          </label>
          <select
            id="city"
            className={styles.select}
            value={pendingFilters.city}
            onChange={(event) => {
              const value = event.target.value;
              setPendingFilters((prev) => ({
                ...prev,
                city: value,
                districts: [],
              }));
              setDistrictQuery("");
              loadFilters(value || undefined).catch((err) => console.error(err));
            }}
          >
            <option value="">(Alle)</option>
            {filterOptions.cities.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </div>

        <div className={styles.filterGroup}>
          <label htmlFor="districts" className={styles.label}>
            Bydeler
          </label>
          <input
            type="text"
            className={styles.searchInput}
            placeholder="Søk i bydeler"
            value={districtQuery}
            onChange={(event) => setDistrictQuery(event.target.value)}
          />
          <select
            id="districts"
            className={styles.select}
            multiple
            value={pendingFilters.districts}
            onChange={(event) =>
              handleMultiSelectChange(
                "districts",
                Array.from(event.target.selectedOptions).map((opt) => opt.value),
              )
            }
          >
            <option value="__all">Alle bydeler</option>
            {filterOptions.districts
              .filter((item) =>
                item && item.toLowerCase().includes(districtQuery.toLowerCase()),
              )
              .map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
          </select>
          <span className={styles.helperText}>
            Hold Ctrl (Windows) eller ⌘ (Mac) for å velge flere.
          </span>
          <button
            type="button"
            className={styles.inlineReset}
            onClick={() => handleMultiSelectChange("districts", [])}
          >
            Nullstill bydeler
          </button>
        </div>

        <div className={styles.filterGroup}>
          <label htmlFor="segments" className={styles.label}>
            Boligsegment
          </label>
          <input
            type="text"
            className={styles.searchInput}
            placeholder="Søk i segment"
            value={segmentQuery}
            onChange={(event) => setSegmentQuery(event.target.value)}
          />
          <select
            id="segments"
            className={styles.select}
            multiple
            value={pendingFilters.segments}
            onChange={(event) =>
              handleMultiSelectChange(
                "segments",
                Array.from(event.target.selectedOptions).map((opt) => opt.value),
              )
            }
          >
            <option value="__all">Alle segmenter</option>
            {filterOptions.segments
              .filter((item) =>
                item && item.toLowerCase().includes(segmentQuery.toLowerCase()),
              )
              .map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
          </select>
          <button
            type="button"
            className={styles.inlineReset}
            onClick={() => handleMultiSelectChange("segments", [])}
          >
            Nullstill segment
          </button>
        </div>

        <div className={styles.filterGroup}>
          <label htmlFor="roles" className={styles.label}>
            Meglerroller
          </label>
          <input
            type="text"
            className={styles.searchInput}
            placeholder="Søk i roller"
            value={roleQuery}
            onChange={(event) => setRoleQuery(event.target.value)}
          />
          <select
            id="roles"
            className={styles.select}
            multiple
            value={pendingFilters.roles}
            onChange={(event) =>
              handleMultiSelectChange(
                "roles",
                Array.from(event.target.selectedOptions).map((opt) => opt.value),
              )
            }
          >
            <option value="__all">Alle roller</option>
            {filterOptions.roles
              .filter((item) =>
                item && item.toLowerCase().includes(roleQuery.toLowerCase()),
              )
              .map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
          </select>
          <button
            type="button"
            className={styles.inlineReset}
            onClick={() => handleMultiSelectChange("roles", [])}
          >
            Nullstill roller
          </button>
        </div>

        <div className={styles.filterGroup}>
          <label htmlFor="chains" className={styles.label}>
            Kjede / kontor
          </label>
          <input
            type="text"
            className={styles.searchInput}
            placeholder="Søk i kjeder"
            value={chainQuery}
            onChange={(event) => setChainQuery(event.target.value)}
          />
          <select
            id="chains"
            className={styles.select}
            multiple
            value={pendingFilters.chains}
            onChange={(event) =>
              handleMultiSelectChange(
                "chains",
                Array.from(event.target.selectedOptions).map((opt) => opt.value),
              )
            }
          >
            <option value="__all">Alle kjeder</option>
            {filterOptions.chains
              .filter((item) =>
                item && item.toLowerCase().includes(chainQuery.toLowerCase()),
              )
              .map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
          </select>
          <button
            type="button"
            className={styles.inlineReset}
            onClick={() => handleMultiSelectChange("chains", [])}
          >
            Nullstill kjede
          </button>
        </div>

        <div className={styles.filterGroup}>
          <label htmlFor="chainKeyword" className={styles.label}>
            Kjedesøk
          </label>
          <input
            id="chainKeyword"
            className={styles.input}
            placeholder="F.eks. Nordvik"
            value={pendingFilters.chainKeyword}
            onChange={(event) =>
              setPendingFilters((prev) => ({
                ...prev,
                chainKeyword: event.target.value,
              }))
            }
          />
        </div>

        <div className={styles.filterGroup}>
          <label htmlFor="sources" className={styles.label}>
            Kilder
          </label>
          <input
            type="text"
            className={styles.searchInput}
            placeholder="Søk i kilder"
            value={sourceQuery}
            onChange={(event) => setSourceQuery(event.target.value)}
          />
          <select
            id="sources"
            className={styles.select}
            multiple
            value={pendingFilters.sources}
            onChange={(event) =>
              handleMultiSelectChange(
                "sources",
                Array.from(event.target.selectedOptions).map((opt) => opt.value),
              )
            }
          >
            <option value="__all">Alle kilder</option>
            {filterOptions.sources
              .filter((item) =>
                item && item.toLowerCase().includes(sourceQuery.toLowerCase()),
              )
              .map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
          </select>
          <button
            type="button"
            className={styles.inlineReset}
            onClick={() => handleMultiSelectChange("sources", [])}
          >
            Nullstill kilder
          </button>
        </div>

        <div className={styles.filterGroup}>
          <label htmlFor="search" className={styles.label}>
            Søk (megler/kontor)
          </label>
          <input
            id="search"
            className={styles.input}
            placeholder="Søk…"
            value={pendingFilters.search}
            onChange={(event) =>
              setPendingFilters((prev) => ({
                ...prev,
                search: event.target.value,
              }))
            }
          />
        </div>

        <div className={styles.filterGroup}>
          <label htmlFor="period" className={styles.label}>
            Tidsperiode
          </label>
          <select
            id="period"
            className={styles.select}
            value={pendingFilters.period}
            onChange={(event) =>
              setPendingFilters((prev) => ({
                ...prev,
                period: event.target.value,
              }))
            }
          >
            {periodOptions.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </div>

        <div className={styles.filterGroup}>
          <label htmlFor="minSales" className={styles.label}>
            Min. antall salg
          </label>
          <div className={styles.sliderRow}>
            <input
              id="minSales"
              type="range"
              min={0}
              max={30}
              value={pendingFilters.minSales}
              onChange={(event) =>
                setPendingFilters((prev) => ({
                  ...prev,
                  minSales: Number(event.target.value),
                }))
              }
            />
            <span className={styles.sliderValue}>
              {pendingFilters.minSales} salg
            </span>
          </div>
        </div>
        <button className={styles.applyButton} onClick={applyFilters}>
          Oppdater visning
        </button>

        <label className={styles.toggleRow}>
          <input
            type="checkbox"
            checked={showOverview}
            onChange={(event) => setShowOverview(event.target.checked)}
          />
          <span>Vis overordnet dashboard</span>
        </label>
      </aside>

      <main className={styles.main}>
        <header className={styles.header}>
          <div>
            <span className={styles.eyebrow}>Fase 2 · MeglerMonitor</span>
            <h1>Megleroversikt</h1>
            <p>
              Se hvilke meglere som dominerer utvalget ditt – med segmenter,
              lokasjoner og utvikling forrige kvartal.
            </p>
          </div>
          <div className={styles.meta}>
            <span className={styles.metaLabel}>Oppdatert</span>
            <strong>
              {ranking
                ? new Date(ranking.generated_at).toLocaleString("nb-NO", {
                    dateStyle: "short",
                    timeStyle: "short",
                  })
                : "–"}
            </strong>
            <span className={styles.metaLabel}>Meglere i utvalget</span>
            <strong>
              {ranking ? ranking.total_brokers.toLocaleString("nb-NO") : "–"}
            </strong>
          </div>
        </header>

        <section className={styles.chipRow}>
          <span className={styles.chipLabel}>Aktive filtre</span>
          <div className={styles.chipGroup}>
            {activeFilterChips.length === 0 ? (
              <span className={styles.noChips}>Ingen aktive filtre – viser hele porteføljen.</span>
            ) : (
              activeFilterChips.map((chip, idx) => (
                <button
                  key={`${chip.key}-${chip.label}-${idx}`}
                  type="button"
                  className={styles.filterChip}
                  onClick={() => handleRemoveChip(chip)}
                >
                  {chip.label}
                  <span aria-hidden>×</span>
                </button>
              ))
            )}
          </div>
          {activeFilterChips.length > 0 && (
            <button
              type="button"
              className={styles.clearAll}
              onClick={() => {
                setPendingFilters(cloneFilters(DEFAULT_FILTERS));
                setAppliedFilters(cloneFilters(DEFAULT_FILTERS));
                setRanking(null);
                setDistrictQuery("");
                setSegmentQuery("");
                setRoleQuery("");
                setChainQuery("");
                setSourceQuery("");
              }}
            >
              Fjern alle
            </button>
          )}
        </section>

        {showOverview && (
          <section className={styles.overviewSection}>
            <div className={styles.sectionHeader}>
              <div>
                <h2>Overordnet statistikk</h2>
                <p>Snapshot for hele markedsutvalget. Juster filtrene for å bore ned.</p>
              </div>
              <div className={styles.sectionMeta}>
                <span>Sist oppdatert</span>
                <strong>
                  {overview
                    ? new Date(overview.generated_at).toLocaleString("nb-NO", {
                        dateStyle: "short",
                        timeStyle: "short",
                      })
                    : "–"}
                </strong>
              </div>
            </div>

            {overviewLoading ? (
              <div className={styles.loadingPanel}>
                <div className={styles.loader} />
                <p>Laster overordnet statistikk…</p>
              </div>
            ) : overview ? (
              <>
                <div className={styles.overviewKpis}>
                  {overviewCards.map((card) => (
                    <article key={card.title} className={styles.overviewKpi}>
                      <span className={styles.overviewKpiLabel}>{card.title}</span>
                      <strong className={styles.overviewKpiValue}>{card.value}</strong>
                      <span className={styles.overviewKpiDelta}>{card.delta}</span>
                    </article>
                  ))}
                </div>
                <div className={styles.overviewContent}>
                  <div className={styles.overviewTimeline}>
                    <div className={styles.timelineHeader}>
                      <h3>Tidslinje</h3>
                      <span>Antall annonser og verdi per måned</span>
                    </div>
                    {timelineData.length === 0 ? (
                      <p className={styles.overviewEmpty}>Ingen tidsserie tilgjengelig.</p>
                    ) : (
                      <>
                        <div className={styles.timelineLegend}>
                          <span>
                            <i className={styles.legendDotListings} /> Annonser
                          </span>
                          <span>
                            <i className={styles.legendDotValue} /> Samlet verdi
                          </span>
                        </div>
                        <div className={styles.timelineBars}>
                          {timelineData.map((point) => (
                            <div key={point.month} className={styles.timelineColumn}>
                              <div className={styles.timelineBarGroup}>
                                <span
                                  className={`${styles.timelineBar} ${styles.timelineListings}`}
                                  style={{ height: `${point.listingsHeight}%` }}
                                  title={`${point.listings.toLocaleString("nb-NO")} annonser`}
                                />
                                <span
                                  className={`${styles.timelineBar} ${styles.timelineValue}`}
                                  style={{ height: `${point.valueHeight}%` }}
                                  title={`${formatCompact(point.value)} verdi`}
                                />
                              </div>
                              <div className={styles.timelineMetrics}>
                                <span>{point.listings.toLocaleString("nb-NO")}</span>
                                <span>{formatCompact(point.value)}</span>
                              </div>
                              <span className={styles.timelineLabel}>{point.month}</span>
                            </div>
                          ))}
                        </div>
                      </>
                    )}
                  </div>
                  <div className={styles.overviewSide}>
                    <div className={styles.overviewMultiCard}>
                      <div className={styles.quickSearchRow}>
                        <input
                          type="text"
                          className={styles.quickSearch}
                          placeholder="Søk etter megler eller kontor…"
                          onChange={(event) => {
                            setBrokerSearch(event.target.value);
                          }}
                          onKeyDown={(event) => {
                            if (event.key === "Enter") {
                              handleQuickSearch();
                            }
                          }}
                          value={brokerSearch}
                        />
                        <button
                          type="button"
                          className={styles.quickSearchButton}
                          onClick={handleQuickSearch}
                        >
                          Søk
                        </button>
                      </div>
                      <div className={styles.overviewListsCompact}>
                        <div className={styles.overviewList}>
                          <h3>Segmenter</h3>
                          <ul>
                            {overview.segments.map((item) => (
                              <li key={item.label}>
                                <div className={styles.listRow}>
                                  <span>{item.label}</span>
                                  <span>
                                    {item.count.toLocaleString("nb-NO")} · {formatPercentValue(item.share)}
                                  </span>
                                </div>
                                <div className={styles.progressTrack}>
                                  <div
                                    className={styles.progressFill}
                                    style={{ width: `${Math.min(100, item.share * 100)}%` }}
                                  />
                                </div>
                              </li>
                            ))}
                          </ul>
                        </div>
                        <div className={styles.overviewList}>
                          <h3>Bydeler</h3>
                        <ul>
                          {overview.locations.map((item) => (
                            <li key={item.label}>
                              <div className={styles.listRow}>
                                <span>{item.label}</span>
                                <span>
                                  {item.count.toLocaleString("nb-NO")} · {formatPercentValue(item.share)}
                                </span>
                              </div>
                              <div className={styles.progressTrack}>
                                <div
                                  className={styles.progressFill}
                                  style={{ width: `${Math.min(100, item.share * 100)}%` }}
                                />
                              </div>
                            </li>
                          ))}
                        </ul>
                        </div>
                        <div className={styles.overviewList}>
                          <h3>Kjeder</h3>
                          <ul>
                            {overview.chains.map((item) => (
                              <li key={item.label}>
                                <div className={styles.listRow}>
                                  <span>{item.label}</span>
                                  <span>
                                    {item.count.toLocaleString("nb-NO")} · {formatPercentValue(item.share)}
                                  </span>
                                </div>
                                <div className={styles.progressTrack}>
                                  <div
                                    className={styles.progressFill}
                                    style={{ width: `${Math.min(100, item.share * 100)}%` }}
                                  />
                                </div>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                      {datasetItems.length > 0 && (
                        <div className={styles.datasetStatus}>
                          {datasetItems.map((item) => (
                            <article key={item.key} className={styles.datasetCard}>
                              <span className={styles.datasetLabel}>{item.label}</span>
                              <strong className={styles.datasetValue}>
                                {item.value.toLocaleString("nb-NO")}
                              </strong>
                              <span className={styles.datasetHint}>{item.hint}</span>
                            </article>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className={styles.overviewEmpty}>Ingen data tilgjengelig for oversikten.</div>
            )}
          </section>
        )}

        {ranking &&
          (topCommissionBrokers.length > 0 || topCommissionOffices.length > 0) && (
            <section className={styles.commissionSection}>
              <div className={styles.commissionToolbar}>
                <label className={styles.listControl} htmlFor="commissionSearch">
                  <span>Søk</span>
                  <input
                    id="commissionSearch"
                    type="text"
                    className={styles.searchInput}
                    placeholder="Søk etter megler eller kontor"
                    value={commissionSearch}
                    onChange={(event) => setCommissionSearch(event.target.value)}
                  />
                </label>
              </div>

              <div className={styles.commissionGrid}>
                {topCommissionBrokers.length > 0 && (
                  <article className={styles.commissionCard}>
                    <header className={styles.commissionHeader}>
                      <h3>Størst provisjonsgrunnlag – meglere</h3>
                      <p>
                        Toppmeglere sortert på estimert provisjon (1,25 %) for aktive boliger.
                      </p>
                    </header>
                    <div className={styles.listControls}>
                      <label className={styles.listControl}>
                        <span>Sorter etter</span>
                        <select
                          value={topBrokerSort}
                          onChange={(event) =>
                            setTopBrokerSort(event.target.value as BrokerSortKey)
                          }
                        >
                          {topBrokerSortOptions.map((option) => (
                            <option key={option.value} value={option.value}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className={styles.listControl}>
                        <span>Vis topp</span>
                        <select
                          value={String(topBrokerLimit)}
                          onChange={(event) =>
                            setTopBrokerLimit(
                              event.target.value === "all"
                                ? "all"
                                : Number(event.target.value),
                            )
                          }
                        >
                          {topLimitOptions.map((option) => (
                            <option key={option} value={option === "all" ? "all" : option}>
                              {option === "all" ? "Alle" : `Topp ${option}`}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                    <div className={styles.commissionListScroll}>
                      <div className={styles.commissionList}>
                        {displayedTopCommissionBrokers.map((row, index) => {
                          const brokerKey = resolveBrokerKey(row.broker, row.chain);
                          const disabled = !brokerKey;
                          return (
                            <button
                              type="button"
                              key={`${row.broker}-${row.chain}-${index}`}
                              className={`${styles.commissionItem} ${disabled ? styles.commissionItemDisabled : ""}`}
                              onClick={() => brokerKey && openBrokerDetail(brokerKey)}
                              disabled={disabled}
                            >
                              <div className={styles.commissionPrimary}>
                                <span className={styles.commissionRank}>#{index + 1}</span>
                                <div className={styles.commissionName}>
                                  <strong>{row.broker ?? "–"}</strong>
                                  <span>{row.chain ?? "–"}</span>
                                </div>
                              </div>
                              <div className={styles.commissionStats}>
                                <span>
                                  <strong>{row.listing_count.toLocaleString("nb-NO")}</strong>
                                  <small>boliger</small>
                                </span>
                                <span>
                                  <strong>{formatCompactWithKr(row.total_value)}</strong>
                                  <small>verdi</small>
                                </span>
                                <span>
                                  <strong>{formatCompactWithKr(row.commission)}</strong>
                                  <small>provisjon</small>
                                </span>
                                <span>
                                  <strong>{formatCompactWithKr(row.commission_avg)}</strong>
                                  <small>snitt</small>
                                </span>
                              </div>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  </article>
                )}

                {topCommissionOffices.length > 0 && (
                  <article className={styles.commissionCard}>
                    <header className={styles.commissionHeader}>
                      <h3>Størst provisjonsgrunnlag – kontorer</h3>
                      <p>Kontorer rangert etter estimert provisjon basert på aktive boliger.</p>
                    </header>
                    <div className={styles.listControls}>
                      <label className={styles.listControl}>
                        <span>Sorter etter</span>
                        <select
                          value={topOfficeSort}
                          onChange={(event) =>
                            setTopOfficeSort(event.target.value as OfficeSortKey)
                          }
                        >
                          {topOfficeSortOptions.map((option) => (
                            <option key={option.value} value={option.value}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className={styles.listControl}>
                        <span>Vis topp</span>
                        <select
                          value={String(topOfficeLimit)}
                          onChange={(event) =>
                            setTopOfficeLimit(
                              event.target.value === "all"
                                ? "all"
                                : Number(event.target.value),
                            )
                          }
                        >
                          {topLimitOptions.map((option) => (
                            <option key={option} value={option === "all" ? "all" : option}>
                              {option === "all" ? "Alle" : `Topp ${option}`}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                    <div className={styles.commissionListScroll}>
                      <div className={styles.commissionList}>
                        {displayedTopCommissionOffices.map((row, index) => (
                          <div
                            key={`${row.office}-${index}`}
                            className={`${styles.commissionItem} ${styles.commissionItemStatic}`}
                          >
                            <div className={styles.commissionPrimary}>
                              <span className={styles.commissionRank}>#{index + 1}</span>
                              <div className={styles.commissionName}>
                                <strong>{row.office ?? "–"}</strong>
                                <span>{row.chain ?? "–"}</span>
                              </div>
                            </div>
                            <div className={styles.commissionStats}>
                              <span>
                                <strong>{row.listing_count.toLocaleString("nb-NO")}</strong>
                                <small>boliger</small>
                              </span>
                              <span>
                                <strong>{formatCompactWithKr(row.total_value)}</strong>
                                <small>verdi</small>
                              </span>
                              <span>
                                <strong>{formatCompactWithKr(row.commission)}</strong>
                                <small>provisjon</small>
                              </span>
                              <span>
                                <strong>{row.chain_broker_count.toLocaleString("nb-NO")}</strong>
                                <small>megler(e)</small>
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </article>
                )}
              </div>
            </section>
          )}

        {ranking &&
          (portfolioGainers.length > 0 || portfolioLosers.length > 0) && (
            <section className={styles.portfolioGrid}>
              <article className={styles.portfolioCard}>
                <header className={styles.commissionHeader}>
                  <h3>Meglere med flere aktive boliger</h3>
                  <p>
                    Endring i porteføljeverdi siste 30 dager sammenlignet med de foregående 30
                    dagene.
                  </p>
                </header>
                {portfolioGainers.length === 0 ? (
                  <p className={styles.commissionEmpty}>
                    Ingen meglere med positiv utvikling i perioden.
                  </p>
                ) : (
                  <ul className={styles.portfolioList}>
                    {portfolioGainers.map((row, index) => {
                      const deltaPositive = row.delta_value >= 0;
                      const deltaColor = deltaPositive ? "#79e6c8" : "#ff9c9c";
                      return (
                        <li key={`${row.broker}-${row.chain}-gain-${index}`}>
                          <div className={styles.portfolioRank}>#{index + 1}</div>
                          <div className={styles.portfolioMain}>
                            <strong>{row.broker ?? "–"}</strong>
                            <span>{row.chain ?? "–"}</span>
                          </div>
                          <div className={styles.portfolioDelta}>
                            <span style={{ color: deltaColor }}>
                              {formatSignedCurrency(row.delta_value)}
                            </span>
                            <small style={{ color: deltaColor }}>
                              {formatSignedPercent(row.delta_pct)}
                            </small>
                          </div>
                          <div className={styles.portfolioNow}>
                            <span>Aktiv portefølje nå</span>
                            <strong>{formatCurrency(row.value_now)}</strong>
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </article>

              <article className={styles.portfolioCard}>
                <header className={styles.commissionHeader}>
                  <h3>Meglere med færre aktive boliger</h3>
                  <p>Størst nedgang i porteføljeverdi i samme 30 dagers-vindu.</p>
                </header>
                {portfolioLosers.length === 0 ? (
                  <p className={styles.commissionEmpty}>
                    Ingen meglere med nedgang i perioden.
                  </p>
                ) : (
                  <ul className={styles.portfolioList}>
                    {portfolioLosers.map((row, index) => {
                      const deltaPositive = row.delta_value >= 0;
                      const deltaColor = deltaPositive ? "#79e6c8" : "#ff9c9c";
                      return (
                        <li key={`${row.broker}-${row.chain}-loss-${index}`}>
                          <div className={styles.portfolioRank}>#{index + 1}</div>
                          <div className={styles.portfolioMain}>
                            <strong>{row.broker ?? "–"}</strong>
                            <span>{row.chain ?? "–"}</span>
                          </div>
                          <div className={styles.portfolioDelta}>
                            <span style={{ color: deltaColor }}>
                              {formatSignedCurrency(row.delta_value)}
                            </span>
                            <small style={{ color: deltaColor }}>
                              {formatSignedPercent(row.delta_pct)}
                            </small>
                          </div>
                          <div className={styles.portfolioNow}>
                            <span>Aktiv portefølje nå</span>
                            <strong>{formatCurrency(row.value_now)}</strong>
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </article>
            </section>
          )}
      </main>
    </div>

      {selectedBrokerKey && (
        <div className={styles.detailOverlay} onClick={closeBrokerDetail}>
          <div
            className={styles.detailPanel}
            onClick={(event) => event.stopPropagation()}
          >
            <button className={styles.detailClose} onClick={closeBrokerDetail}>
              ×
            </button>
            {detailLoading ? (
              <div className={styles.detailLoading}>
                <div className={styles.loader} />
                <p>Laster meglerkort…</p>
              </div>
            ) : detailError ? (
              <div className={styles.detailError}>{detailError}</div>
            ) : brokerDetail ? (
              <div className={styles.detailContent}>
                <header className={styles.detailHeader}>
                  <div>
                    <h2>{brokerDetail.broker}</h2>
                    <p>
                      {brokerDetail.chain}
                      {brokerDetail.broker_role ? ` · ${brokerDetail.broker_role}` : ""}
                    </p>
                    <p className={styles.detailMeta}>
                      {brokerDetail.city || "(ukjent by)"} · {brokerDetail.primary_location || "(ukjent bydel)"}
                    </p>
                  </div>
                  {brokerDetail.high_volume && (
                    <span className={styles.detailChip}>Høyvolum</span>
                  )}
                </header>

                {(brokerDetail.dominant_segment ||
                  brokerDetail.segment_summary ||
                  brokerDetail.location_summary) && (
                  <section className={styles.detailSummary}>
                    {brokerDetail.dominant_segment && (
                      <div className={styles.detailSummaryCard}>
                        <span>Fokussegment</span>
                        <strong>{brokerDetail.dominant_segment}</strong>
                      </div>
                    )}
                    {brokerDetail.segment_summary && (
                      <div className={styles.detailSummaryCard}>
                        <span>Segmentinnsikt</span>
                        <p>{brokerDetail.segment_summary}</p>
                      </div>
                    )}
                    {brokerDetail.location_summary && (
                      <div className={styles.detailSummaryCard}>
                        <span>Områdeinnsikt</span>
                        <p>{brokerDetail.location_summary}</p>
                      </div>
                    )}
                  </section>
                )}

                <section className={styles.detailKpis}>
                  <article>
                    <span>Totalt salg</span>
                    <strong>{brokerDetail.metrics.total_sales.toLocaleString("nb-NO")}</strong>
                  </article>
                  <article>
                    <span>Samlet verdi</span>
                    <strong>{formatCompact(brokerDetail.metrics.total_value)}</strong>
                  </article>
                  <article>
                    <span>Snittpris</span>
                    <strong>{formatCurrency(brokerDetail.metrics.avg_price)}</strong>
                  </article>
                  <article>
                    <span>Estimert provisjon</span>
                    <strong>{formatCompact(brokerDetail.metrics.commission_estimate)}</strong>
                  </article>
                </section>

                <section className={styles.detailGrid}>
                  <div>
                    <h3>Segmenter</h3>
                    {brokerSegments.length === 0 ? (
                      <p className={styles.detailEmpty}>Ingen segmentdata.</p>
                    ) : (
                      <ul>
                        {displayedSegments.map((item) => (
                          <li key={item.segment}>
                            <span>{item.segment}</span>
                            <strong>{item.count}</strong>
                          </li>
                        ))}
                      </ul>
                    )}
                    {segmentsToggleable && (
                      <button
                        type="button"
                        className={styles.detailListToggle}
                        onClick={() => setShowAllSegments((prev) => !prev)}
                      >
                        {showAllSegments ? "Vis færre" : `Vis alle (${brokerSegments.length})`}
                      </button>
                    )}
                  </div>
                  <div>
                    <h3>Lokasjoner</h3>
                    {brokerLocations.length === 0 ? (
                      <p className={styles.detailEmpty}>Ingen bydeler registrert.</p>
                    ) : (
                      <ul>
                        {displayedLocations.map((item) => (
                          <li key={item.district}>
                            <span>{item.district}</span>
                            <strong>{item.count}</strong>
                          </li>
                        ))}
                      </ul>
                    )}
                    {locationsToggleable && (
                      <button
                        type="button"
                        className={styles.detailListToggle}
                        onClick={() => setShowAllLocations((prev) => !prev)}
                      >
                        {showAllLocations ? "Vis færre" : `Vis alle (${brokerLocations.length})`}
                      </button>
                    )}
                  </div>
                  <div>
                    <h3>Prisnivå</h3>
                    <ul>
                      <li>
                        <span>Min</span>
                        <strong>{formatCurrency(brokerDetail.price_stats.min ?? null)}</strong>
                      </li>
                      <li>
                        <span>Median</span>
                        <strong>{formatCurrency(brokerDetail.price_stats.median ?? null)}</strong>
                      </li>
                      <li>
                        <span>Snitt</span>
                        <strong>{formatCurrency(brokerDetail.price_stats.mean ?? null)}</strong>
                      </li>
                      <li>
                        <span>Maks</span>
                        <strong>{formatCurrency(brokerDetail.price_stats.max ?? null)}</strong>
                      </li>
                    </ul>
                  </div>
                  <div>
                    <h3>Utvikling siste 90 dager</h3>
                    <p>
                      Nå: {brokerDetail.metrics.recent_sales.current} · Tidligere: {brokerDetail.metrics.recent_sales.previous}
                    </p>
                    {brokerDetail.timeline.length === 0 ? (
                      <p className={styles.detailEmpty}>Ingen tidsserie tilgjengelig.</p>
                    ) : (
                      <ul className={styles.timelineList}>
                        {brokerDetail.timeline.slice(-6).map((point) => (
                          <li key={point.month}>
                            <span>{point.month}</span>
                            <span>{point.listings.toLocaleString("nb-NO")}</span>
                            <span>{formatCompact(point.value)}</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                </section>

                <section className={styles.detailSplit}>
                  <div>
                    <h3>Peers</h3>
                    {brokerPeers.length === 0 ? (
                      <p className={styles.detailEmpty}>Ingen sammenlignbare meglere funnet.</p>
                    ) : (
                      <ul>
                        {displayedPeers.map((peer) => (
                          <li key={peer.broker_key}>
                            <div>
                              <strong>{peer.broker}</strong>
                              <span>{peer.chain}</span>
                            </div>
                            <span>{formatCompact(peer.total_value)}</span>
                          </li>
                        ))}
                      </ul>
                    )}
                    {peersToggleable && (
                      <button
                        type="button"
                        className={styles.detailListToggle}
                        onClick={() => setShowAllPeers((prev) => !prev)}
                      >
                        {showAllPeers ? "Vis færre" : "Vis flere"}
                      </button>
                    )}
                  </div>
                  <div>
                    <h3>Du vil også like</h3>
                    {brokerRecommendations.length === 0 ? (
                      <p className={styles.detailEmpty}>Ingen anbefalinger akkurat nå.</p>
                    ) : (
                      <ul>
                        {displayedRecommendations.map((rec) => (
                          <li key={rec.broker_key}>
                            <div>
                              <strong>{rec.broker}</strong>
                              <span>{rec.chain}</span>
                            </div>
                            <span>{rec.primary_location}</span>
                          </li>
                        ))}
                      </ul>
                    )}
                    {recommendationsToggleable && (
                      <button
                        type="button"
                        className={styles.detailListToggle}
                        onClick={() =>
                          setShowAllRecommendations((prev) => !prev)
                        }
                      >
                        {showAllRecommendations ? "Vis færre" : "Vis flere"}
                      </button>
                    )}
                  </div>
                </section>

                {brokerDetail.profile && (
                  <section className={styles.detailProfile}>
                    <h3>Profilinformasjon</h3>
                    <ul>
                      {brokerDetail.profile.linkedin_url && (
                        <li>
                          <span>LinkedIn</span>
                          <a href={brokerDetail.profile.linkedin_url} target="_blank" rel="noreferrer">
                            Åpne profil
                          </a>
                        </li>
                      )}
                      {brokerDetail.profile.experience_years !== null && (
                        <li>
                          <span>Erfaring</span>
                          <strong>{Math.round(brokerDetail.profile.experience_years)} år</strong>
                        </li>
                      )}
                      {brokerDetail.profile.age !== null && (
                        <li>
                          <span>Alder</span>
                          <strong>{brokerDetail.profile.age} år</strong>
                        </li>
                      )}
                    </ul>
                  </section>
                )}
              </div>
            ) : (
              <div className={styles.detailError}>Ingen detaljer tilgjengelig.</div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
