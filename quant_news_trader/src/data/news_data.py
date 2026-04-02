from typing import Optional, Iterable
import logging
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests
import pandas as pd


logger = logging.getLogger(__name__)

_DEFAULT_MACRO_NEWS_QUERIES = [
    "Federal Reserve",
    "US inflation CPI",
    "US nonfarm payrolls",
    "Treasury yields",
    "S&P 500 market",
    "NASDAQ market",
    "VIX volatility",
]


def _fetch_google_news_rss(
    query: str,
    lookback_days: int = 7,
    max_articles: int = 200,
) -> pd.DataFrame:
    """Fetch news from Google News RSS without requiring an API key."""
    rss_url = (
        "https://news.google.com/rss/search"
        f"?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    )

    try:
        resp = requests.get(rss_url, timeout=20)
        if not resp.ok:
            logger.warning("Google News RSS request failed: status=%s", resp.status_code)
            return pd.DataFrame(columns=["published_at", "title", "description", "content"])

        root = ET.fromstring(resp.content)
        items = root.findall("./channel/item")

        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)
        rows = []
        for item in items:
            pub_date = item.findtext("pubDate")
            published_at = pd.to_datetime(pub_date, errors="coerce", utc=True)
            if pd.isna(published_at) or published_at < cutoff:
                continue

            title = item.findtext("title")
            description = item.findtext("description")
            link = item.findtext("link")
            source = item.findtext("source")

            rows.append(
                {
                    "published_at": published_at,
                    "title": title,
                    "description": description,
                    "content": description,
                    "source": source,
                    "url": link,
                    "provider": "google_rss",
                }
            )

            if len(rows) >= max_articles:
                break

        df = pd.DataFrame(rows)
        if not df.empty:
            df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
            df = df.dropna(subset=["published_at"]).sort_values("published_at", ascending=False)
        return df
    except Exception as e:
        logger.warning("Google News RSS parse failed: %s", e)
        return pd.DataFrame(columns=["published_at", "title", "description", "content"])


def fetch_news_free(
    query: str,
    language: str = "en",
    lookback_days: int = 7,
    max_articles: int = 200,
) -> pd.DataFrame:
    """Fetch news from free RSS sources only."""
    logger.info("Using Google News RSS for query=%s", query)
    return _fetch_google_news_rss(
        query=query,
        lookback_days=lookback_days,
        max_articles=max_articles,
    )


def _normalize_ticker_query(ticker: str) -> str:
    t = str(ticker or "").strip().upper()
    if not t:
        return ""
    return f"{t} stock"


def _dedupe_ordered_strings(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        v = str(raw or "").strip()
        if not v:
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def build_news_query_list(
    base_query: str | None = None,
    tickers: Optional[list[str]] = None,
    include_macro_news: bool = True,
    macro_queries: Optional[list[str]] = None,
) -> list[str]:
    """Build an ordered query list: tickers first, then optional base and macro terms."""
    ticker_queries = [_normalize_ticker_query(t) for t in (tickers or [])]
    ticker_queries = [q for q in ticker_queries if q]

    queries: list[str] = []
    queries.extend(ticker_queries)

    if base_query and str(base_query).strip():
        queries.append(str(base_query).strip())

    if include_macro_news:
        queries.extend(macro_queries or _DEFAULT_MACRO_NEWS_QUERIES)

    return _dedupe_ordered_strings(queries)


def fetch_news_for_watchlist(
    base_query: str | None = None,
    tickers: Optional[list[str]] = None,
    language: str = "en",
    lookback_days: int = 7,
    max_articles: int = 200,
    include_macro_news: bool = True,
    macro_queries: Optional[list[str]] = None,
    per_query_article_cap: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch and merge stock-specific + macro news into one DataFrame.

    - Queries include each ticker as "{TICKER} stock"
    - Optionally includes base_query and macro queries
    - Deduplicates articles by URL then title/content fingerprints
    """
    queries = build_news_query_list(
        base_query=base_query,
        tickers=tickers,
        include_macro_news=include_macro_news,
        macro_queries=macro_queries,
    )
    if not queries:
        return pd.DataFrame(columns=["published_at", "title", "description", "content"])

    per_query_cap = int(per_query_article_cap or max(10, max_articles // max(1, len(queries))))
    parts: list[pd.DataFrame] = []

    for q in queries:
        try:
            df_q = fetch_news_free(
                query=q,
                language=language,
                lookback_days=lookback_days,
                max_articles=per_query_cap,
            )
            if df_q is None or df_q.empty:
                continue

            df_q = df_q.copy()
            df_q["query"] = q
            df_q["news_scope"] = "macro" if q in (macro_queries or _DEFAULT_MACRO_NEWS_QUERIES) else "ticker"
            parts.append(df_q)
        except Exception as e:
            logger.warning("Aggregated news fetch failed for query=%s: %s", q, e)

    if not parts:
        return pd.DataFrame(columns=["published_at", "title", "description", "content"])

    merged = pd.concat(parts, ignore_index=True)
    if "published_at" in merged.columns:
        merged["published_at"] = pd.to_datetime(merged["published_at"], errors="coerce", utc=True)

    if "url" in merged.columns:
        merged = merged.drop_duplicates(subset=["url"], keep="first")

    dedupe_cols = [c for c in ["published_at", "title", "description", "content"] if c in merged.columns]
    if dedupe_cols:
        merged = merged.drop_duplicates(subset=dedupe_cols, keep="first")

    if "published_at" in merged.columns:
        merged = merged.dropna(subset=["published_at"]).sort_values("published_at", ascending=False)

    return merged.head(max_articles).reset_index(drop=True)
