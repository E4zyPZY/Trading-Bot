from datetime import datetime, timedelta
from typing import Optional
import os
import logging
from pathlib import Path
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests
import pandas as pd


logger = logging.getLogger(__name__)
_ENV_LOADED = False


def _load_local_env_vars() -> None:
    """Load .env values into process env without external dependencies."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    env_paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]

    for env_path in env_paths:
        if not env_path.exists():
            continue

        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except Exception as e:
            logger.warning("Failed loading .env from %s: %s", env_path, e)

    _ENV_LOADED = True


def _fetch_news_newsapi(
    query: str,
    api_key: str,
    language: str = "en",
    lookback_days: int = 7,
    max_articles: int = 200,
) -> pd.DataFrame:
    """Fetch news from NewsAPI.org (requires API key)."""
    end = pd.Timestamp.now("UTC")
    start = end - pd.Timedelta(days=lookback_days)

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": language,
        "from": start.isoformat(),
        "to": end.isoformat(),
        "sortBy": "publishedAt",
        "pageSize": min(max_articles, 100),
        "apiKey": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=20)
        if not resp.ok:
            logger.warning(
                "NewsAPI request failed: status=%s query=%s response=%s",
                resp.status_code,
                query,
                resp.text[:200],
            )
            return pd.DataFrame(columns=["published_at", "title", "description", "content"])

        data = resp.json()
        articles = data.get("articles", [])

        rows = []
        for a in articles:
            rows.append(
                {
                    "published_at": a.get("publishedAt"),
                    "title": a.get("title"),
                    "description": a.get("description"),
                    "content": a.get("content") or a.get("description"),
                    "source": (a.get("source") or {}).get("name"),
                    "url": a.get("url"),
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
            df = df.dropna(subset=["published_at"]).sort_values("published_at", ascending=False)
        return df
    except Exception as e:
        logger.warning("NewsAPI fetch failed: %s", e)
        return pd.DataFrame(columns=["published_at", "title", "description", "content"])


def _fetch_news_gnews(
    query: str,
    api_key: str,
    language: str = "en",
    lookback_days: int = 7,
    max_articles: int = 200,
) -> pd.DataFrame:
    """Fetch news from GNews (requires API key)."""
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)

    url = "https://gnews.io/api/v4/search"
    params = {
        "q": query,
        "lang": language,
        "from": start.isoformat(timespec="seconds") + "Z",
        "to": end.isoformat(timespec="seconds") + "Z",
        "max": max_articles,
        "expand": "content",
        "apikey": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=20)
        if not resp.ok:
            logger.warning(
                "GNews request failed: status=%s query=%s response=%s",
                resp.status_code,
                query,
                resp.text[:200],
            )
            return pd.DataFrame(columns=["published_at", "title", "description", "content"])

        data = resp.json()
        articles = data.get("articles", [])

        rows = []
        for a in articles:
            rows.append(
                {
                    "published_at": a.get("publishedAt"),
                    "title": a.get("title"),
                    "description": a.get("description"),
                    "content": a.get("content"),
                    "source": a.get("source", {}).get("name"),
                    "url": a.get("url"),
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
            df = df.dropna(subset=["published_at"]).sort_values("published_at", ascending=False)
        return df
    except Exception as e:
        logger.warning("GNews fetch failed: %s", e)
        return pd.DataFrame(columns=["published_at", "title", "description", "content"])


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
    """
    Example free-ish news fetcher using GNews API (no AI, just headlines).
    This is a placeholder; feel free to swap to any news endpoint you like.
    """
    _load_local_env_vars()

    # Priority 1: NewsAPI (if key provided)
    newsapi_key = os.getenv("NEWSAPI_KEY") or os.getenv("NEWS_API_KEY")
    if newsapi_key:
        df = _fetch_news_newsapi(
            query=query,
            api_key=newsapi_key,
            language=language,
            lookback_days=lookback_days,
            max_articles=max_articles,
        )
        if not df.empty:
            return df
        logger.warning("NewsAPI returned no rows for query=%s; trying fallbacks.", query)

    # Priority 2: GNews (if key provided)
    gnews_key = os.getenv("GNEWS_API_KEY")
    if gnews_key:
        df = _fetch_news_gnews(
            query=query,
            api_key=gnews_key,
            language=language,
            lookback_days=lookback_days,
            max_articles=max_articles,
        )
        if not df.empty:
            return df
        logger.warning("GNews returned no rows for query=%s; trying RSS fallback.", query)

    # Priority 3: no-key Google News RSS
    logger.info("Using Google News RSS fallback for query=%s", query)
    return _fetch_google_news_rss(
        query=query,
        lookback_days=lookback_days,
        max_articles=max_articles,
    )
