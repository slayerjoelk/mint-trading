import re
import hashlib
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Optional
from email.utils import parsedate_to_datetime

try:
    import feedparser
    _FEEDPARSER = True
except ImportError:
    _FEEDPARSER = False

RSS_FEEDS = {
    "Reuters":     "https://feeds.reuters.com/reuters/businessNews",
    "CNBC":        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_marketpulse",
    "Bloomberg":   "https://feeds.bloomberg.com/markets/news.rss",
}

POSITIVE_WORDS = {
    "beat", "beats", "surge", "surges", "surged", "rally", "rallies", "rallied",
    "upgrade", "upgrades", "upgraded", "buy", "growth", "profit", "profits",
    "rise", "rises", "rose", "gain", "gains", "gained", "record", "strong",
    "exceed", "exceeds", "exceeded", "boost", "boosts", "boosted", "outperform",
    "top", "tops", "topped", "better", "higher", "increase", "increases",
}

NEGATIVE_WORDS = {
    "miss", "misses", "missed", "drop", "drops", "dropped", "decline", "declines",
    "declined", "downgrade", "downgrades", "downgraded", "sell", "loss", "losses",
    "crash", "crashes", "crashed", "fall", "falls", "fell", "weak", "disappoint",
    "disappoints", "disappointed", "cut", "cuts", "below", "concern", "concerns",
    "risk", "risks", "warning", "warnings", "lower", "decrease", "decreases",
    "layoff", "layoffs", "recall", "recalls", "investigation", "fraud",
}

# Company name fragments → ticker (longest match wins)
COMPANY_TICKER_MAP: dict[str, str] = {
    "apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN", "google": "GOOGL",
    "alphabet": "GOOGL", "meta platforms": "META", "meta": "META",
    "facebook": "META", "nvidia": "NVDA", "tesla": "TSLA",
    "berkshire hathaway": "BRK.B", "berkshire": "BRK.B",
    "jpmorgan chase": "JPM", "jpmorgan": "JPM", "jp morgan": "JPM",
    "johnson & johnson": "JNJ", "johnson and johnson": "JNJ",
    "unitedhealth": "UNH", "exxon mobil": "XOM", "exxon": "XOM",
    "visa": "V", "mastercard": "MA", "procter & gamble": "PG",
    "procter and gamble": "PG", "walmart": "WMT", "eli lilly": "LLY",
    "chevron": "CVX", "abbvie": "ABBV", "home depot": "HD",
    "merck": "MRK", "pepsico": "PEP", "pepsi": "PEP",
    "broadcom": "AVGO", "costco": "COST", "coca-cola": "KO", "coke": "KO",
    "adobe": "ADBE", "salesforce": "CRM", "intel": "INTC", "cisco": "CSCO",
    "netflix": "NFLX", "comcast": "CMCSA", "wells fargo": "WFC",
    "bank of america": "BAC", "pfizer": "PFE", "disney": "DIS",
    "caterpillar": "CAT", "mcdonald's": "MCD", "mcdonalds": "MCD",
    "nike": "NKE", "boeing": "BA", "at&t": "T", "verizon": "VZ",
    "general electric": "GE", "ibm": "IBM", "qualcomm": "QCOM",
    "advanced micro devices": "AMD", "amd": "AMD",
    "uber": "UBER", "airbnb": "ABNB", "palantir": "PLTR",
    "snowflake": "SNOW", "shopify": "SHOP", "paypal": "PYPL",
    "block": "SQ", "square": "SQ", "twilio": "TWLO", "zoom": "ZM",
    "spotify": "SPOT", "lyft": "LYFT", "snap": "SNAP", "pinterest": "PINS",
    "coinbase": "COIN", "roblox": "RBLX", "draftkings": "DKNG",
    "amc entertainment": "AMC", "gamestop": "GME", "rivian": "RIVN",
    "lucid motors": "LCID", "lucid": "LCID", "ford": "F",
    "general motors": "GM", "stellantis": "STLA", "starbucks": "SBUX",
    "target": "TGT", "lowe's": "LOW", "lowes": "LOW", "cvs health": "CVS",
    "cvs": "CVS", "walgreens": "WBA", "kroger": "KR",
    "dollar general": "DG", "dollar tree": "DLTR",
    "fedex": "FDX", "ups": "UPS", "deere": "DE", "lockheed martin": "LMT",
    "lockheed": "LMT", "raytheon": "RTX", "northrop grumman": "NOC",
    "general dynamics": "GD", "halliburton": "HAL", "conocophillips": "COP",
    "marathon petroleum": "MPC", "valero": "VLO",
    "american express": "AXP", "amex": "AXP",
    "goldman sachs": "GS", "morgan stanley": "MS",
    "citigroup": "C", "citi": "C", "blackrock": "BLK",
    "charles schwab": "SCHW", "servicenow": "NOW", "workday": "WDAY",
    "crowdstrike": "CRWD", "datadog": "DDOG", "cloudflare": "NET",
    "fortinet": "FTNT", "palo alto networks": "PANW", "palo alto": "PANW",
    "zscaler": "ZS", "okta": "OKTA", "sentinelone": "S",
    "mongodb": "MDB", "elastic": "ESTC", "splunk": "SPLK", "veeva": "VEEV",
    "intuitive surgical": "ISRG", "danaher": "DHR",
    "thermo fisher": "TMO", "edwards lifesciences": "EW",
    "boston scientific": "BSX", "regeneron": "REGN",
    "moderna": "MRNA", "biogen": "BIIB", "gilead": "GILD",
    "vertex pharmaceuticals": "VRTX", "vertex": "VRTX",
    "amgen": "AMGN", "bristol-myers squibb": "BMY", "bristol myers": "BMY",
    "prologis": "PLD", "american tower": "AMT", "crown castle": "CCI",
    "equinix": "EQIX", "digital realty": "DLR", "simon property": "SPG",
    "public storage": "PSA", "welltower": "WELL",
}

KNOWN_TICKERS = set(COMPANY_TICKER_MAP.values())


def _now_unix() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _parse_ts(raw: str) -> int:
    """Parse an RSS date string to a Unix timestamp, falling back to now."""
    try:
        dt = parsedate_to_datetime(raw)
        return int(dt.timestamp())
    except Exception:
        pass
    try:
        return int(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp())
    except Exception:
        return _now_unix()


class NewsPipeline:
    def __init__(self, db):
        self.db = db
        self._cache: list[dict] = []
        self._cache_ts: int = 0
        self._cache_ttl: int = 300  # 5 minutes

    def fetch_headlines(self, max_per_source: int = 20) -> list[dict]:
        now = _now_unix()
        if self._cache and (now - self._cache_ts) < self._cache_ttl:
            return self._cache

        results: list[dict] = []
        for source, url in RSS_FEEDS.items():
            try:
                items = self._fetch_feed(url, max_per_source)
                for item in items:
                    headline = item["headline"]
                    ticker = self.extract_ticker_from_headline(headline)
                    sentiment = self.score_sentiment(headline)
                    ts = item["timestamp"]
                    news_id = hashlib.md5(f"{headline}{ts}".encode()).hexdigest()
                    entry = {
                        "id": news_id,
                        "ticker": ticker,
                        "headline": headline,
                        "source": source,
                        "timestamp": ts,
                        "sentiment": sentiment,
                    }
                    results.append(entry)
                    self.db.record_news(
                        id=news_id,
                        ticker=ticker,
                        headline=headline,
                        sentiment=sentiment,
                        source=source,
                        published_at=ts,
                        recorded_at=now,
                    )
            except Exception as exc:
                print(f"[NewsPipeline] {source} error: {exc}")

        self._cache = results
        self._cache_ts = now
        return results

    def _fetch_feed(self, url: str, limit: int) -> list[dict]:
        if _FEEDPARSER:
            return self._parse_with_feedparser(url, limit)
        return self._parse_with_http(url, limit)

    def _parse_with_feedparser(self, url: str, limit: int) -> list[dict]:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:limit]:
            title = getattr(entry, "title", "").strip()
            if not title:
                continue
            raw_date = getattr(entry, "published", "") or getattr(entry, "updated", "")
            items.append({"headline": title, "timestamp": _parse_ts(raw_date)})
        return items

    def _parse_with_http(self, url: str, limit: int) -> list[dict]:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
        titles = re.findall(r"<title>\s*<!\[CDATA\[([^\]]{10,200})\]\]>\s*</title>|<title>([^<]{10,200})</title>", body)
        now = _now_unix()
        results = []
        for m in titles[1:limit + 1]:
            text = (m[0] or m[1]).strip()
            if text:
                results.append({"headline": text, "timestamp": now})
        return results

    def extract_ticker_from_headline(self, headline: str) -> Optional[str]:
        hl_lower = headline.lower()
        best_ticker: Optional[str] = None
        best_len = 0
        for name, ticker in COMPANY_TICKER_MAP.items():
            if name in hl_lower and len(name) > best_len:
                best_ticker = ticker
                best_len = len(name)
        if best_ticker:
            return best_ticker
        # Match $TICKER or (TICKER)
        for m in re.finditer(r'\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b', headline):
            candidate = m.group(1) or m.group(2)
            if candidate in KNOWN_TICKERS:
                return candidate
        return None

    def score_sentiment(self, headline: str) -> float:
        words = re.findall(r"\b\w+\b", headline.lower())
        score = sum(1 if w in POSITIVE_WORDS else -1 if w in NEGATIVE_WORDS else 0 for w in words)
        if score == 0:
            return 0.0
        return max(-1.0, min(1.0, score / 3.0))

    def get_ticker_sentiment(self, ticker: str) -> float:
        cutoff = int((datetime.now(timezone.utc) - timedelta(hours=24)).timestamp())
        relevant = [
            h for h in self.fetch_headlines()
            if h.get("ticker") == ticker and h.get("timestamp", 0) >= cutoff
        ]
        if not relevant:
            return 0.0
        return sum(h["sentiment"] for h in relevant) / len(relevant)

    def get_market_sentiment(self) -> float:
        headlines = self.fetch_headlines()
        if not headlines:
            return 0.0
        return sum(h["sentiment"] for h in headlines) / len(headlines)
