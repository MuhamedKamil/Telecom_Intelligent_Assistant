"""
Telecom Egypt (te.eg) knowledge base scraper.

Uses LangChain's RecursiveUrlLoader for the crawl/fetch loop, with a custom
BeautifulSoup extractor that strips the site's nav/footer boilerplate and
keeps only the main content region.

Output: one JSON file per page in ./scraped/, ready for the chunk+embed step.

Usage:
    python scrape_te_eg.py
"""

import json
import re
import time
import urllib.robotparser as robotparser
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader

# ---- Config -----------------------------------------------------------

# Seed URLs. Add more sections as needed (business, sustainability, etc.)
SEED_URLS = [
    "https://www.te.eg/en/personal",
    "https://www.te.eg/ar/personal",
]

# Only crawl these subdomains — skip login-walled apps (my.te.eg, billing.te.eg,
# shop.te.eg) since they're transactional SPAs behind auth, not scrapable content.
ALLOWED_NETLOCS = {"www.te.eg", "te.eg"}

MAX_DEPTH = 4          # how many link-hops from each seed
REQUEST_DELAY = 0.5    # seconds between requests — be a polite crawler
USER_AGENT = "TelecomEgyptKB-Bot/1.0 (+internal RAG knowledge base ingestion)"
OUTPUT_DIR = Path("scraped")

# --------------------------------------------------------------------------


def check_robots_allowed(base_url: str, user_agent: str) -> robotparser.RobotFileParser:
    """Load and parse robots.txt once, so every fetch can be checked against it."""
    rp = robotparser.RobotFileParser()
    parsed = urlparse(base_url)
    rp.set_url(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
    try:
        rp.read()
    except Exception:
        pass  # if robots.txt is unreachable, proceed cautiously with defaults
    return rp


def extract_clean_text(html: str) -> dict:
    """
    Strip nav/header/footer boilerplate and return the main content text,
    plus a title and detected language.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove obviously non-content elements first
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
        tag.decompose()

    # te.eg has a "skip to main content" link -> #main-content. That id is the
    # most reliable anchor for the real content region on this Liferay site.
    main = soup.find(id="main-content") or soup.find("main")

    if main is None:
        # Fallback: remove common boilerplate containers by class/id keywords,
        # then use whatever's left of <body>.
        for tag in soup.find_all(True, {"class": re.compile(r"(menu|footer|header|breadcrumb|nav)", re.I)}):
            tag.decompose()
        main = soup.body or soup

    text = main.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse excess blank lines

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    html_tag = soup.find("html")
    lang = (html_tag.get("lang") if html_tag else None) or ""

    return {"text": text, "title": title, "lang": lang}


def save_document(url: str, extracted: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # filesystem-safe filename derived from the URL path
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", urlparse(url).path).strip("_") or "index"
    path = out_dir / f"{slug}.json"
    record = {
        "url": url,
        "title": extracted["title"],
        "lang": extracted["lang"],
        "text": extracted["text"],
        "source": "te.eg",
        "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def crawl(seed_url: str, rp: robotparser.RobotFileParser) -> list[dict]:
    def link_filter(url: str) -> bool:
        parsed = urlparse(url)
        if parsed.netloc not in ALLOWED_NETLOCS:
            return False
        if not rp.can_fetch(USER_AGENT, url):
            return False
        return True

    loader = RecursiveUrlLoader(
        url=seed_url,
        max_depth=MAX_DEPTH,
        prevent_outside=True,
        check_response_status=True,
        continue_on_failure=True,
        headers={"User-Agent": USER_AGENT},
        extractor=lambda html: extract_clean_text(html)["text"],
        link_regex=None,  # keep default <a href> discovery
    )

    docs = []
    for doc in loader.lazy_load():
        url = doc.metadata.get("source", "")
        if not link_filter(url):
            continue
        extracted = extract_clean_text(doc.metadata.get("html", "") or "")
        # RecursiveUrlLoader already ran our extractor into doc.page_content;
        # re-run extract_clean_text only if raw html was preserved, else use page_content.
        text = extracted["text"] if extracted["text"] else doc.page_content
        if len(text.strip()) < 100:
            continue  # skip near-empty pages (redirects, error pages, etc.)
        record = {"text": text, "title": doc.metadata.get("title", ""), "lang": doc.metadata.get("language", "")}
        save_document(url, record, OUTPUT_DIR)
        docs.append(record)
        time.sleep(REQUEST_DELAY)

    return docs


if __name__ == "__main__":
    rp = check_robots_allowed(SEED_URLS[0], USER_AGENT)
    total = 0
    for seed in SEED_URLS:
        print(f"Crawling from seed: {seed}")
        results = crawl(seed, rp)
        print(f"  -> {len(results)} pages saved")
        total += len(results)
    print(f"Done. {total} pages saved to {OUTPUT_DIR.resolve()}")