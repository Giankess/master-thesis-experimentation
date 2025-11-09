"""Web scraper for financial news using BeautifulSoup."""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
import time


class FinancialDataScraper:
    """Scraper for financial news articles and data."""

    def __init__(self, user_agent: Optional[str] = None):
        """
        Initialize the scraper.

        Args:
            user_agent: Optional custom user agent string
        """
        self.user_agent = user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

    def scrape_url(self, url: str, parser: str = "html.parser") -> Optional[BeautifulSoup]:
        """
        Scrape a single URL and return BeautifulSoup object.

        Args:
            url: URL to scrape
            parser: Parser to use (default: html.parser)

        Returns:
            BeautifulSoup object or None if failed
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.content, parser)
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    def extract_text_from_soup(self, soup: BeautifulSoup, tag: str = "p") -> str:
        """
        Extract text from BeautifulSoup object.

        Args:
            soup: BeautifulSoup object
            tag: HTML tag to extract (default: p for paragraphs)

        Returns:
            Extracted text
        """
        if soup is None:
            return ""
        paragraphs = soup.find_all(tag)
        return " ".join([p.get_text(strip=True) for p in paragraphs])

    def scrape_multiple_urls(
        self, urls: List[str], delay: float = 1.0
    ) -> List[Dict[str, str]]:
        """
        Scrape multiple URLs with delay between requests.

        Args:
            urls: List of URLs to scrape
            delay: Delay between requests in seconds

        Returns:
            List of dictionaries with url, text, and timestamp
        """
        results = []
        for url in urls:
            soup = self.scrape_url(url)
            if soup:
                text = self.extract_text_from_soup(soup)
                results.append(
                    {
                        "url": url,
                        "text": text,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            time.sleep(delay)
        return results

    def scrape_to_dataframe(self, urls: List[str], delay: float = 1.0) -> pd.DataFrame:
        """
        Scrape multiple URLs and return as DataFrame.

        Args:
            urls: List of URLs to scrape
            delay: Delay between requests in seconds

        Returns:
            DataFrame with scraped data
        """
        results = self.scrape_multiple_urls(urls, delay)
        return pd.DataFrame(results)

    def extract_article_metadata(
        self, soup: BeautifulSoup
    ) -> Dict[str, Optional[str]]:
        """
        Extract metadata from article (title, date, author).

        Args:
            soup: BeautifulSoup object

        Returns:
            Dictionary with metadata
        """
        metadata = {"title": None, "date": None, "author": None}

        if soup is None:
            return metadata

        # Try to extract title
        title_tag = soup.find("title") or soup.find("h1")
        if title_tag:
            metadata["title"] = title_tag.get_text(strip=True)

        # Try to extract date (common meta tags)
        date_meta = soup.find("meta", {"property": "article:published_time"}) or soup.find(
            "meta", {"name": "date"}
        )
        if date_meta and date_meta.get("content"):
            metadata["date"] = date_meta["content"]

        # Try to extract author
        author_meta = soup.find("meta", {"name": "author"}) or soup.find(
            "meta", {"property": "article:author"}
        )
        if author_meta and author_meta.get("content"):
            metadata["author"] = author_meta["content"]

        return metadata
