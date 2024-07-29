import asyncio
import aiohttp
from aiohttp import ClientSession, TCPConnector
from aiohttp_retry import RetryClient, ExponentialRetry
from urllib.parse import urljoin, urlparse
from lxml import html
import json
import re
import time
import logging
import argparse
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
from collections import defaultdict
import aiofiles
from async_timeout import timeout
from urllib.robotparser import RobotFileParser
from fake_useragent import UserAgent
import random
import asyncio_throttle
from tenacity import retry, stop_after_attempt, wait_exponential
import signal
import spacy
from transformers import pipeline
from bs4 import BeautifulSoup
import csv
import pandas as pd
from langdetect import detect
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load NLP models
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading spaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

sentiment_analyzer = pipeline("sentiment-analysis")

@dataclass
class AIOptimizedScrapedData:
    url: str
    title: str
    text_content: str
    cleaned_text: str
    html_content: str
    links: List[str]
    metadata: Dict[str, Any]
    status_code: int
    headers: Dict[str, str]
    language: str
    named_entities: Dict[str, List[str]]
    sentiment: Dict[str, float]
    word_count: int
    sentence_count: int
    paragraph_count: int
    domain_specific_data: Dict[str, Any]
    hash: str

class AdaptiveRateLimiter:
    def __init__(self, initial_rate: float = 1.0, max_rate: float = 10.0, decrease_factor: float = 0.5):
        self.current_rate = initial_rate
        self.max_rate = max_rate
        self.decrease_factor = decrease_factor
        self.throttler = asyncio_throttle.Throttler(rate_limit=self.current_rate)

    async def acquire(self):
        await self.throttler.acquire()

    def increase_rate(self):
        self.current_rate = min(self.current_rate * 2, self.max_rate)
        self.throttler.rate_limit = self.current_rate
        logger.info(f"Increased rate limit to {self.current_rate}")

    def decrease_rate(self):
        self.current_rate *= self.decrease_factor
        self.throttler.rate_limit = self.current_rate
        logger.info(f"Decreased rate limit to {self.current_rate}")

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning("Circuit breaker opened")

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def can_proceed(self):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "HALF-OPEN"
                logger.info("Circuit breaker half-open")
            return self.state != "OPEN"
        return True

class AIOptimizedWebScraper:
    def __init__(self, base_url: str, max_pages: int = 100, max_depth: int = 3,
                 concurrency: int = 10, initial_rate_limit: float = 1.0,
                 proxy_list: List[str] = None, domain_specific_extractors: Dict[str, callable] = None):
        self.base_url = base_url
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.concurrency = concurrency
        self.rate_limiter = AdaptiveRateLimiter(initial_rate=initial_rate_limit)
        self.visited_urls = set()
        self.to_visit = asyncio.Queue()
        self.data = defaultdict(list)
        self.session: Optional[ClientSession] = None
        self.robots_parser = RobotFileParser()
        self.user_agent = UserAgent()
        self.proxy_list = proxy_list or []
        self.circuit_breaker = CircuitBreaker()
        self.retry_client: Optional[RetryClient] = None
        self.robots_txt_fetched = False
        self.domain_specific_extractors = domain_specific_extractors or {}

    async def initialize(self):
        connector = TCPConnector(limit=self.concurrency, ttl_dns_cache=300)
        self.session = ClientSession(connector=connector)
        retry_options = ExponentialRetry(attempts=3)
        self.retry_client = RetryClient(client_session=self.session, retry_options=retry_options)
        await self.setup_robots_parser()

    async def setup_robots_parser(self):
        robots_url = urljoin(self.base_url, '/robots.txt')
        try:
            async with self.session.get(robots_url) as response:
                if response.status == 200:
                    robots_content = await response.text()
                    self.robots_parser.parse(robots_content.splitlines())
                    logger.info("Successfully parsed robots.txt")
                    self.robots_txt_fetched = True
                else:
                    logger.warning(f"Failed to fetch robots.txt: Status code {response.status}")
                    self.robots_txt_fetched = False
        except Exception as e:
            logger.error(f"Error fetching robots.txt: {str(e)}")
            self.robots_txt_fetched = False

    async def crawl(self):
        await self.to_visit.put((self.base_url, 0))
        tasks = [asyncio.create_task(self.worker()) for _ in range(self.concurrency)]
        await self.to_visit.join()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def worker(self):
        while True:
            try:
                url, depth = await self.to_visit.get()
                if url not in self.visited_urls and len(self.visited_urls) < self.max_pages and depth <= self.max_depth:
                    if not self.robots_txt_fetched or self.robots_parser.can_fetch(self.user_agent.random, url):
                        await self.scrape_page(url, depth)
                    else:
                        logger.info(f"Skipping {url} as per robots.txt")
                self.to_visit.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in worker: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def scrape_page(self, url: str, depth: int):
        if not self.circuit_breaker.can_proceed():
            logger.warning(f"Circuit breaker open, skipping {url}")
            return

        logger.info(f"Scraping {url} at depth {depth}")
        self.visited_urls.add(url)
        try:
            await self.rate_limiter.acquire()
            headers = {'User-Agent': self.user_agent.random}
            proxy = random.choice(self.proxy_list) if self.proxy_list else None
            
            async with timeout(30):
                async with self.retry_client.get(url, headers=headers, proxy=proxy) as response:
                    if response.status == 200:
                        content = await response.text()
                        await self.process_content(url, content, depth, response.status, dict(response.headers))
                        self.rate_limiter.increase_rate()
                        self.circuit_breaker.record_success()
                    else:
                        logger.warning(f"Failed to fetch {url}: Status code {response.status}")
                        self.rate_limiter.decrease_rate()
                        self.circuit_breaker.record_failure()
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            self.rate_limiter.decrease_rate()
            self.circuit_breaker.record_failure()
            raise

    async def process_content(self, url: str, content: str, depth: int, status_code: int, headers: Dict[str, str]):
        try:
            soup = BeautifulSoup(content, 'lxml')
            tree = html.fromstring(content)
            
            title = self.extract_text(soup.title.string) if soup.title else ""
            text_content = self.extract_text(soup.get_text())
            cleaned_text = self.clean_text(text_content)
            links = self.extract_links(tree, url)
            metadata = self.extract_metadata(tree)
            
            language = detect(cleaned_text)
            named_entities = self.extract_named_entities(cleaned_text)
            sentiment = self.analyze_sentiment(cleaned_text)
            word_count = len(cleaned_text.split())
            sentence_count = len(list(nlp(cleaned_text).sents))
            paragraph_count = len(soup.find_all('p'))
            
            domain_specific_data = self.extract_domain_specific_data(soup, url)
            
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            scraped_data = AIOptimizedScrapedData(
                url=url,
                title=title,
                text_content=text_content,
                cleaned_text=cleaned_text,
                html_content=content,
                links=links,
                metadata=metadata,
                status_code=status_code,
                headers=headers,
                language=language,
                named_entities=named_entities,
                sentiment=sentiment,
                word_count=word_count,
                sentence_count=sentence_count,
                paragraph_count=paragraph_count,
                domain_specific_data=domain_specific_data,
                hash=content_hash
            )

            self.data[url].append(asdict(scraped_data))

            for link in links:
                if link not in self.visited_urls:
                    await self.to_visit.put((link, depth + 1))

        except Exception as e:
            logger.error(f"Error processing content from {url}: {str(e)}")

    @staticmethod
    def extract_text(text: str) -> str:
        return ' '.join(text.split()) if text else ""

    @staticmethod
    def clean_text(text: str) -> str:
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        return text

    @staticmethod
    def extract_links(tree, base_url: str) -> List[str]:
        links = []
        for href in tree.xpath('//a/@href'):
            full_url = urljoin(base_url, href)
            if urlparse(full_url).netloc == urlparse(base_url).netloc:
                links.append(full_url)
        return links

    @staticmethod
    def extract_metadata(tree) -> Dict[str, Any]:
        metadata = {}
        for meta in tree.xpath('//meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[name] = content
        return metadata

    @staticmethod
    def extract_named_entities(text: str) -> Dict[str, List[str]]:
        doc = nlp(text)
        entities = defaultdict(list)
        for ent in doc.ents:
            entities[ent.label_].append(ent.text)
        return dict(entities)

    @staticmethod
    def analyze_sentiment(text: str) -> Dict[str, float]:
        result = sentiment_analyzer(text[:512])[0]  # Limit to 512 tokens
        return {"label": result["label"], "score": float(result["score"])}

    def extract_domain_specific_data(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        domain = urlparse(url).netloc
        extractor = self.domain_specific_extractors.get(domain)
        if extractor:
            return extractor(soup)
        return {}

    async def run(self):
        await self.initialize()
        await self.crawl()
        await self.session.close()
        return dict(self.data)

    def save_data(self, output_file: str):
        # Save as JSON
        with open(f"{output_file}.json", "w") as f:
            json.dump(self.data, f, indent=2)

        # Save as CSV
        flat_data = []
        for url, data_list in self.data.items():
            for data in data_list:
                flat_data.append({
                    "url": data["url"],
                    "title": data["title"],
                    "cleaned_text": data["cleaned_text"],
                    "language": data["language"],
                    "sentiment": data["sentiment"]["label"],
                    "sentiment_score": data["sentiment"]["score"],
                    "word_count": data["word_count"],
                    "sentence_count": data["sentence_count"],
                    "paragraph_count": data["paragraph_count"],
                    "named_entities": json.dumps(data["named_entities"]),
                    "domain_specific_data": json.dumps(data["domain_specific_data"]),
                    "hash": data["hash"]
                })
        
        df = pd.DataFrame(flat_data)
        df.to_csv(f"{output_file}.csv", index=False)

        logger.info(f"Data saved to {output_file}.json and {output_file}.csv")

async def main(args):
    # Example domain-specific extractor
    def extract_news_data(soup):
        return {
            "author": soup.find("meta", {"name": "author"})["content"] if soup.find("meta", {"name": "author"}) else "",
            "published_date": soup.find("meta", {"name": "publishedDate"})["content"] if soup.find("meta", {"name": "publishedDate"}) else "",
            "category": soup.find("meta", {"name": "category"})["content"] if soup.find("meta", {"name": "category"}) else "",
        }

    domain_specific_extractors = {
        "news.example.com": extract_news_data,
        # Add more domain-specific extractors here
    }

    scraper = AIOptimizedWebScraper(
        base_url=args.url,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        concurrency=args.concurrency,
        initial_rate_limit=args.rate_limit,
        proxy_list=args.proxies,
        domain_specific_extractors=domain_specific_extractors
    )
    await scraper.run()
    
    output_file = args.output or 'ai_optimized_scraped_data'
    scraper.save_data(output_file)

    logger.info(f"Scraping completed. Processed {len(scraper.visited_urls)} pages. Data saved to {output_file}.[json/csv/parquet]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-Optimized Web Scraper 2024")
    parser.add_argument("url", help="Base URL to scrape")
    parser.add_argument("--max-pages", type=int, default=100, help="Maximum number of pages to scrape")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum depth to crawl")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument("--rate-limit", type=float, default=1.0, help="Initial number of requests per second")
    parser.add_argument("--output", help="Output file name (without extension)")
    parser.add_argument("--proxies", nargs='+', help="List of proxy servers to use")
    
    args = parser.parse_args()
    
    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(
            s, lambda s=s: asyncio.create_task(shutdown(loop, signal=s))
        )
    
    try:
        loop.run_until_complete(main(args))
    finally:
        loop.close()

async def shutdown(loop, signal=None):
    if signal:
        logger.info(f"Received exit signal {signal.name}...")
    logger.info("Shutting down gracefully...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()
