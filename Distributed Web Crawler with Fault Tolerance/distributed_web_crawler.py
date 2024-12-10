import threading
import requests
from queue import Queue
from bs4 import BeautifulSoup

class DistributedWebCrawler:
    def __init__(self, seed_urls, max_threads=10):
        """
        Initializes the distributed web crawler with seed URLs and a thread pool.
        """
        self.queue = Queue()
        self.visited = set()
        self.max_threads = max_threads

        for url in seed_urls:
            self.queue.put(url)

    def fetch_url(self, url):
        """
        Fetches and parses the content of a URL.
        """
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def parse_links(self, html, base_url):
        """
        Extracts and enqueues all links from a page.
        """
        soup = BeautifulSoup(html, "html.parser")
        for link in soup.find_all("a", href=True):
            full_url = requests.compat.urljoin(base_url, link["href"])
            if full_url not in self.visited:
                self.queue.put(full_url)

    def crawl(self):
        """
        The main crawl logic for a single thread.
        """
        while not self.queue.empty():
            url = self.queue.get()
            if url in self.visited:
                continue
            print(f"Crawling: {url}")
            self.visited.add(url)
            html = self.fetch_url(url)
            if html:
                self.parse_links(html, url)

    def start(self):
        """
        Starts the distributed web crawler with multithreading.
        """
        threads = []
        for _ in range(self.max_threads):
            thread = threading.Thread(target=self.crawl)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

# Example Usage:
# seed_urls = ["https://example.com"]
# crawler = DistributedWebCrawler(seed_urls, max_threads=5)
# crawler.start()
