import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import threading

class MultithreadedWebCrawler:
    def __init__(self, base_url, max_threads=10):
        """
        Initializes the web crawler.
        Args:
            base_url (str): The starting URL.
            max_threads (int): Maximum number of threads.
        """
        self.base_url = base_url
        self.visited_urls = set()
        self.lock = threading.Lock()
        self.max_threads = max_threads

    def fetch_links(self, url):
        """
        Fetches all links from the given URL.
        Args:
            url (str): The target URL.
        """
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.find_all('a', href=True):
                    full_url = urljoin(self.base_url, link['href'])
                    if self.is_valid_url(full_url):
                        with self.lock:
                            if full_url not in self.visited_urls:
                                self.visited_urls.add(full_url)
                                threading.Thread(target=self.fetch_links, args=(full_url,)).start()
        except Exception as e:
            print(f"Error fetching {url}: {e}")

    def is_valid_url(self, url):
        """
        Validates if the URL is within the same domain.
        Args:
            url (str): The URL to validate.
        """
        parsed_base = urlparse(self.base_url)
        parsed_url = urlparse(url)
        return parsed_base.netloc == parsed_url.netloc

    def crawl(self):
        """
        Starts crawling from the base URL.
        """
        self.visited_urls.add(self.base_url)
        threads = []
        for _ in range(self.max_threads):
            thread = threading.Thread(target=self.fetch_links, args=(self.base_url,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

# Example usage:
# crawler = MultithreadedWebCrawler("https://example.com")
# crawler.crawl()
# print("Visited URLs:", crawler.visited_urls)
