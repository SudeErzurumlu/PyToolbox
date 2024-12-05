import requests
from bs4 import BeautifulSoup
import csv

class AdvancedScraper:
    def __init__(self, base_url, output_file):
        """
        Initializes the scraper.
        Args:
            base_url (str): The URL to scrape data from.
            output_file (str): The CSV file to save the data.
        """
        self.base_url = base_url
        self.output_file = output_file

    def scrape_page(self, url):
        """
        Scrapes data from a single page.
        Args:
            url (str): URL of the page to scrape.
        Returns:
            list: List of dictionaries containing product data.
        """
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            products = []
            for product in soup.select(".product-card"):
                name = product.select_one(".product-title").text.strip()
                price = product.select_one(".product-price").text.strip()
                rating = product.select_one(".product-rating").text.strip() if product.select_one(".product-rating") else "N/A"
                products.append({"Name": name, "Price": price, "Rating": rating})
            return products
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return []

    def save_to_csv(self, data):
        """
        Saves scraped data to a CSV file.
        Args:
            data (list): List of dictionaries containing scraped data.
        """
        keys = data[0].keys()
        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)

    def scrape_all_pages(self):
        """
        Scrapes all pages from the base URL.
        """
        all_products = []
        page = 1
        while True:
            url = f"{self.base_url}?page={page}"
            products = self.scrape_page(url)
            if not products:
                break
            all_products.extend(products)
            print(f"Scraped page {page}")
            page += 1
        if all_products:
            self.save_to_csv(all_products)
            print(f"Data saved to {self.output_file}")

# Example usage:
# scraper = AdvancedScraper("https://example-ecommerce.com/products", "products.csv")
# scraper.scrape_all_pages()
