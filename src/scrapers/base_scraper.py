from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup as bs
import pandas as pd
import time
import random
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseScraper:
    """Base class for all FBRef scrapers with common Selenium functionality"""

    def __init__(self, headless=True, implicit_wait=10):
        self.driver = None
        self.wait = None
        self.setup_driver(headless, implicit_wait) # sets up the driver so that its children classes won't have to

    def setup_driver(self, headless=True, implicit_wait=10):
        """Set up Chrome WebDriver with anti-detection measures"""
        chrome_options = Options()

        # Anti-detection measures
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        # Realistic browser settings
        chrome_options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-images")

        if headless:
            chrome_options.add_argument("--headless")

        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(implicit_wait)
            self.wait = WebDriverWait(self.driver, 15)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            logger.info("WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise

    def random_delay(self, min_seconds=1, max_seconds=3):
        """Add random delay to mimic human behavior"""
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)

    def safe_get(self, url, retries=3):
        """Safely get a URL with retries"""
        for attempt in range(retries):
            try:
                logger.info(f"Accessing: {url} (Attempt {attempt + 1})")
                self.driver.get(url)

                # was implemented to prevent 403 error from arising, commented because it slowed down scrapingW
                #self.random_delay(2, 4)

                if "403" in self.driver.title or "Forbidden" in self.driver.page_source:
                    logger.warning(f"403 Forbidden encountered on attempt {attempt + 1}")
                    if attempt < retries - 1:
                        self.random_delay(5, 10)
                        continue
                    else:
                        raise Exception("403 Forbidden - Access denied")

                return True

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    self.random_delay(5, 10)
                else:
                    raise
        return False

    def close(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()
            logger.info("WebDriver closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    try:
        with BaseScraper() as scraper:
            scraper.setup_driver(headless=True)

            # Test basic access first
            test_urls = [
                'https://fbref.com',
                'https://fbref.com/en/comps/9/Premier-League-Stats'
                ''
            ]

            for url in test_urls:
                try:
                    result = scraper.safe_get(url)
                    if result:
                        print(f"✓ Successfully accessed: {url}")
                        print(f"  Title: {scraper.driver.title}")
                    else:
                        print(f"✗ Failed to access: {url}")
                    time.sleep(3)
                except Exception as e:
                    print(f"✗ Exception for {url}: {e}")

        print('Enhanced BaseScraper test completed')

    except Exception as e:
        print(f"Enhanced BaseScraper test failed: {e}")

if __name__ == '__main__':
    main()