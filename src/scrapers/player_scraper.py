from src.scrapers.base_scraper import BaseScraper
from bs4 import BeautifulSoup as bs


class PlayerScraper(BaseScraper):

    def __init__(self, headless=True, implicit_wait=10):
        super().__init__(headless, implicit_wait)

    def scrape_league_players(self, league_url):
        try:
            self.safe_get(league_url)
            soup = bs(self.driver.page_source, 'html.parser')

        except Exception as e:
            print(e)