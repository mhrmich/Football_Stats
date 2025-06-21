from datetime import datetime
import random
import re
import time
from typing import Dict, List, Any

from base_scraper import BaseScraper
from bs4 import BeautifulSoup as bs
import logging

logger = logging.getLogger(__name__)

# Default dictionary containing the ID number for each of the top 5 leagues in the FBRef site


class LeagueScraper(BaseScraper):

    def __init__(self, headless=True, implicit_wait=10):
        super().__init__(headless, implicit_wait)
        self.leagues_config = {

            #"Premier-League": 9,
            #"La-Liga": 12,
            #"Serie-A": 11,
            #"Bundesliga": 20,
            "Ligue-1": 13,
            #"EFL-Championship": 10,
        }

    def get_team_links(self, season_url) -> dict[str, list[str | Any] | str | Any]:
        """

        :param season_url: URL for the FBRef page for a given league season
        :return: list of URLs for match log pages for every team in the league
        """
        try:
            # attempts to safely obtain and read the inputted season URL
            self.safe_get(season_url)
            soup = bs(self.driver.page_source, 'html.parser')

            base_url = 'https://fbref.com'
            team_links = []

            # finds a table on the page with 'stats_table' in the class name, throws and error if not found
            table = soup.find('table', class_=lambda c: c and 'stats_table' in c)
            if not table:
                logger.error("Could not find stats table")
                return []

            # iterates across all rows in the table
            rows = table.find_all('tr')
            for row in rows:
                # skips if the row is a header row (which appear throughout in FBRef tables)
                if row.get('class') and 'thead' in row.get('class'):
                    continue

                # finds the cell of the table containing the team name
                team_cell = row.find('td', {'data-stat': 'team'})
                if team_cell:
                    link = team_cell.find('a')
                    if link:
                        team_links.append(base_url + link.get('href'))
                        logger.info(f"Found team: {link.text.strip()}")

            logger.info(f"Found {len(team_links)} teams")

            league_season_info = self._extract_league_from_season_url(season_url)
            return {
                'league': league_season_info['league'],
                'season': league_season_info['season'],
                'team_links': team_links
            }

        except Exception as e:
            logger.error(f"Error getting team links: {e}")
            return []

    def _extract_league_from_season_url(self, season_url):
        '''

        :param season_url: Formatted like 'https://fbref.com/en/comps/9/2023-2024/2023-2024-Premier-League-Stats'
        :return: dictionary containing the year and league name
        '''

        url_split = season_url.split('/')
        last_part = url_split[-1]
        season = re.search(r'(\d{4}-\d{4})', last_part).group() # extracts the 'XXXX-XXXX' from the URL
        league_name = last_part.replace(f"{season}-", "" ).replace("-Stats", "") # extracts just the league name

        return {
            "season": season,
            "league": league_name
        }


    def scrape_league(self, league_season_url):
        """
        Complete workflow to scrape all teams' statistics for a given league season
        :param league_url: format like 'https://fbref.com/en/comps/9/2023-2024/2023-2024-Premier-League-Stats'
        :return:
        """
        from team_scraper import TeamScraper

        try:

            logger.info(f"Starting full league scrape for: {league_season_url}")
            league_info = self.get_team_links(league_season_url) # dictionary with 'league', 'season', 'team_links' keys

            # Throws an error if there are no teams found in the league
            if not league_info or not league_info.get('team_links'):
                logger.error("No team links found")
                return []

            logger.info(f"Found {len(league_info['team_links'])} teams to scrape")

            # Initialize match scraper with league information
            with TeamScraper() as team_scraper:
                team_scraper.set_league_info(league_info['league'], league_info['season'])

                saved_files = []
                failed_teams = []

                for i, team_url in enumerate(league_info['team_links']):
                    try:
                        team_name = team_scraper._extract_team_name_from_url(team_url)
                        logger.info(f"Scraping team {i}/{len(league_info['team_links'])}: {team_name}")

                        file_path = team_scraper.save_match_data(team_url)
                        saved_files.append(file_path)

                    except Exception as e:
                        team_name = team_scraper._extract_team_name_from_url(team_url) if team_url else "Unknown"
                        logger.error(f"Failed to scrape {team_name}: {e}")
                        failed_teams.append(team_name)
                        continue


        except Exception as e:
            logger.error(f"Error getting league info: {e}")


    def _generate_seasons(self, start_season='2017-2018', end_season='2024-2025'):
        """
        Generates a list of seasons in the format of "2024-2025" to use in the URLs for scraping
        """
        start_year = int(start_season.split('-')[0])
        end_year = int(end_season.split('-')[0])

        seasons = []
        for year in range(start_year, end_year+1): # loops from 2017 to 2024
            seasons.append(f"{year}-{year+1}") # appends in the form "2017-2018"

        return seasons

    def _build_league_season_url(self, league_name, league_id, season):
        """
        URL for a specific league and season, i.e. "fbref.com/en/comps/9/2024-2025/2024-2025-Premier-League-Stats"
        :param league_name: league name as found in URL, like "Premier-League"
        :param league_id: the number found after /comps/, like 9 for the premier league
        :param season: "2024-2025" or similar
        :return: "fbref.com/en/comps/9/2024-2025/2024-2025-Premier-League-Stats" or similar
        """

        return f"https://fbref.com/en/comps/{league_id}/{season}/{season}-{league_name}-stats"




    def scrape_multiple_leagues_seasons(self, leagues_config, start_season='2017-2018', end_season='2023-2024'):
        """

        :param leagues_config: Dictionary mapping the numerical ID of each league season, i.e. "Premier-League":9
        :param start_season: the first season to scrape (defaults to 2017-2018, the first season with full data)
        :param end_season: the last season to scrape (defaults to 2024-2025)
        :return:
        """

        if leagues_config is None:
            leagues_config = self.leagues_config

        seasons = self._generate_seasons(start_season, end_season)

        total_leagues = len(leagues_config)
        total_seasons = len(seasons)
        total_combinations = total_leagues * total_seasons

        successful_scrapes = []
        failed_scrapes = []

        combination_count = 0

        for league_name, league_id in leagues_config.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Starting league: {league_name}")
            logger.info(f"{'='*50}")

            for season in seasons:
                combination_count += 1

                try:
                    logger.info(f"\nProcessing {combination_count}/{total_combinations}: {season} {league_name}")
                    season_url = self._build_league_season_url(league_name, league_id, season)

                    # begins scraping the league and logs the time and duration of scraping
                    start_time = datetime.now()
                    self.scrape_league(season_url)
                    end_time = datetime.now()

                    duration = (end_time - start_time).total_seconds()
                    logger.info(f"✅ Successfully completed {league_name} {season} in {duration:.1f}s")

                    successful_scrapes.append({
                        "league": league_name,
                        "season": season,
                        "duration": duration,
                        "timestamp": end_time
                    })

                    delay = random.uniform(30, 60)
                    time.sleep(delay)

                except Exception as e:
                    logger.error(f"❌ Failed to scrape {league_name} {season}: {e}")
                    failed_scrapes.append({
                        'league': league_name,
                        'season': season,
                        'error': str(e),
                        'timestamp': datetime.now()
                    })

                    # Continue with next combination even if one fails
                    continue



def main():
    try:
        with LeagueScraper() as scraper:
            scraper.scrape_multiple_leagues_seasons(scraper.leagues_config, start_season='2022-2023', end_season='2023-2024')

    except Exception as e:
        print(f"LeagueScraper test failed: {e}")

if __name__ == '__main__':
    main()