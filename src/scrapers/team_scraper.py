import os
from base_scraper import BaseScraper
from league_scraper import LeagueScraper
import re
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup as bs
import logging

logger = logging.getLogger(__name__)

class TeamScraper(BaseScraper):

    def __init__(self, headless=True, implicit_wait=10):
        super().__init__(headless, implicit_wait)
        self.current_league_info = None

    def set_league_info(self, league_name, season):
        """Sets the league info for the current scraping session"""
        self.current_league_info = {
            'league': league_name,
            'season': season,
        }

    # Define all available stat categories, which will be inputted in the URL for match logs
    STAT_CATEGORIES = [
        'shooting',
        'keeper',
        'passing',
        'passing_types',
        'gca', # goal creating actions
        'defense',
        'possession',
        'misc'
    ]

    def get_team_matchlog_url(self, team_url, stat_type):
        """
        Transforms the URL for a team as provided by LeagueScraper.get_team_links into a URL for match logs

        :param team_url: URL of a team from a given season as returned by get_team_links from LeagueScraper
        :return: URL of the FBRef page for match logs for the given team and season
        """

        # partitions the original URL
        url_base = 'https://fbref.com/en/squads'
        parts = team_url.split('/')
        team_id = parts[-3]
        season_id = parts[-2]
        team_name = parts[-1][:-6]

        # rejoins the parts to create the URL for the match log page.
        matchlog_url = f"{url_base}/{team_id}/{season_id}/matchlogs/all_comps/{{stat_type}}/{team_name}-Match-Logs-All-Competitions"
        return matchlog_url.format(stat_type=stat_type)

    def scrape_matchlog(self, matchlog_url):
        """
        Extracts the data table from a given match log page and stores it in a dataframe

        :param matchlog_url: URL for a given stats page (possession, shooting, etc.) for match logs for a team
        :return: returns a Pandas DataFrame containing the match log for a stats category for a team
        """
        try:
            logger.info(f"Scraping detailed match data from: {matchlog_url}")
            self.safe_get(matchlog_url)
            soup = bs(self.driver.page_source, 'html.parser')
            table_html= soup.find('table', class_=lambda c: c and 'stats_table' in c)
            table = pd.read_html(StringIO(str(table_html)))
            return table

        except Exception as e:
            logger.error(f"Failed to scrape match data for {matchlog_url}: {e}")

    def scrape_all_matchlogs(self, team_url):
        """
        Given a team URL, scrapes and concenates every match log into a single dataframe representing every match
        played by the club in a given season, and returns a singular dataframe representing this data

        :param team_url: URL for a given team, as given by get_team_links from LeagueScraper
        :return: dataframe containing all match log data for a club in a given season
        """
        all_dataframes = {}
        for category in self.STAT_CATEGORIES:
            try:
                matchlog_url = self.get_team_matchlog_url(team_url, category)
                print(matchlog_url)
                df = self.scrape_matchlog(matchlog_url)[0]
                if df is not None:
                    all_dataframes[category] = df

            except Exception as e:
                logger.error(f"Failed to scrape match data for {matchlog_url}: {e}")
                continue

        matchlog_df = self._merge_dataframes(all_dataframes)
        return matchlog_df

    def _process_dataframe(self, df):
        """
        Processes dataframe by squishing multi level headers and formatting/standardizing column names

        :param df: Raw dataframe from scraping, containing multi
        :return: Processed dataframe
        """

        # Flattens multi-level headers and cleans their column names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] for col in df.columns]
            df.columns = [self._clean_column_name(col) for col in df.columns]
            df.drop('Match Report', axis=1)
            return df

    def _clean_column_name(self, column_name):
        """
        Clean individual column names by removing unwanted text and standardizing format

        :param column_name: Original column name
        :return: Cleaned and standardized column name
        """

        # Removes "For {Team Name}" from the start of the column name
        column_name = re.sub(r'^For\s+[A-Za-z\s-]+_', '', column_name)

        # Removes the "Unnamed: X_level_y" from the column name
        column_name = re.sub(r'Unnamed:\s*\d+_level_\d+_', '', column_name)

        return column_name

    def _merge_dataframes(self, dataframes_dict):
        """
        Used to merge match log data frames for different statistical categories for a given club and season

        :param dataframes_dict: Dictionary of dataframes by state type
        :return: Merged dataframe
        """

        try:
            df_list = []
            for stat_type, df in dataframes_dict.items():
                df_list.append(df)

            merged_df = pd.concat(df_list, axis=1)
            merged_df = self._process_dataframe(merged_df)
            return merged_df
        except Exception as e:
            logger.error(f"Failed to merge dataframes for {stat_type}: {e}")

    def _extract_team_name_from_url(self, team_url):
        """Extract team name from FBRef URL"""
        # URL format: https://fbref.com/en/squads/{id}/{season}/{team-name}-Stats
        parts = team_url.split('/')
        team_part = parts[-1]  # e.g., "Manchester-City-Stats"
        team_name = team_part.replace('-Stats', '').replace('-', ' ')
        return team_name

    def save_match_data(self, team_url):
        """
        Obtains statistics via scrape_all_matchlogs() and saves the raw data in a CSV file in the correct folder

        :param team_url: URL for a given team, as given by get_team_links from LeagueScraper
        :return:
        """
        scraper_directory = os.path.dirname(os.path.abspath(__file__)) # get the directory of this file

        # get the directory of the whole project (two levels back from the current file)
        project_root = os.path.dirname(os.path.dirname(scraper_directory))
        #match_data = self.scrape_all_matchlogs(team_url)

        league_directory_name = self.current_league_info['league'].replace("-", "_").lower()

        data_directory = os.path.join(
            project_root,
            'data',
            'raw',
            'leagues',
            league_directory_name,
            self.current_league_info['season']
        )

        # creates the directory if it does not already exist
        os.makedirs(data_directory, exist_ok=True)

        # scrapes the match data and stores it in pandas DataFrame match_data
        match_data = self.scrape_all_matchlogs(team_url)

        team_name = self._extract_team_name_from_url(team_url).lower()
        file_path = os.path.join(data_directory, f'{team_name}.csv')

        match_data.to_csv(file_path, index=False)

        return file_path


    def _extract_team_name_from_url(self, team_url):
        """
        Returns a formatted version of the team name to be used for directory. e.g. Manchester City is 'manchester_city'
        :param team_url: Format like 'https://fbref.com/en/squads/b8fd03ef/2023-2024/Manchester-City-Stats'
        :return: name of the team
        """

        last_section = team_url.split('/')[-1]
        formatted_team_name = last_section.replace('-Stats', '').replace('-', '_').lower() #
        return formatted_team_name


def main():
    try:
        pd.set_option('display.max_columns', None)
        with TeamScraper() as scraper:
            matchlog_url = 'https://fbref.com/en/squads/b8fd03ef/2023-2024/Manchester-City-Stats'
            #matchlogs = scraper.scrape_all_matchlogs('https://fbref.com/en/squads/b8fd03ef/2023-2024/Manchester-City-Stats')
            #matchlogs.to_csv('test.csv')
            scraper.set_league_info('premier_league', '2023-2024')
            print(scraper._extract_team_name_from_url(matchlog_url))
            scraper.save_match_data(matchlog_url)

    except Exception as e:
        logging.exception(e)

if __name__ == '__main__':
    main()