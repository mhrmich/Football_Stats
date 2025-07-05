from typing import Dict
import pandas as pd
import numpy as np
import os
import logging
from team_name_mapping_tool import MappingDiscoveryTool


logger = logging.getLogger(__name__)

class MatchCompiler:
    """
    Converts team-level match data into match-level data suitable for ML training.
    Each row represents a complete match with both teams' data
    """

    def __init__(self, project_root: str = None):
        if project_root is None:
            # project_root finds the directory root of the project assuming we're in src/data_processing
            self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.project_root = project_root

        self.cleaned_data_path = os.path.join(self.project_root, 'data', 'cleaned')
        self.processed_data_path = os.path.join(self.project_root, 'data', 'processed')


        self.mapping_tool = MappingDiscoveryTool(project_root=self.project_root)

    def compile_league_season(self, league: str, season: str):
        """
        Compiles all team data for a specific league and season into a single DataFrame with match-level data
        :param league: league name (such as 'premier_league')
        :param season: season name (such as '2024-2025')
        :return: DataFrame with match-level data
        """

        # transforms 'premier_league' into 'premier_league_stats'
        league_directory_name = f"{league}_stats"

        # creates directory like data/cleaned/leagues/premier_league_stats/2024-2025
        league_path = os.path.join(self.cleaned_data_path, 'leagues', league_directory_name, season)

        if not os.path.exists(league_path):
            return pd.DataFrame()

        team_data = self._load_all_teams_data(league_path)
        print(team_data)

        if not team_data:
            return pd.DataFrame()

        matches_df = self._create_match_dataset(team_data, league, season)
        #return matches_df


    def _load_all_teams_data(self, league_path: str) -> Dict[str, pd.DataFrame]:
        """
        Load all team CSV files from a league and season directory.
        :param league_path: Directory path to league CSV files
        :return: Dictionary containing each team's match log data as values; team names as keys
        """

        team_data = {}

        for filename in os.listdir(league_path):
            if filename.endswith('.csv'):
                team_name = filename.replace('.csv', '') # removes the .csv from the file name to obtain team name
                file_path = os.path.join(league_path, filename)

                try:
                    df = pd.read_csv(file_path)
                    domestic_data_df = self._keep_only_domestic_leagues(df)
                    if len(df) > 0:
                        team_data[team_name] = domestic_data_df
                except Exception as e:
                    print(e)
                    continue

        return team_data

    def _keep_only_domestic_leagues(self, team_data):
        domestic_leagues = ['Premier League', 'Serie A', 'Ligue 1', 'La Liga', 'Bundesliga']
        if 'Comp' in team_data.columns:
            domestic_league_df = team_data[team_data['Comp'].isin(domestic_leagues)]
            return domestic_league_df
        else:
            return team_data

    def _create_match_dataset(self, team_data: Dict[str, pd.DataFrame], league, season) -> pd.DataFrame:
        """
        Create match-level dataset from team data
        :param team_data: DataFrame with all team data from a specific season
        :param league: name of the league
        :param season:
        :return:
        """
        # iterates through all teams' DataFrames in from a given league season
        for team_name, team_df in team_data.items():

            # iterates through all rows of a team's DataFrame to assign a unique match ID to each match
            for _, match_row in team_df.iterrows():
                opponent_name = match_row['Opponent']
                standard_opponent_name = self.mapping_tool.team_mappings[opponent_name]
                date = match_row['Date']
                match_id = self._create_match_id(team_name, standard_opponent_name, date)
                print(match_id)

    def _create_match_id(self, team1, team2, date):
        """
        Creates a unique match ID for a match by combining the team names and the match's date
        :param team1: Standardized name for one team
        :param team2: Standardized name of the other team
        :param date: Date of the match in format YYYY-MM-DD
        :return:
        """
        teams = sorted([team1, team2])
        id = f"{date}-{teams[0]}-{teams[1]}"
        return id


def main():
    compiler = MatchCompiler()
    compiler.compile_league_season('premier_league', '2018-2019')


if __name__ == '__main__':
    main()
