from typing import Dict
import pandas as pd
import numpy as np
import os
import logging
from collections import defaultdict
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

        # Columns to exclude from team-specific prefixing
        self.match_info_columns = {
            'Date', 'Time', 'Comp', 'Round', 'Day', 'Venue', 'Result',
            'GF', 'GA', 'Opponent'
        }

    def compile_league_season(self, league: str, season: str):
        """
        The main function to compile all team data for a league season into a single DataFrame with match-level data
        :param league: league name (such as 'premier_league')
        :param season: season name (such as '2024-2025')
        :return: DataFrame with match-level data
        """

        # transforms 'premier_league' into 'premier_league_stats'
        league_directory_name = f"{league}_stats"

        # creates directory like data/cleaned/leagues/premier_league_stats/2024-2025
        league_path = os.path.join(self.cleaned_data_path, 'leagues', league_directory_name, season)

        # returns empty DataFrame if the league path does not exist
        if not os.path.exists(league_path):
            logger.warning(f"League path does not exist: {league_path}")
            return pd.DataFrame()

        # team_data is a dictionary containing team names as keys and their season match statistics as values
        team_data = self._load_all_teams_data(league_path)


        # returns empty DataFrame if the league season dataframe is empty
        if not team_data:
            logger.warning(f"No data found for {league} in {season}")
            return pd.DataFrame()

        matches_df = self._create_match_dataset(team_data, league, season)
        return matches_df


    def _load_all_teams_data(self, league_path: str) -> Dict[str, pd.DataFrame]:
        """
        Load all team CSV files from a league and season directory.
        :param league_path: Directory path to league CSV files
        :return: Dictionary containing each team's match log data as values; team names as keys
        """

        # team_data is a dictionary where team names are keys and their data for a given season is the corresponding value
        team_data = {}

        # iterates through all the files in the league season directory to find all teams' csv files
        for filename in os.listdir(league_path):
            if filename.endswith('.csv'):
                team_name = filename.replace('.csv', '') # removes the .csv from the file name to obtain team name
                file_path = os.path.join(league_path, filename)

                # filters the DataFrame so only domestic leagues are kept
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
        """Filters DataFrame to only contain matches from domestic league competition"""
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

        # Step 1: Create a match records list that contains every match in a league season from both teams' perspectives
        match_records = self._extract_match_records(team_data)

        # Step 2: Merge team data by match into a singular comprehensive dataset
        matches_df = self._merge_team_data_by_match(match_records)

        #matches_df.to_csv('test_comprehensive.csv', index=False)



        '''
        # iterates through all teams' DataFrames in from a given league season; team_name is key, team_df is value
        for team_name, team_df in team_data.items():

            # iterates through all rows of a team's DataFrame to assign a unique match ID to each match
            for _, match_row in team_df.iterrows():
                opponent_name = match_row['Opponent']
                standard_opponent_name = self.mapping_tool.team_mappings[opponent_name]
                date = match_row['Date']
                match_id = self._create_match_id(team_name, standard_opponent_name, date)
                print(match_id)
        '''

        return matches_df

    def _create_match_id(self, team1, team2, date):
        """
        Creates a unique match ID for a match by combining the team names and the match's date
        Format is in the style of 2018-12-10-everton-watford
        :param team1: Standardized name for one team
        :param team2: Standardized name of the other team
        :param date: Date of the match in format YYYY-MM-DD
        :return:
        """
        teams = sorted([team1, team2])
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        match_id = f"{date_str}-{teams[0]}-{teams[1]}"
        return match_id

    def _extract_match_records(self, team_data: Dict[str, pd.DataFrame]):

        # match_records is a list containing dictionaries with information about each match
        match_records = []

        # iterates across all teams in a league season's dictionary
        for team_name, team_df in team_data.items():

            # iterates through all matches within a specific team's dataframe
            for _, match_row in team_df.iterrows():
                try:
                    opponent_name = match_row['Opponent']
                    if opponent_name in self.mapping_tool.team_mappings:
                        standard_opponent_name = self.mapping_tool.team_mappings[opponent_name]
                    else:
                        # tries to standardize opponent name, e.g. "Manchester United" becomes "manchester_united"
                        standard_opponent_name = opponent_name.lower().replace(' ', '_')

                    date = match_row['Date']
                    match_id = self._create_match_id(team_name, standard_opponent_name, date)

                    # creates a match record for each team's perspective on each match
                    match_record = {
                        'match_id': match_id,
                        'team': team_name,
                        'opponent': standard_opponent_name,
                        'venue': match_row['Venue'],
                        'date': date,
                        'match_data': match_row.to_dict()
                    }

                    match_records.append(match_record)

                except Exception as e:
                    logger.error(f"Failed to extract match records for team {team_name}: {e}")
                    continue

        return match_records

    def _merge_team_data_by_match(self, match_records: list):
        """
        Group match records by their unique id and merge home/away team data
        :param match_records: List where each item is the match record for a single team's perspective on a match as created by _extract_match_records
        :return:
        """
        # creates a defaultdict item
        matches_by_id = defaultdict(list)

        # iterates through all the rows in the match_records list and places them in the defaultdict based on match id
        for record in match_records:
            match_id = record['match_id']
            matches_by_id[match_id].append(record)

        consolidated_matches = []

        # iterates through all the unique match ids in the defaultdict, verifying that there are 2 perspectives on each match
        for match_id, team_records in matches_by_id.items():

            if len(team_records) != 2:
                logger.warning(f"Match {match_id} has {len(team_records)} team records, expected 2")
                continue

            home_record, away_record = self._identify_home_away_teams(team_records)

            if home_record is None or away_record is None:
                logger.warning(f"Could not identify home/away teams for match {match_id}")
                continue

            consolidated_match = self._create_consolidated_match_record(match_id, home_record, away_record)

            consolidated_matches.append(consolidated_match)

        return pd.DataFrame(consolidated_matches)

    def _identify_home_away_teams(self, team_records):
        """
        From the team_records defaultdict item, identify which team record is home and which is away
        :param team_records: defaultdict with match id as keys and two DataFrames as values
        :return:
        """

        home_record = None
        away_record = None

        # returns the home and away record as distinct objects
        for record in team_records:
            if record['venue'] == 'Home':
                home_record = record
            elif record['venue'] == 'Away':
                away_record = record

        return home_record, away_record

    def _create_consolidated_match_record(self, match_id, home_record, away_record):
        """
        Create a single match record with both teams' data
        :param match_id:
        :param home_record:
        :param away_record:
        :return:
        """

        consolidated = {
            'match_id': match_id,
            'date': home_record['date'],
            'competition': home_record['match_data']['Comp'], # the 'Comp' key is stored within the 'match_data' dict
            'home_team': home_record['team'],
            'away_team': away_record['team'],
            'home_goals': home_record['match_data']['GF'],
            'away_goals': away_record['match_data']['GF'],
            'match_result': self._determine_match_result(home_record['match_data']['GF'], away_record['match_data']['GF']),
        }

        home_stats = self._extract_team_statistics(home_record['match_data'], 'home')
        consolidated.update(home_stats)

        away_stats = self._extract_team_statistics(away_record['match_data'], 'away')
        consolidated.update(away_stats)

        return consolidated

    def _determine_match_result(self, home_goals, away_goals):
        """
        Computes a simple categorical score for if the game is a win, draw, or loss for the home team
        :param home_goals: Number of goals scored by the home team
        :param away_goals: Number of goals scored by the away team
        :return: 1 if home win, 0 if draw, -1 if away win
        """
        if home_goals > away_goals:
            return 1
        elif home_goals < away_goals:
            return -1
        else:
            return 0


    def _extract_team_statistics(self, match_data: dict, prefix: str):
        """
        Extracts team statistics from the match record and adds prefix home_ or away_ to feature names
        :param match_data: part of the dataframe containing match data
        :param prefix: "home" or "away" used to prefix feature names
        :return:
        """

        stats = {}

        for col, value in match_data.items():
            if col in self.match_info_columns:
                continue

            if pd.isna(value):
                value = 0

            stats[f"{prefix}_{col}"] = value

        return stats

def main():
    compiler = MatchCompiler()
    compiler.compile_league_season('premier_league', '2018-2019')


if __name__ == '__main__':
    main()
