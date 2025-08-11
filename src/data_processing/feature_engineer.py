import os
from typing import Dict

import pandas as pd
import numpy as np
import logging

from pandas.core.interchange import column

logger = logging.getLogger(__name__)

class MasterFeatureEngineer:
    """
    Creates rolling form features and predictive features from the master dataset
    """

    def __init__(self, project_root: str = None):

        # directory path leading to the entire project
        if project_root is None:
            self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.project_root = project_root

        # directory path where processed data (on which we will engineer features) can be found, and where we will
        # deposit output
        self.processed_data_path = os.path.join(self.project_root, 'data', 'processed','master_dataset.csv')
        self.final_data_path = os.path.join(self.project_root, 'data', 'final','final_dataset.csv')

        # stats that we want to create rolling features for
        self.key_stats = [
            # Shooting stats
            'Standard_Gls','Standard_Sh', 'Standard_SoT', 'Standard_SoT%', 'Standard_Dist', # shot distance
            'Expected_xG', 'Expected_npxG',

            # GK Passing stats
            'Launched_Cmp', 'Launched_Att',

            # Passing stats, goal and shot creation
            'Total_Cmp', 'Total_Att', 'Total_Cmp%', 'Total_TotDist', 'Total_PrgDist', 'Short_Cmp', 'Short_Att',
            'Short_Cmp%', 'Medium_Cmp', 'Medium_Att', 'Medium_Cmp%', 'Long_Cmp', 'Long_Att', 'Long_Cmp%', 'xAG', 'xA',
            'KP', '1/3', 'PPA', 'CrsPA', 'PrgP', 'SCA Types_SCA', 'GCA',

            # Defensive actions
            'Tckles_Tkl', 'Tackles_TklW', 'Tackles_Def 3rd', 'Tackles_Mid 3rd', 'Challenges_Tkl', 'Challenges_Att',
            'Blocks_ Blocks', 'Int', 'Clr',

            # Possession
            'Poss', 'Touches_Def Pen', 'Touches_Def 3rd', 'Touches_Mid 3rd', 'Touches_Att 3rd', 'Touches_Att Pen',
            'Take-Ons_Att', 'Take-Ons_Succ', 'Take-Ons_Succ%', 'Take-Ons_Tkld%', 'Carries_Carries', 'Carries_TotDist',
            'Carries_PrgDist', 'Carries_PrgC',

            # Miscellaneous
            'CrdY', 'CrdR', 'Fls',
        ]

        self.rolling_windows = [3, 5, 10]


    def engineer_all_features(self, master_df_path: str) -> pd.DataFrame:
        """
        Main method to engineer all predictive features from the master dataset
        :param master_df_path: Filepath to the master dataset CSV
        :return: DataFrame with all engineered features ready for ML
        """

        logger.info('Loading master dataset into Pandas')
        df = pd.read_csv(master_df_path)

        logger.info(f"Loaded {len(df)} matches from {df['date'].min()} to {df['date'].max()}")

        #Step 1: Prepare data
        df = self._prepare_data(df)

        if df is None:
            print('df is None')

        #Step 2: Create rolling features for each team's data
        df = self._create_rolling_features(df)

        #Step 3: Remove non-rolling features
        df = self._remove_not_rolling_features(df)


        #Step 5: Since we are building a model to predict matches based on historical form data, we remove
        # the first few matches, in which there is no form data
        df = self._filter_by_match_experience(df, 5)

        print(df.head())

        return df

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The first step of feature engineering, so far this only transforms datetime data type and sorts rows
        :param df: Inputted original DataFrame
        :return: DataFrame with date data in proper data type and sorted by date
        """

        # converts the date column into datetime format, sorts the dataframe again, and drops the index
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date').reset_index(drop=True)
        if 'season' not in df.columns:
            df['season'] = df['date'].apply(self._extract_season)

        logger.info(f"Step 1 complete: Prepared {len(df)} matches from {df['date'].min()} to {df['date'].max()}")
        return df

    def _extract_season(self, date) -> str:
        if date.month >= 8:
            return f"{date.year}-{str(date.year+1)[2:]}"
        else:
            return f"{date.year-1}-{str(date.year)[2:]}"

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The second step of feature engineering, this is the main process for creating rolling form-based features
        :param df:
        :return:
        """

        # Creates a set containing the names of all times from the home and away team columns
        all_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())

        # team_histories is a dictionary; each key is the name of a team, each value
        team_histories = self._build_team_histories(df, all_teams)

        rolling_features_list = []

        for idx, match in df.iterrows(): # idx is index

            # Progress tracker
            if idx % 1000 == 0:
                logger.info(f"Processing rolling features: {idx}/{len(df)} matches")


            match_date = match['date']
            home_team = match['home_team']
            away_team = match['away_team']

            # Obtains the form features for both the home team and the away teams and
            home_features = self._get_team_rolling_features(team_histories[home_team], match_date, 'home')
            away_features = self._get_team_rolling_features(team_histories[away_team], match_date, 'away')

            # For the given match, combine the home and away teams' rolling features into a single dictionary, which will
            # become a single row in the master DataFrame
            match_features = {**home_features, **away_features}
            rolling_features_list.append(match_features)

        rolling_df = pd.DataFrame(rolling_features_list)
        df = pd.concat([df.reset_index(drop=True), rolling_df.reset_index(drop=True)], axis=1)

        return df

    def _build_team_histories(self, df: pd.DataFrame, all_teams: set) -> Dict[str, pd.DataFrame]:
        """
        Build chronological match history for each team across all leagues. We have verified that this function works
        :param df: DataFrame containing match data for all matches across all leagues
        :param all_teams: Set of unique team names
        :return: Dictionary matching each team to a DataFrame containing its chronological match history in a DataFrame
        """

        # team_histories is a dictionary containing the past match data of each team
        team_histories = {}

        # Iterates across all teams and divides their matches into home and away to create separate form records
        for team in all_teams:
            logger.info(f'Constructing match histories for {team}')
            home_matches = df[df['home_team'] == team].copy()
            away_matches = df[df['away_team'] == team].copy()

            # a list containing the standardized match records for all the matches a team plays in a season
            team_history = []

            # creates match records for all a team's home matches
            for _, match in home_matches.iterrows():
                record = self._create_team_match_record(match, 'home')
                team_history.append(record)

            # creates match records for all a team's away matches
            for _, match in away_matches.iterrows():
                record = self._create_team_match_record(match, 'away')
                team_history.append(record)

            # Since team_history is a list of dictionaries, we can transform it into a DataFrame
            if team_history:
                team_df = pd.DataFrame(team_history)
                team_df = team_df.sort_values('date').reset_index(drop=True)

                # each DataFrame containing a team's standardized matches can be added to the master team_histories dict
                team_histories[team] = team_df


        return team_histories

    def _create_team_match_record(self, match, venue_type):
        """
        Creates a standardized team match record
        :param match: Row of the DataFrame containing match data
        :param venue_type: 'home' or 'away'
        :return:
        """

        # team_stats is a dictionary containing each statistic column as a key and its corresponding value for a match
        team_stats = {}

        # creates a standardized match record if the match was a home match
        if venue_type == 'home':

            # set of standard match data to record
            goals_for = match['home_goals']
            goals_against = match['away_goals']
            opponent = match['away_team']
            venue = 'Home'

            # Removes the home_ prefix from each stat column
            for col in match.index:
                if col.startswith('home_') and col != 'home_team' and col != 'home_goals':
                    stat_name = col.replace('home_', '')
                    team_stats[stat_name] = match[col]

        # process to create a standardized match record if the match was played away
        else:

            # set of standard match data to record
            goals_for = match['away_goals']
            goals_against = match['home_goals']
            opponent = match['home_team']
            venue = 'Away'

            for col in match.index:
                if col.startswith('away_') and col != 'away_team' and col != 'away_goals':
                    stat_name = col.replace('away_', '')
                    team_stats[stat_name] = match[col]

        # codifies the result of the match as a number: -1 for loss, 0 for draw, 1 for win
        if goals_for > goals_against:
            result = 1 # 1 for a win
        elif goals_for < goals_against:
            result = -1 # -1 for a loss
        else:
            result = 0 # 0 for a draw


        # record combines a team's statistics with objective match data to create a full match record
        record = {
            'date': match['date'],
            'league': match['league'],
            'opponent': opponent,
            'venue': venue,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'result': result,
            **team_stats
        }

        return record

    def _get_team_rolling_features(self, team_history, match_date, prefix):
        """
        Calculate rolling features for a team before a specific match date
        :param team_history: DataFrame containing all a team's matches sorted by date
        :param match_date: datetime value containing the date of a match
        :param prefix: 'home' or 'away'
        :return:
        """

        # if there is no team_history, which should not happen
        if team_history.empty:
            return self._get_empty_rolling_features(prefix)

        # filters the DataFrame to only include matches before the given parameter date
        prior_matches = team_history[team_history['date'] < match_date]

        #
        if prior_matches.empty:
            return self._get_empty_rolling_features(prefix)

        # Dictionary containing the names of rolling features as keys and the corresponding feature values for a match
        features = {}

        # creates rolling features for 3, 5, or 10 matches; if recent_matches is shorter, no error should arise
        for window in self.rolling_windows:
            recent_matches = prior_matches.tail(window) # 3, 5, or 10 most recent matches

            # Skips if there are no prior/recent matches
            if len(recent_matches) == 0:
                continue

            # Basic form metrics, averages from the 3, 5, or 10 match windows
            features[f'{prefix}_points_per_game_L{window}'] = self._calculate_points_per_game(recent_matches)
            features[f'{prefix}_goals_scored_avg_L{window}'] = recent_matches['goals_for'].mean()
            features[f'{prefix}_goals_conceded_avg_L{window}'] = recent_matches['goals_against'].mean()
            features[f'{prefix}_goal_difference_avg_L{window}'] = (
                    recent_matches['goals_for'].mean() - recent_matches['goals_against'].mean()
            )

            # Win/Draw/Loss percentages
            features[f'{prefix}_win_pct_L{window}'] = (recent_matches['result'] == 1).mean()
            features[f'{prefix}_draw_pct_L{window}'] = (recent_matches['result'] == 0).mean()
            features[f'{prefix}_loss_pct_L{window}'] = (recent_matches['result'] == -1).mean()

            features[f'{prefix}_clean_sheet_pct_L{window}'] = (recent_matches['goals_against'] == 0).mean()
            features[f'{prefix}_failed_to_score_pct_L{window}'] = (recent_matches['goals_for'] == 0).mean()

            # Fills in the advanced statistics if available
            for stat in self.key_stats:
                if stat in recent_matches.columns:
                    features[f'{prefix}_{stat}_avg_L{window}'] = recent_matches[stat].mean()

        features[f'{prefix}_total_matches'] = len(prior_matches)
        features[f'{prefix}_overall_ppg'] = self._calculate_points_per_game(prior_matches)
        features[f'{prefix}_overall_goal_ratio'] = (
            prior_matches['goals_for'].mean() / max(prior_matches['goals_against'].mean(), 0.1)
        )

        return features

    def _calculate_points_per_game(self, matches: pd.DataFrame) -> float:
        """
        Calculates a team's average points per game over a specific period of matches
        :param matches: DataFrame containing a set of matches over which to calculate points per game
        :return: Average points per game over the matches
        """

        if matches.empty:
            return 0
        points = matches['result'].apply(lambda x: 3 if x==1 else (1 if x==0 else 0)) # 3 for win, 1 for draw, 0 for loss
        return points.mean()




    def _get_empty_rolling_features(self, prefix):
        """
        Return default features when no prior matches exist
        :param prefix: 'home' or 'away'
        :return:
        """
        features = {}

        for window in self.rolling_windows:
            features[f'{prefix}_points_per_game_L{window}'] = 0
            features[f'{prefix}_goals_scored_avg_L{window}'] = 0
            features[f'{prefix}_goals_conceded_avg_L{window}'] = 0
            features[f'{prefix}_goal_difference_avg_L{window}'] = 0
            features[f'{prefix}_win_pct_L{window}'] = 0
            features[f'{prefix}_draw_pct_L{window}'] = 0
            features[f'{prefix}_loss_pct_L{window}'] = 0
            features[f'{prefix}_clean_sheet_pct_L{window}'] = 0
            features[f'{prefix}_failed_to_score_pct_L{window}'] = 0

            for stat in self.key_stats:
                features[f'{prefix}_{stat}_avg_L{window}'] = 0

        features[f'{prefix}_total_matches'] = 0
        features[f'{prefix}_overall_ppg'] = 0
        features[f'{prefix}_overall_goal_ratio'] = 1.0

        return features

    def _remove_not_rolling_features(self, df):


        essential_columns = [

            # date, competition, and team name are excluded for now; might re-implement later
            #'competition', 'home_team', 'away_team',
            'date', 'home_goals', 'away_goals', 'match_result',
        ]

        rolling_patters = ['_L3', '_L5', '_L10', '_total_matches', '_overall_ppg', '_overall_goal_ratio']
        rolling_columns = []

        for col in df.columns:
            if any(pattern in col for pattern in rolling_patters):
                rolling_columns.append(col)

        # These are the essential + rolling columns that are necessary for the ML algorithm
        cols_to_keep = essential_columns + rolling_columns

        # finds the existing columns to keep that actually exist
        existing_cols_to_keep = [col for col in cols_to_keep if col in df.columns]

        df_clean = df[existing_cols_to_keep].copy()

        return df_clean

    def _filter_by_match_experience(self, df, min_matches = 5):
        """
        Step 5 of the preprocessing/feature engineering pipeline, we remove the earliest matches which do not have
        adequate rolling data.
        :param df: DataFrame containing all match data
        :param min_matches: minimum number of matches that both teams must have played to keep the row (default is 5)
        :return: Filtered DataFrame, removing the earliest matches with inaccurate rolling data
        """

        # this is the filter condition
        sufficient_history = (df['home_total_matches'] >= min_matches) & (df['away_total_matches'] >= min_matches)

        df_filtered = df[sufficient_history].copy()

        return df_filtered

def main():

    pd.set_option('display.max_rows', None)
    engineer = MasterFeatureEngineer()
    df = engineer.engineer_all_features(engineer.processed_data_path)
    df.to_csv(engineer.final_data_path)



    test_df = pd.read_csv(engineer.final_data_path)
    test_df = test_df[test_df['date'] < '2019-01-01']
    test_df.to_csv('test.csv')
    print(test_df.dtypes)



if __name__ == '__main__':
    main()
