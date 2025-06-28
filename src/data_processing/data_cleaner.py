import re
from typing import Dict, Any
import pandas as pd
import numpy as np
import os

class DataCleaner:
    def __init__(self, config=None):
        self.config = self._default_config()

        if config:
            self.config.update(config)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def _default_config(self) -> Dict[str, Any]:
        """Default cleaning configuration"""
        return {
            # Basic cleaning options
            'drop_duplicate_columns': True,
            'standardize_column_names': True,
            'handle_missing_values': True,
            'convert_data_types': True,
            'remove_invalid_matches': True,
            'min_required_columns': ['Date', 'Opponent', 'Result'],
            'max_missing_threshold': 0.8,  # Drop rows with >80% missing values

            # Competition filtering options
            'european_competitions': True,
            'domestic_cups': False,

        }

    def clean_team_file(self, df):
        """
        Full process to clean the team CSV file from raw data to processed data
        :param file_path: Path to the CSV file containing a team's data for a single season
        :return: Cleaned DataFrame
        """

        try:
            print(f"Shape before: {df.shape}")

            # Basic cleaning configurations
            df = self._fix_duplicate_columns(df) # removes all the duplicate columns that arise as a result of scraping
            df = self._standardize_column_names(df) # "For Brighton & Hove Albion_Date" becomes "Date" for example
            df = self._remove_unnecessary_rows_cols(df) # removes all the internal header columns
            df = self._handle_missing_values(df)
            df = self._clean_penalty_shootouts(df) # Scores transcribed as 1 (4) just becomes 1 to represent normal play
            df = self._convert_data_types(df) # Converts data into datetime or numeric format when appropriate

            df = self._filter_competitions(df) # Filters which competitions to keep based on DataCleaner configurations

            print(f"Shape after: {df.shape}")

            return df

        except Exception as e:
            error_msg = f"Error cleaning file: {str(e)}"
            print(error_msg)
            return pd.DataFrame()


    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove unwanted team prefixes from column names (i.e. "For Brighton & Hove Albion_Date" to become just "Date")
        :param df: Raw DataFrame which might contain uncleaned column names
        :return: Standardized column names
        """
        original_cols = df.columns.tolist()
        new_cols = []

        for col in df.columns:
            if col.startswith('For ') and "_" in col:
                standardized_name = col.split("_")[-1] # selects only the section after the underscore if it exists
                new_cols.append(standardized_name)
            else:
                new_cols.append(col)

        df.columns = new_cols
        return df


    def _fix_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes all duplicate columns from a DataFrame by removing all columns ending in .1, .2, etc. (i.e. Comp.1)
        :param df: DataFrame to be cleaned
        :return: DataFrame with duplicate columns removed
        """
        duplicate_pattern = r'\.(\d+)$' # duplicate columns end with .1, .2, etc.: \. is a period, (\d+) is one or more digits
        cols_to_keep = []
        for col in df.columns:

            # only appends columns that do not contain the duplicate pattern
            if not re.search(duplicate_pattern, col):
                cols_to_keep.append(col)

        return df[cols_to_keep]



    def _remove_unnecessary_rows_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes rows containing internal headers and removes the final row, which contains column totals, from dataframe
        :param df:
        :return:
        """


        internal_header_rows = (
                df['Date'].astype(str).str.contains("Date") |  # the reiterated header column
                df['Date'].astype(str).str.startswith("For ") # column above internal header ("for Arsenal")
        )

        df = df[~internal_header_rows].iloc[:-1] # drops the internal header rows and drops the last row
        df = df.drop(columns=['Match Report']) # drops the unnecessary "Match Report" column

        return df


    def _clean_penalty_shootouts(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ['GF', 'GA']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.extract(r'(\d+)')[0]

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:

        full_data_leagues = ['Premier League', 'Bundesliga', 'Ligue 1', 'La Liga', 'Serie A'] # domestic leagues with full data
        european_competitions = ['Europa Lg', 'Champions Lg', 'Conf Lg'] # international European competitions with full data
        incomplete_data_competitions = ['EFL Cup', 'FA Cup', 'DFB-Pokal', 'Coupe de France', 'Coupe de la Ligue',
                                        'Copa del Rey', 'Coppa Italia'] # domestic cups with incomplete data

        # unpacks all the competitions into a dictionary and maps them to the "tier" (1 for domestic league, 2 for
        # European competition, 3 for domestic cup; competitions in 3 also have incomplete data
        df['competition_tier'] = df['Comp'].map({
            **{comp: 1 for comp in full_data_leagues},
            **{comp: 2 for comp in european_competitions},
            **{comp: 3 for comp in incomplete_data_competitions}
        }).fillna(3) # any match that doesn't even list a competition is filled with 3 (incomplete data tier)

        return df

    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts data in each column to the proper data type
        :param df: DataFrame with each column as type "object"
        :return: DataFrame with data returned as int64 or float64
        """


        for col in df.columns:
            if col not in ['Date', 'Time', 'Comp', 'Round', 'Day', 'Venue', 'Result', 'Opponent']:
                df[col] = pd.to_numeric(df[col], errors='coerce') # transforms each numeric column into numeric format
            else:
                df[col] = df[col].astype('category') # transforms remaining columns into categorical data
        df['Date'] = pd.to_datetime(df['Date'])  # Transforms the 'Date' column into datetime format

        return df

    def _filter_competitions(self, df: pd.DataFrame) -> pd.DataFrame:

        comp_tiers = [1]
        if self.config['european_competitions']:
            comp_tiers.append(2)
        if self.config['domestic_cups']:
            comp_tiers.append(3)

        df = df[df['competition_tier'].isin(comp_tiers)]
        return df


    def clean_all_teams(self):
        """
        Iterates through and cleans all the raw CSV files
        """
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        raw_data_path = os.path.join(project_root, 'data', 'raw')

        for root, dirs, files in os.walk(raw_data_path): # iterates through all recursive directory paths
            if files:
                for filename in files:
                    try:
                        file_path = os.path.join(root, filename) # creates the path of the raw data file
                        df = pd.read_csv(file_path) # opens the CSV file as a DataFrame
                        cleaned_df = self.clean_team_file(df) # cleans the DataFrame with clean_team_file()

                        league = root.split('\\')[-2]
                        season = root.split('\\')[-1]

                        output_directory = os.path.join(
                            project_root,
                            'data',
                            'cleaned',
                            'leagues',
                            league,
                            season
                        )

                        # creates the directory if it does not already exist
                        os.makedirs(output_directory, exist_ok=True)

                        # joins the directory with teamname.csv
                        output_filepath = os.path.join(output_directory, filename)

                        cleaned_df.to_csv(output_filepath, index=False)

                    except Exception as e:
                        print(f"Error cleaning {file_path}: {str(e)}")




    def main(self):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sample_file = os.path.join(project_root, 'data', 'raw', 'leagues', 'premier_league_stats', '2023-2024', 'arsenal.csv')
        sample_file = os.path.join(project_root, 'data', 'raw', 'leagues', 'premier_league_stats', '2023-2024', 'brighton_and_hove_albion.csv')
        #sample_file = "test.csv"
        df = pd.read_csv(sample_file)

        #df = self.clean_team_file(sample_file)
        self.clean_all_teams()


        #df.to_csv('test.csv', index=False)



if __name__ == '__main__':
    try:
        with DataCleaner() as cleaner:
            cleaner.main()
    except Exception as e:
        print(e)