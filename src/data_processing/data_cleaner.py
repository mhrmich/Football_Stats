import re
from typing import Dict, Any
import pandas as pd
import numpy as np
import os

class DataCleaner:
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def _default_config(self) -> Dict[str, Any]:
        """Default cleaning configuration"""
        return {
            'drop_duplicate_columns': True,
            'standardize_column_names': True,
            'handle_missing_values': True,
            'convert_data_types': True,
            'remove_invalid_matches': True,
            'min_required_columns': ['Date', 'Opponent', 'Result'],
            'max_missing_threshold': 0.8  # Drop rows with >80% missing values
        }

    def clean_team_file(self, file_path):
        """
        Full process to clean the team CSV file from raw data to processed data
        :param file_path: Path to the CSV file containing a team's data for a single season
        :return: Cleaned DataFrame
        """

        try:
            # Load raw data
            df = pd.read_csv(file_path)
            original_shape = df.shape
            print(f"Original shape: {original_shape}")

            df = self._fix_duplicate_columns(df) # removes all the duplicate columns that arise as a result of scraping
            df = self._remove_unnecessary_rows(df) # removes all the internal header columns
            '''
            NEED TO DO:
            -Transform date and time columns to datetime
            -Remove the penalty shootout data from cup scores
            -
            '''

            return df

        except Exception as e:
            error_msg = f"Error cleaning {file_path}: {str(e)}"
            self.cleaning_stats['errors'].append(error_msg)
            print(error_msg)
            return pd.DataFrame()


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


    def main(self):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sample_file = os.path.join(project_root, 'data', 'raw', 'leagues', 'premier_league_stats', '2023-2024', 'arsenal.csv')
        #sample_file = "test.csv"
        df = pd.read_csv(sample_file)

        #pd.set_option('display.max_columns', None)
        print(f"Shape before: {df.shape}")

        df = self._fix_duplicate_columns(df)
        df = self._remove_unnecessary_rows_cols(df)
        df = self._clean_penalty_shootouts(df)


        df.to_csv('test3.csv', index=False)

        print(f"Shape after: {df.shape}")
        print(df.tail(50))



if __name__ == '__main__':
    try:
        with DataCleaner() as cleaner:
            cleaner.main()
    except Exception as e:
        print(e)