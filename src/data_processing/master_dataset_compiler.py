import pandas as pd
import numpy as np
import os
import logging
from match_compiler import MatchCompiler
from team_name_mapping_tool import MappingDiscoveryTool

logger = logging.getLogger(__name__)

class MasterDatasetCompiler:
    """
    Uses MatchCompiler to combine all leagues and seasons into a single master dataset for ML training
    """

    def __init__(self, project_root: str = None):
        if project_root is None:
            self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.project_root = project_root

        self.processed_data_path = os.path.join(self.project_root, 'data', 'processed')
        self.final_data_path = os.path.join(self.project_root, 'data', 'final')

        self.match_compiler = MatchCompiler(project_root = self.project_root)
        self.mapping_tool = MappingDiscoveryTool(project_root=self.project_root)

        self.leagues = ['premier_league', 'la_liga', 'ligue_1', 'serie_a']
        self.seasons = ['2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024']

        os.makedirs(self.processed_data_path, exist_ok=True)
        os.makedirs(self.final_data_path, exist_ok=True)

    def compile_all_data(self):
        """Main method to compile all league data into a single master dataset for ML training"""

        all_matches = []

        for league in self.leagues:
            for season in self.seasons:
                logger.info(f"Compiling {league} {season}")

                try:

                    season_df = self.match_compiler.compile_league_season(league=league, season=season)

                    if not season_df.empty:
                        season_df['league'] = league # create a column for league

                        all_matches.append(season_df)

                except Exception as e:
                    logger.error(e)

        print(all_matches)
        master_df = pd.concat(all_matches, ignore_index=True)

        return master_df


def main():
    master_compiler = MasterDatasetCompiler()
    master_df = master_compiler.compile_all_data()
    master_filepath = os.path.join(master_compiler.processed_data_path, 'master_dataset.csv')
    master_df.to_csv(master_filepath)

if __name__ == '__main__':
    main()
