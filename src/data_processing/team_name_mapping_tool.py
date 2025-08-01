import pandas as pd
import os
from collections import Counter
from typing import Dict
from difflib import SequenceMatcher

class MappingDiscoveryTool:
    """
    Tool to automatically discover missing team name mappings by analyzing files and opponent names.
    """
    def __init__(self, project_root: str = None):
        if project_root is None:
            # gets the project root assuming the file is in src/data_processing
            self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.project_root = project_root

        self.cleaned_data_path = os.path.join(self.project_root, 'data', 'cleaned')

        self.team_mappings = self._create_team_mappings()

    def scan_all_data(self):
        """
        Scan all cleaned datafiles to find team names and opponent names
        :return: Dictionary containing all the names of teams and opponents
        """
        leagues_seasons = []
        team_files_names = set()  # list of the team names found by iterating through the file names
        opponent_names_counter = Counter()
        opponent_names = set()  # list of all the unmapped team names found by iterating the Opponent column


        # iterates through all directories containing cleaned data to find team names
        for root, dirs, files in os.walk(self.cleaned_data_path):
            if files:

                # iterates across all individual teams' data files
                for filename in files:
                    if filename.endswith('.csv'):
                        team_name = filename.replace('.csv', '') # gathers team name from file name
                        team_files_names.add(team_name)

                        # Try to read the file to get opponent names
                        file_path = os.path.join(root, filename)
                        try:

                            # loads a team's season into a DataFrame and reads all the unique opponents
                            df = pd.read_csv(file_path)
                            if 'Opponent' in df.columns:
                                for opponent in df['Opponent'].unique():
                                    if pd.notna(opponent):
                                        opponent_names_counter[opponent] += 1

                                        normalized = self.normalize_team_name(opponent)
                                        if normalized not in team_files_names:
                                            opponent_names.add(opponent)

                        except Exception as e:
                            print(e)


        return {
            'team_files_names': team_files_names, # names of teams from file names
            'opponent_names_counter': opponent_names_counter,
            'opponent_names': opponent_names # names of teams from Opponent column of DataFrames
        }

    def find_mapping_suggestions(self, scan_results: Dict):
        """
        Find suggested mappings for opponent names that are not already mapped in the team_mapping dictionary
        :param scan_results: Result of the scan_all_data() function
        :return: List of (opponent_name, suggested_file, confidence_score) tuples
        """

        suggestions = []
        team_files_names = scan_results['team_files_names'] # team names from files are taken from scan results
        opponent_names = scan_results['opponent_names'] # team names from Opponent columns are taken from scan results
        unmapped_opponents = list(filter(lambda x: x not in self.team_mappings.keys(), opponent_names))
        print(self.team_mappings.keys())
        unmapped_opponents.sort()
        print(f"Unmapped opponents: {unmapped_opponents}")


    def _create_team_mappings(self):
        """
        Creates the dictionary containing team mappings that can be used to compile matches
        """
        mappings = {
            # League club
            "Ajaccio": "ajaccio",
            "Alavés": "alaves",
            "Almería": "almeria",
            "Amiens": "amiens",
            "Angers": "angers",
            "Arminia": "arminia",
            "Arsenal": "arsenal",
            "Aston Villa": "aston_villa",
            "Atalanta": "atalanta",
            "Athletic Club": "athletic_club",
            "Atlético Madrid": "atletico_madrid",
            "Augsburg": "augsburg",
            "Auxerre": "auxerre",
            "Barcelona": "barcelona",
            "Bayern Munich": "bayern_munich",
            "Benevento": "benevento",
            "Betis": "real_betis",
            "Bochum": "bochum",
            "Bologna": "bologna",
            "Bordeaux": "bordeaux",
            "Bournemouth": "bournemouth",
            "Brentford": "brentford",
            "Brescia": "brescia",
            "Brest": "brest",
            "Brighton": "brighton_and_hove_albion", # defaulted to everton, manually changed
            "Burnley": "burnley",
            "Caen": "caen",
            "Cagliari": "cagliari",
            "Cardiff City": "cardiff_city",
            "Celta Vigo": "celta_vigo",
            "Chelsea": "chelsea",
            "Chievo": "chievo",
            "Clermont Foot": "clermont_foot",
            "Cremonese": "cremonese",
            "Crotone": "crotone",
            "Crystal Palace": "crystal_palace",
            "Cádiz": "cadiz",
            "Darmstadt 98": "darmstadt_98",
            "Dijon": "dijon",
            "Dortmund": "dortmund",
            "Düsseldorf": "dusseldorf",
            "Eibar": "eibar",
            "Eint Frankfurt": "eintracht_frankfurt",
            "Elche": "elche",
            "Empoli": "empoli",
            "Espanyol": "espanyol",
            "Everton": "everton",
            "Fiorentina": "fiorentina",
            "Freiburg": "freiburg",
            "Frosinone": "frosinone",
            "Fulham": "fulham",
            "Genoa": "genoa",
            "Getafe": "getafe",
            "Girona": "girona",
            "Gladbach": "monchengladbach",
            "Granada": "granada",
            "Greuther Fürth": "greuther_furth",
            "Guingamp": "guingamp",
            "Hamburger SV": "hamburger_sv",
            "Hannover 96": "hannover_96",
            "Heidenheim": "heidenheim",
            "Hellas Verona": "hellas_verona",
            "Hertha BSC": "hertha_bsc",
            "Hoffenheim": "hoffenheim",
            "Huddersfield": "huddersfield_town",
            "Huesca": "huesca",
            "Inter": 'internazionale', # Manually inputted, could not find from fuzzy matching
            "Juventus": "juventus",
            "Köln": "koln",
            "La Coruña": "deportivo_la_coruna",
            "Las Palmas": "las_palmas",
            "Lazio": "lazio",
            "Le Havre": "le_havre",
            "Lecce": "lecce",
            "Leeds United": "leeds_united",
            "Leganés": "leganes",
            "Leicester City": "leicester_city",
            "Lens": "lens",
            "Levante": "levante",
            "Leverkusen": "bayer_leverkusen",
            "Lille": "lille",
            "Liverpool": "liverpool",
            "Lorient": "lorient",
            "Luton Town": "luton_town",
            "Lyon": "lyon",
            "Mainz 05": "mainz_05",
            "Mallorca": "mallorca",
            "Manchester City": "manchester_city",
            "Manchester Utd": "manchester_united",
            "Marseille": "marseille",
            "Metz": "metz",
            "Milan": "milan",
            "Monaco": "monaco",
            "Montpellier": "montpellier",
            "Monza": "monza",
            "Málaga": "malaga",
            "Nantes": "nantes",
            "Napoli": "napoli",
            "Newcastle Utd": "newcastle_united",
            "Nice": "nice",
            "Norwich City": "norwich_city",
            "Nott'ham Forest": "nottingham_forest",
            "Nîmes": "nimes",
            "Nürnberg": "nurnberg",
            "Osasuna": "osasuna",
            "Paderborn 07": "paderborn_07",
            "Paris S-G": "paris_saint_germain",
            "Parma": "parma",
            "RB Leipzig": "rb_leipzig",
            "Rayo Vallecano": "rayo_vallecano",
            "Real Madrid": "real_madrid",
            "Real Sociedad": "real_sociedad",
            "Reims": "reims",
            "Rennes": "rennes",
            "Roma": "roma",
            "SPAL": "spal",
            "Saint-Étienne": "saint_etienne",
            "Salernitana": "salernitana",
            "Sampdoria": "sampdoria",
            "Sassuolo": "sassuolo",
            "Schalke 04": "schalke_04",
            "Sevilla": "sevilla",
            "Sheffield Utd": "sheffield_united",
            "Southampton": "southampton",
            "Spezia": "spezia",
            "Stoke City": "stoke_city",
            "Strasbourg": "strasbourg",
            "Stuttgart": "stuttgart",
            "Swansea City": "swansea_city",
            "Torino": "torino",
            "Tottenham": "tottenham_hotspur",
            "Toulouse": "toulouse",
            "Troyes": "troyes",
            "Udinese": "udinese",
            "Union Berlin": "union_berlin",
            "Valencia": "valencia",
            "Valladolid": "valladolid",
            "Venezia": "venezia",
            "Villarreal": "villarreal",
            "Watford": "watford",
            "Werder Bremen": "werder_bremen",
            "West Brom": "west_bromwich_albion",
            "West Ham": "west_ham_united",
            "Wolfsburg": "wolfsburg",
            'Wolves': 'wolverhampton_wanderers', # Manually inputted

            # UCL Namings
            "de Dortmund": "dortmund",
            "de Eint Frankfurt": "eintracht_frankfurt",
            "de Freiburg": "freiburg",
            "de Gladbach": "monchengladbach",
            "de Hertha BSC": "hertha_bsc",
            "de Hoffenheim": "hoffenheim",
            "de Köln": "koln",
            "de Leverkusen": "bayer_leverkusen",
            "de RB Leipzig": "rb_leipzig",
            "de Schalke 04": "schalke_04",
            "de Union Berlin": "union_berlin",
            "de Wolfsburg": "wolfsburg",

            "eng Arsenal": "arsenal",
            "eng Aston Villa": "aston_villa",
            "eng Brighton": "brighton_and_hove_albion", # defaulted to everton, manually changed
            "eng Chelsea": "chelsea",
            "eng Everton": "everton",
            "eng Leicester City": "leicester_city",
            "eng Liverpool": "liverpool",
            "eng Manchester City": "manchester_city",
            "eng Manchester Utd": "manchester_united",
            "eng Newcastle Utd": "newcastle_united",
            "eng Tottenham": "tottenham_hotspur",
            "eng West Ham": "west_ham_united",

            "es Athletic Club": "athletic_club",
            "es Atlético Madrid": "atletico_madrid",
            "es Barcelona": "barcelona",
            "es Betis": "real_betis",
            "es Espanyol": "espanyol",
            "es Getafe": "getafe",
            "es Granada": "granada",
            "es Real Madrid": "real_madrid",
            "es Real Sociedad": "real_sociedad",
            "es Sevilla": "sevilla",
            "es Valencia": "valencia",
            "es Villarreal": "villarreal",

            "fr Lens": "lens",
            "fr Lille": "lille",
            "fr Lyon": "lyon",
            "fr Marseille": "marseille",
            "fr Monaco": "monaco",
            "fr Nantes": "nantes",
            "fr Nice": "nice",
            "fr Paris S-G": "paris_saint_germain",
            "fr Rennes": "rennes",
            "fr Saint-Étienne": "saint_etienne",
            "fr Strasbourg": "strasbourg",
            "fr Toulouse": "toulouse",

            "it Atalanta": "atalanta",
            "it Fiorentina": "fiorentina",
            "it Inter": "internazionale",
            "it Juventus": "juventus",
            "it Lazio": "lazio",
            "it Milan": "milan",
            "it Napoli": "napoli",
            "it Roma": "roma",
            "it Torino": "torino",
        }
        return mappings


    def normalize_team_name(self, team_name: str) -> str:
        """
        Normalizes a team name
        :param team_name: Raw team name from data
        :return: Team name for file lookup
        """

        if not team_name:
            return ""

        cleaned = self._clean_team_name(team_name)

    def _clean_team_name(self, team_name: str) -> str:

        # convert to lowercase and strips leading and trailing whitespace
        cleaned = team_name.lower().strip()
        cleaned = ' '.join(cleaned.split()) # removes any accidental double space or additional whitespace

        # Deal with common forms of punctuation
        cleaned = cleaned.replace("'", "") # Remove apostrophes
        cleaned = cleaned.replace("-", " ") # Remove hyphens
        cleaned = cleaned.replace(",", "")
        cleaned = cleaned.replace(".", "")

        return cleaned


    def _fuzzy_match(self, team_name: str, team_options):
        best_score = 0
        best_match = None
        standard_team_name = self._clean_team_name(team_name)

        for match_team in team_options:
            standard_match_team_name = self._clean_team_name(match_team)

            score = SequenceMatcher(None, standard_team_name, standard_match_team_name).ratio()
            if score > best_score:
                best_match = match_team
                best_score = score

        return {team_name: best_match}


def main():
    mapper = MappingDiscoveryTool()
    scan_results = mapper.scan_all_data()
    file_name_teams = scan_results['team_files_names']
    file_name_opponents = scan_results['opponent_names']

    '''
    match_dict = {}
    for team in file_name_opponents:
        match_dict = {
            **match_dict,
            **mapper._fuzzy_match(team, file_name_teams)
        }

    for match_team, team_name in sorted(match_dict.items()):
        print(f'"{match_team}": "{team_name}"', end=',\n')
    '''
    mapper.find_mapping_suggestions(scan_results)

if __name__ == '__main__':
    main()


