import pandas as pd
from sklearn.model_selection import train_test_split

def get_team_matches(team_name):
    # given the name of a team,
    df = pd.read_csv('Scraping_Help/merged_df.csv')

    # creates a DataFrame containing matches for a given team
    team_matches = df.loc[(df['Home_Team'] == team_name) | (df['Away_Team'] == team_name)]
    return team_matches

def sort_home_away(team_matches):

    team_opponent_df = pd.DataFrame()
    result_mapping = {'W': 1, 'D': 0, 'L': -1}
    


    return team_matches




def predict(team_name):
    pass




def main():
    man_city = get_team_matches('Manchester City')
    print(sort_home_away(man_city))

if __name__ == '__main__':
    main()