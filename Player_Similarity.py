import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('Stats_Data/top_5_leagues_stats_2023_24.csv')
for col in df.columns:
    print(col, end=", ")


df.fillna(0, inplace=True)

performance_metrics = ['Playing Time Min', 'Performance Gls', 'Performance Ast', 'Performance PK', 'Performance PKatt',
                         'Performance CrdY', 'Performance CrdR', 'Expected xG', 'Expected npxG', 'Expected xAG',
                         'Progression PrgC', 'Progression PrgP', 'Progression PrgR', 'Standard Sh', 'Standard SoT',
                         'Standard Dist', 'Standard FK', 'Total Cmp', 'Total Att', 'Total TotDist', 'Total PrgDist',
                         'Short Cmp', 'Short Att', 'Medium Cmp', 'Medium Att', 'Long Cmp', 'Long Att', 'KP', '1/3',
                         'PPA', 'CrsPA', 'SCA SCA', 'Tackles Tkl', 'Tackles TklW', 'Tackles Def 3rd', 'Tackles Mid 3rd',
                         'Tackles Att 3rd', 'Challenges Tkl', 'Challenges Att', 'Blocks Blocks', 'Int', 'Clr', 'Err',
                       'Touches Def Pen', 'Touches Def 3rd', 'Touches Mid 3rd', 'Touches Att 3rd', 'Touches Att Pen',
                       'Take-Ons Att', 'Take-Ons Succ', 'Take-Ons Tkld', 'Carries Carries', 'Carries TotDist',
                       'Carries PrgDist', 'Carries PrgC', 'Carries 1/3', 'Carries CPA', 'Carries Mis', 'Carries Dis',
                       'Receiving Rec', 'Receiving PrgR']

df_metrics = df[['Player', 'Pos', 'Squad'] + performance_metrics]
df_metrics = df_metrics[df_metrics['Pos'] != 'GK'].copy()
df_metrics.sort_values(by='Performance Gls', ascending=False)

# Scale the data
X = df_metrics[performance_metrics]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=performance_metrics)

kmeans = KMeans(n_clusters=20, random_state=42)
df_metrics['Cluster'] = kmeans.fit_predict(X=X_scaled_df)

clusters = df_metrics.sort_values(by='Cluster', ascending=False)

clusters_grouped = clusters.groupby('Cluster')

'''
for cluster_id, group in clusters_grouped:
    sorted_group = group.sort_values(by='Playing Time Min', ascending=False)
    top_10 = sorted_group.head(10)
    print(f'\nCluster {cluster_id}:')
    print(top_10[['Player', 'Pos', 'Squad', 'Playing Time Min', 'Performance Gls']].to_string(index=False))
'''

def find_similar_players_nearest_neighbors(player_name, df, model, scaler):
    player_stats = df[df['Player'] == player_name][performance_metrics]
    player_scaled = scaler.transform(player_stats)
    distances, indices = model.kneighbors(player_scaled)
    similar_players = df.iloc[indices[0]]

    return similar_players[['Player', 'Pos', 'Squad', 'Playing Time Min', 'Performance Gls']]


def find_similar_player_cosine_similarity(player_name, df):
    similarity_matrix = cosine_similarity(df)
    similarity_df = pd.DataFrame(similarity_matrix, index=df_metrics['Player'], columns=df_metrics['Player'])
    similar_players = similarity_df[player_name].sort_values(ascending=False)
    return similar_players


def sort_table_by_stat(df, stat, ascending=False):
    if stat not in df.columns:
        raise ValueError(f'Stat{stat} not found in columns')
    sorted_df = df.sort_values(by=stat, ascending=ascending)
    return sorted_df

#nn_model = NearestNeighbors(n_neighbors=10, metric='euclidean')
#nn_model.fit(X_scaled_df)
#similar_players = find_similar_players_nearest_neighbors('Cole Palmer', df_metrics, nn_model, scaler)
#print(similar_players)


similarity = find_similar_player_cosine_similarity('Ante Budimir', X_scaled_df)
print(similarity.head(15))

sorted_goals = sort_table_by_stat(df, 'Performance Gls')
print(sorted_goals[['Player', 'Pos', 'Squad', 'Performance Gls']].head(15))
