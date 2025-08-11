import os
import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier


class FootballMatchPredictor:
    """
    Complete ML pipeline for football match prediction
    """

    def __init__(self, project_root = None):
        if project_root is None:
            self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.project_root = project_root

        # Directory paths to data, the saved models, and the results
        self.data_path = os.path.join(self.project_root, 'data', 'final')
        self.models_path = os.path.join(self.project_root, 'models')
        self.results_path = os.path.join(self.project_root, 'results')

        # Creates the models and results directories if they do not already exist
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)

        self.scaler = StandardScaler()
        self.feature_selector = None
        self.models = {}
        self.best_model = None
        self.feature_names = []

    def load_and_prepare_data(self, data_filename='final_dataset.csv'):
        data_path = os.path.join(self.data_path, data_filename) # the specific path leading to the data file
        df = pd.read_csv(data_path)

        # drops the index column that holds no data
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['season'] = df['date'].apply(self._to_season)

        X, y, feature_names = self._prepare_features_target(df) # Obtains feature and target variables

        # Creates a temporal (chronological split)
        X_train, X_test, y_train, y_test = self._season_split(df, X, y)


        # fills in any missing values with medians
        train_medians = X_train.median(numeric_only=True)
        X_train = X_train.fillna(train_medians)
        X_test = X_test.fillna(train_medians)

        #Scale features
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns, index=X_test.index)


        # dictionary containing summary of data that we will return
        data_dict = {
            'X_train': X_train,
            'X_test': X_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'original_df': df,
            'data_info': {
                'total_matches': len(df),
                'train_matches': len(y_train),
                'test_matches': len(y_test)
            }
        }

        print('Prepared dat shit')

        return data_dict

    def train_evaluate_models(self, data_dict):
        # Load data from the data dictionary parameter
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        X_train_scaled = data_dict['X_train_scaled']
        X_test_scaled = data_dict['X_test_scaled']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']

        # Model 1: Random Forest Classifier
        hgb = self.train_histgb(X_train, y_train)
        y_pred = hgb.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print(f"Score: {score}")
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        disp.plot()
        plt.show()

        # Model 2:

    def train_random_forest(self, X_train, y_train, draw_weight: float = 1.0):


        rf = RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
        )
        rf.fit(X_train, y_train)
        return rf

    def train_histgb(self, X_train, y_train, draw_weight: float = 1.0):

        hgb = HistGradientBoostingClassifier(
            learning_rate = 0.08,
            max_leaf_nodes=63,
            min_samples_leaf=20,
            max_depth=None,
            random_state=42
        )

        hgb.fit(X_train, y_train)
        return hgb


    # ------------------------- Helpers -------------------------
    def _prepare_features_target(self, df):
        essential_cols = ['match_id', 'date', 'competition', 'home_team', 'away_team', 'home_goals', 'away_goals']

        # essentially all the rolling feature columns
        feature_cols = [col for col in df.columns if col not in essential_cols + ['match_result']]

        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df['match_result']

        return X, y, list(X.columns)

    def _to_season(self, date: pd.Timestamp):
        if pd.isna(date):
            return 'unknown'
        if date.month >= 7:
            return f"{date.year}/{date.year+1}"
        else:
            return f"{date.year-1}/{date.year}"


    def _temporal_split(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series, train_portion=0.8):
        """
        Chronological train-test split based on a proportion of data to split; we are using _season_split instead though
        :param X: Feature matrix
        :param y: Target array
        :param dates: List of match dates
        :param train_portion: Portion of train data; set default so we have an 80-20 train-test split
        :return:
        """

        # Sort by date; should not mix up order
        sorted_index = dates.argsort() # returns the indices
        X_sorted = X.iloc[sorted_index]
        y_sorted = y.iloc[sorted_index]
        dates_sorted = dates.iloc[sorted_index]


        split_idx = int(train_portion * len(X_sorted))

        X_train = X_sorted[:split_idx]
        y_train = y_sorted[:split_idx]
        X_test = X_sorted[split_idx:]
        y_test = y_sorted[split_idx:]

        return X_train, X_test, y_train, y_test

    def _season_split(self, df, X, y):
        """

        :param df:
        :param X:
        :param y:
        :return:
        """
        # Training data before most recent season; test on recent season
        train_mask = df['season'].isin(['2017/2018', '2018/2019', '2019/2020', '2020/2021', '2021/2022', '2022/2023'])
        test_mask = df['season'].isin(['2023/2024'])

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            #Fallback: split by date threshold at July 1, 2023
            threshold = pd.Timestamp('2023-07-01')
            train_mask = df['date'] < threshold
            test_mask = df['date'] >= threshold

        X_train, X_test = X.loc[train_mask].copy(), X.loc[test_mask].copy()
        y_train, y_test = y.loc[train_mask].copy(), y.loc[test_mask].copy()
        return X_train, X_test, y_train, y_test

    def build_model(self):
        model = LogisticRegression()


def main():
    predictor = FootballMatchPredictor()
    data_path = os.path.join(predictor.data_path, 'final_dataset.csv')

    pd.set_option('display.max_columns', None)



    # loads and prepares the data
    data_dict = predictor.load_and_prepare_data()
    predictor.train_evaluate_models(data_dict)

    '''
    #model = LogisticRegression(random_state=42, max_iter=2000)
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    #model = GradientBoostingClassifier(n_estimators=200, max_depth=10)
    #model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(y_pred)
    y_pred_proba = model.predict_proba(X_test)

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(cm)
    print(f"Accuracy: {accuracy}")
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()
    '''



if __name__ == '__main__':
    main()

