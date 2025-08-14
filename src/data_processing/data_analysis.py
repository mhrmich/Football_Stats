import pandas as pd
import os

def main():

    # Load data into Pandas
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(project_root, 'data', 'final', 'final_dataset.csv')
    df = pd.read_csv(file_path)
    pd.set_option('display.max_rows', None)
    print(df.head())

    for column in df.columns:
        print(column)

    # Check for missing values
    missing_stats = df.isnull().sum()
    print(missing_stats)



if __name__ == '__main__':
    main()