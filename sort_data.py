import pandas as pd

df = pd.read_csv('./data/thing_train_data/analysis_kishino.csv')
df['arrival_time'] = pd.to_datetime(df['arrival_time'])
df_sorted = df.sort_values(by='arrival_time')

df_sorted['id'] = range(len(df))
df_sorted.to_csv('./data/thing_train_data/sorted_kishino.csv', index=False)
