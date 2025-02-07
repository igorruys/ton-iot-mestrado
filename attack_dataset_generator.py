import pandas as pd

df_original = pd.read_csv('data/train_test_network.csv')

df = df_original[df_original.type != 'normal']

df.to_csv('data/attack_dataset.csv', index=False)