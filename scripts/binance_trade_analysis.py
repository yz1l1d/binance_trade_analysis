import pandas as pd
import numpy as np
import ast
from pandas import json_normalize


df = pd.read_csv("data\\TRADES_CopyTr_90D_ROI.csv")


df = df.dropna(subset=['Trade_History'])

print("Before type conversion:")
print(df['Trade_History'].apply(type).value_counts())  

df['Trade_History'] = df['Trade_History'].apply(ast.literal_eval)

print("After type conversion:")
print(df['Trade_History'].apply(type).value_counts())  

df = df.explode('Trade_History').reset_index(drop=True)

print("After explode:")
print(df.head()) 

if isinstance(df['Trade_History'].iloc[0], dict):  
    df = df.join(pd.json_normalize(df['Trade_History'])).drop(columns=['Trade_History'])

print("After normalization:")
print(df.head())  

df.to_csv("data\\processed_trades.csv", index=False)
