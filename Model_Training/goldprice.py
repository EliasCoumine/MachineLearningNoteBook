from datasets import load_dataset
import pandas as pd


dataset = load_dataset("aaronjpi/gold-price-5-years")

dataframe = pd.DataFrame(dataset['train'])
df = dataframe[['Open','Date']]

print("Dataset Information:")

print(df.head())







