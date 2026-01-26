from datasets import load_dataset
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns


dataset = load_dataset("3zden/fbref_football_player_performance_2024-2025")

 # print(dataset)


dataframe = pd.DataFrame(dataset['train'])

brazil_df = dataframe[dataframe["Nation"] == "BRA"]

print(brazil_df)


print(len(brazil_df))
print(brazil_df.shape)
print(brazil_df.dtypes)
print(brazil_df.info())


plt.figure(figsize=(10, 6))  # Makes it wider
sns.scatterplot(x="Matches Played", y="Goals", data=brazil_df)
plt.title("Minutes Played vs Goals (Brazil)")
plt.xlabel("Matches Played")
plt.ylabel("Goals Scored")
plt.grid(True)
plt.tight_layout()
plt.show()



\



