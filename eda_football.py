from datasets import load_dataset
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import great_expectations as gx

# Load dataset
dataset = load_dataset("3zden/fbref_football_player_performance_2024-2025")
dataframe = pd.DataFrame(dataset['train'])

# Sample if too big
if len(dataframe) > 5000:
    dataframe = dataframe.sample(n=5000, random_state=42)

# Initial EDA
print(f"Shape: {dataframe.shape}")
print(dataframe.dtypes)
print(dataframe.head())
print(dataframe.describe())
print(dataframe.isnull().sum())


brazil_df = dataframe[dataframe["Nation"] == "BRA"]
sns.scatterplot(x="Matches Played", y="Goals", data=brazil_df)
plt.title("Matches Played vs Goals (Brazil)")
plt.xlabel("Matches Played")
plt.ylabel("Goals Scored")
plt.grid(True)
plt.tight_layout()
plt.savefig('brazil_scatter.png')
plt.show()
plt.close()

# Great Expectations - Data Quality
context = gx.get_context()
datasource = context.sources.add_or_update_pandas("my_data")
data_asset = datasource.add_dataframe_asset(name="football")
batch_request = data_asset.build_batch_request(dataframe=dataframe)
suite = context.add_or_update_expectation_suite("quality_suite")
validator = context.get_validator(batch_request=batch_request, expectation_suite_name="quality_suite")
    
    # Run checks
validator.expect_column_to_exist('Goals')
validator.expect_column_values_to_not_be_null('Goals')
validator.expect_column_values_to_be_between('Goals', min_value=0)
    
print("\n Great Expectations checks passed")
    

