import pandas as pd

# Read the first CSV file into a DataFrame
df1 = pd.read_csv('movies_metadata.csv')

# Read the second CSV file into a DataFrame
df2 = pd.read_csv('keywords.csv')

# Convert the 'id' column to string type in both DataFrames
df1['id'] = df1['id'].astype(str)
df2['id'] = df2['id'].astype(str)

# Merge the two DataFrames based on the common 'id' column
merged_df = pd.merge(df1, df2, on='id', how='inner')

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('merged_file.csv', index=False)