import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('merged_file.csv')

# print out how many rows before
print(f'Number of rows after cleaning: {len(df)}')

# Specify the column name that contains the arrays
column_name = 'keywords'

# Remove rows where the specified column is an empty array
df = df[df[column_name].astype(str) != '[]']

# Save the cleaned DataFrame back to a CSV file
df.to_csv('cleaned_file.csv', index=False)

# print out how many rows after
print(f'Number of rows after cleaning: {len(df)}')