import pandas as pd

# Load the dataset
df = pd.read_csv("titanic.csv")

# Display first few rows
print("Initial Data:")
print(df.head())

# 1. Data Cleaning
# Fill missing 'Age' with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing 'Embarked' with the most common value
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop columns not needed
df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

# 2. Noisy Data - Binning 'Age'
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 100],
                        labels=['Child', 'Teen', 'Adult', 'Middle-Aged', 'Senior'])

# 3. Data Integration - Checking correlation between numerical values
print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

# Save the cleaned and processed data
df.to_csv("titanic_cleaned.csv", index=False)

print("\nPreprocessing complete. Cleaned data saved as 'titanic_cleaned.csv'.")
