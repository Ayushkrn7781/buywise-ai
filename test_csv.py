import pandas as pd

df = pd.read_csv(
    "Amazon_Reviews.csv",
    encoding='utf-8',
    engine='python',
    on_bad_lines='skip'
)

print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("\nRating samples:")
print(df['Rating'].head(10))