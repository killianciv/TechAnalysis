
import pandas as pd

input_file = "data/future_opportunities.csv"
output_file = "data/opportunities_sorted.csv"

df = pd.read_csv(input_file, na_values=["None", "none", ""])

# Remove rows where both dates are missing
df = df.dropna(subset=["slump began", "opportunity date"], how="all")

df["slump began"] = pd.to_datetime(df["slump began"], errors="coerce")
df["opportunity date"] = pd.to_datetime(df["opportunity date"], errors="coerce")

df = df.sort_values("slump began", ascending=True)

df.to_csv(output_file, index=False, date_format="%Y-%m-%d")

print(f"Saved to {output_file}")
print(f"Remaining opportunities: {len(df)}")

