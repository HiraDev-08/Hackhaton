import pandas as pd
from full_eda import full_eda

# Load dataset
df = pd.read_csv("AmesHousing.csv")

# Run full EDA
clean_df, insights = full_eda(df, target="SalePrice")

# Print insights
print("=== INSIGHTS ===")
for key, value in insights.items():
    print(f"\n{key}:")
    print(value)
