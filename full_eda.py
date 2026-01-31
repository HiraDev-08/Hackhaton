# full_eda.py
import pandas as pd
from eda_missing import missing_percentage, drop_missing, duplicate_rows
from eda_outliers import iqr_outliers
from eda_features import num_vs_target, cat_vs_target, plot_num_vs_target, plot_cat_vs_target

def full_eda(df, target, top_n=10, plot=True):
    """
    Perform full exploratory data analysis on a DataFrame.

    Parameters:
    df      : pandas.DataFrame - input dataset
    target  : str - target column name
    top_n   : int - number of top numerical features to visualize
    plot    : bool - whether to generate plots

    Returns:
    df_clean: pandas.DataFrame - cleaned dataset
    insights: dict - dictionary of analysis results and top features
    """

    # --- Input Validation ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    insights = {}
    df_clean = df.copy()

    # --- Missing Values ---
    insights["missing_percentage"] = missing_percentage(df_clean)

    # Fill numeric columns EXCEPT target
    num_cols = df_clean.select_dtypes(include="number").columns.drop(target, errors='ignore')
    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())

    # Drop rows only if target is missing
    df_clean = df_clean.dropna(subset=[target])

    # Optional: Fill categorical columns missing values with "Missing"
    cat_cols = df_clean.select_dtypes(include="object").columns
    df_clean[cat_cols] = df_clean[cat_cols].fillna("Missing")

    # Detect duplicates
    insights["duplicates"] = duplicate_rows(df_clean)

    # --- Target Analysis ---
    insights["target_mean"] = df_clean[target].mean()
    insights["target_std"] = df_clean[target].std()
    insights["target_outliers"] = iqr_outliers(df_clean[target]).tolist()

    # --- Feature Analysis ---
    num_rel = num_vs_target(df_clean, target)
    cat_rel = cat_vs_target(df_clean, target)

    insights["num_vs_target_corr"] = num_rel
    insights["cat_vs_target_p"] = cat_rel

    # --- Top Features ---
    top_num = sorted(num_rel.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    insights["top_features"] = top_num

    # --- Plots (optional) ---
    if plot:
        print("\nPlotting top numerical features vs target...")
        plot_num_vs_target(df_clean, target, top_n=top_n)
        print("\nPlotting top categorical features vs target...")
        plot_cat_vs_target(df_clean, target, top_n=top_n)

    return df_clean, insights
