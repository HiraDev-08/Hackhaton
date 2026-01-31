# eda_features.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# -----------------------------
# Numerical Features vs Target
# -----------------------------
def num_vs_target(df, target):
    """
    Analyze correlation of numerical features with target variable.

    Parameters:
    df     : pandas.DataFrame
    target : str, target column name

    Returns:
    dict : {feature_name: correlation_with_target}
    """
    if target not in df.columns:
        raise ValueError(f"{target} column not found in DataFrame")

    result = {}
    numeric_cols = df.select_dtypes(include="number").columns.drop(target, errors='ignore')
    for col in numeric_cols:
        if df[col].nunique() > 1:  # skip constant columns
            result[col] = df[col].corr(df[target])
    return result

# -----------------------------
# Categorical Features vs Target
# -----------------------------
def cat_vs_target(df, target):
    """
    Analyze categorical features vs target using chi-square test.

    Parameters:
    df     : pandas.DataFrame
    target : str, target column name

    Returns:
    dict : {feature_name: p_value_with_target or None if not testable}
    """
    if target not in df.columns:
        raise ValueError(f"{target} column not found in DataFrame")
    
    result = {}
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        table = pd.crosstab(df[col], df[target])
        if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
            result[col] = None  # skip non-testable columns
            continue
        chi2, p, _, _ = chi2_contingency(table)
        result[col] = p
    return result

# -----------------------------
# Plot Numerical Features
# -----------------------------
def plot_num_vs_target(df, target, top_n=10):
    correlations = num_vs_target(df, target)
    top_features = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    top_cols = [f[0] for f in top_features]
    
    for col in top_cols:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df[col], y=df[target])
        plt.title(f"{col} vs {target} (corr={correlations[col]:.2f})")
        plt.xlabel(col)
        plt.ylabel(target)
        plt.tight_layout()
        plt.show()

# -----------------------------
# Plot Categorical Features
# -----------------------------
def plot_cat_vs_target(df, target, top_n=10):
    p_values = cat_vs_target(df, target)
    
    # filter out None values
    p_values = {k:v for k,v in p_values.items() if v is not None}
    top_features = sorted(p_values.items(), key=lambda x: x[1])[:top_n]  # lowest p-values
    top_cols = [f[0] for f in top_features]
    
    for col in top_cols:
        plt.figure(figsize=(8,4))
        sns.boxplot(x=df[col], y=df[target])
        plt.title(f"{col} vs {target} (p-value={p_values[col]:.2e})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
