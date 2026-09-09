"""
Exploratory Data Analysis — Health Insurance Portfolio / Lapse Dataset
------------------------------------------------------------------------
Column reference (from the sample you shared):

id / ID_policy / ID_insured        - keys
period                              - observation year
date_effect_insured / date_lapse_insured   - insured-level start/lapse dates
date_effect_policy  / date_lapse_policy    - policy-level start/lapse dates
year_effect_* / year_lapse_*        - year versions of the above
exposure_time                       - fraction of the year the policy was active
lapse                               - target-ish flag (but check encoding, see below)
seniority_insured / seniority_policy - tenure in years
type_policy / type_policy_dg        - policy type / policy type detail
type_product                        - product type (S, P, ...)
reimbursement                       - Yes/No
new_business                        - Yes/No
distribution_channel                - A, I, ...
gender, age                         - demographics
premium, cost_claims_year           - money
n_medical_services                  - utilization count
n_insured_pc / n_insured_mun / n_insured_prov  - portfolio concentration counts
IICIMUN / IICIPROV                  - concentration indices (mun/province)
C_H, C_GI, C_II, C_IE_*, C_GE_*, C_C - risk/cost segmentation codes & scores

NOTE on `lapse`: in your sample it takes values 1, 2, 3 — not the usual 0/1.
Don't assume it's already a clean binary target; the script below inspects
it explicitly (cell 2) before you build any model on top of it. It looks
like it may encode something like {1: lapsed mid-history, 2: still active,
3: lapsed at period end} — confirm against your data dictionary if you have
one, or infer from date_lapse_insured / date_lapse_policy being populated.

Usage: update CSV_PATH below, then run cell by cell (or as a whole script).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 160)
sns.set_theme(style="whitegrid")

CSV_PATH = "your_file.csv"   # <-- point this at your actual file (csv/xlsx/parquet)
SEP = ","                    # change to ";" or "\t" if needed

# ----------------------------------------------------------------------
# 1. LOAD & BASIC PROFILE
# ----------------------------------------------------------------------
df = pd.read_csv(CSV_PATH, sep=SEP)
print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nMissing values (%):\n", (df.isna().mean() * 100).round(1).sort_values(ascending=False))
print("\nDuplicate rows:", df.duplicated().sum())

DATE_COLS = ["date_effect_insured", "date_lapse_insured",
             "date_effect_policy", "date_lapse_policy"]
for c in DATE_COLS:
    df[c] = pd.to_datetime(df[c], format="%d-%m-%Y", errors="coerce")

# ----------------------------------------------------------------------
# 2. UNDERSTAND THE TARGET (`lapse`) BEFORE MODELING ANYTHING
# ----------------------------------------------------------------------
print("\nlapse value counts:\n", df["lapse"].value_counts(dropna=False))

# Cross-check against whether a lapse date actually exists in this period
df["has_lapse_date"] = df["date_lapse_insured"].notna() | df["date_lapse_policy"].notna()
print("\nlapse code vs has_lapse_date:\n",
      pd.crosstab(df["lapse"], df["has_lapse_date"]))

# If you want a clean binary target for modeling, define it explicitly, e.g.:
# df["target_lapse"] = df["has_lapse_date"].astype(int)
# (swap for whatever mapping matches your data dictionary)

# ----------------------------------------------------------------------
# 3. UNIVARIATE LOOKS AT KEY NUMERIC FIELDS
# ----------------------------------------------------------------------
numeric_cols = ["age", "premium", "cost_claims_year", "n_medical_services",
                 "exposure_time", "seniority_insured", "seniority_policy"]
print("\nDescribe numeric cols:\n", df[numeric_cols].describe().T)

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.flat, numeric_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=ax, bins=30)
    ax.set_title(col)
plt.tight_layout()
plt.savefig("univariate_distributions.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 4. LOSS RATIO / CLAIMS BEHAVIOR
# ----------------------------------------------------------------------
df["loss_ratio"] = df["cost_claims_year"] / df["premium"].replace(0, np.nan)
df["cost_per_service"] = df["cost_claims_year"] / df["n_medical_services"].replace(0, np.nan)

print("\nLoss ratio summary:\n", df["loss_ratio"].describe())

# Loss ratio by categorical segments
for cat in ["type_product", "type_policy", "distribution_channel", "gender", "new_business"]:
    print(f"\n--- Mean loss ratio & lapse rate by {cat} ---")
    grp = df.groupby(cat).agg(
        n=("id", "count"),
        mean_premium=("premium", "mean"),
        mean_claims=("cost_claims_year", "mean"),
        mean_loss_ratio=("loss_ratio", "mean"),
        lapse_rate=("has_lapse_date", "mean"),
    ).round(2)
    print(grp)

# ----------------------------------------------------------------------
# 5. AGE / SENIORITY VS LAPSE
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df, x="has_lapse_date", y="age", ax=axes[0])
axes[0].set_title("Age vs lapse")
sns.boxplot(data=df, x="has_lapse_date", y="seniority_insured", ax=axes[1])
axes[1].set_title("Seniority vs lapse")
plt.tight_layout()
plt.savefig("age_seniority_vs_lapse.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 6. GEOGRAPHIC / PORTFOLIO CONCENTRATION
# ----------------------------------------------------------------------
conc_cols = ["n_insured_pc", "n_insured_mun", "n_insured_prov", "IICIMUN", "IICIPROV"]
print("\nConcentration index summary:\n", df[conc_cols].describe().T)

# ----------------------------------------------------------------------
# 7. CORRELATION HEATMAP (numeric features only)
# ----------------------------------------------------------------------
corr_cols = numeric_cols + ["loss_ratio", "n_insured_pc", "n_insured_mun",
                             "n_insured_prov", "IICIMUN", "IICIPROV"]
corr = df[corr_cols].corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 8. YEAR-OVER-YEAR TRENDS (if you have multiple `period` values)
# ----------------------------------------------------------------------
if df["period"].nunique() > 1:
    trend = df.groupby("period").agg(
        n_policies=("id", "count"),
        lapse_rate=("has_lapse_date", "mean"),
        mean_premium=("premium", "mean"),
        mean_claims=("cost_claims_year", "mean"),
    )
    print("\nYear-over-year trend:\n", trend)

print("\nDone. PNGs saved in the working directory:")
print(" - univariate_distributions.png")
print(" - age_seniority_vs_lapse.png")
print(" - correlation_heatmap.png")