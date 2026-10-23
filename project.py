"""
INT234 Predictive Analytics - CA2 Project
Project : Students Performance Analysis and Prediction
Dataset : StudentsPerformance.csv (1000 students, 8 columns)
Run     : python students_performance_project.py
          (keep StudentsPerformance.csv in the SAME folder, or in a 'data' folder)

HOW THIS FILE IS ORGANISED
Every section starts with a banner "PART n - TITLE" that tells you:
  WHAT  : what the part does
  UNIT  : which INT234 syllabus unit it covers
  OBJ   : which project objective it answers
Search for "PART" to jump between sections. Nothing is hardcoded: every number
you quote in your report or viva comes from what this script prints or saves.

PROJECT OBJECTIVES (the five required)
  OBJ1  Which background factors (test prep, lunch, parental education, gender,
        race/ethnicity) are associated with higher exam scores?
  OBJ2  How strongly are the three subject scores correlated with each other?
  OBJ3  Can math score be predicted? (regression, 9 models compared)
  OBJ4  Can students at risk of failing math be identified? (classification,
        10 models compared)
  OBJ5  Do students form natural performance groups, and which attributes occur
        together? (K-Means, K-Medoids, hierarchical, PCA, association rules)

SYLLABUS COVERAGE MAP
  Unit I   : PART 1-4 (preprocessing, EDA, feature engineering)
  Unit II  : PART 6 (simple/multiple/polynomial regression, OLS, logistic in PART 7)
  Unit III : PART 7 (KNN, Naive Bayes, Decision Tree, SVM + all metrics)
  Unit IV  : PART 8 (K-Means, K-Medoids, hierarchical + 4 linkages), PART 9 (Apriori)
  Unit V   : PART 10 (PCA), PART 6-7 (MLP feedforward neural network)
  Unit VI  : PART 11 (bias-variance), PART 6-7 (bagging, boosting, random forest)
"""

# =============================================================================
# PART 0 - SETTINGS AND IMPORTS
# WHAT : load libraries, fix random seed, create output folders
# UNIT : Unit I (workflow set-up)
# =============================================================================
from itertools import combinations
from pathlib import Path
import json
import warnings

import matplotlib
SHOW_PLOTS = False            # True = show plots on screen. False = only save PNG files
if not SHOW_PLOTS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier,
                              BaggingRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             confusion_matrix, f1_score, log_loss, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder, PolynomialFeatures,
                                   StandardScaler)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")
SEED = 42                     # reproducibility: same split and results every run
np.random.seed(SEED)
sns.set_theme(style="whitegrid")

PASS_MARK = 50                # ASSUMPTION (yours to justify/change): math < 50 = "at risk"
EDU_ORDER = ["some high school", "high school", "some college",
             "associate's degree", "bachelor's degree", "master's degree"]
SCORE_COLS = ["math_score", "reading_score", "writing_score"]
NOMINAL_COLS = ["gender", "race_ethnicity", "lunch", "test_prep"]   # no natural order

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
FIG_DIR = HERE / "outputs" / "figures"
TAB_DIR = HERE / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY = {}                  # key numbers collected for the report (saved at the end)


def save_fig(name):
    """Save the current figure into outputs/figures (and show it if SHOW_PLOTS)."""
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def save_table(table, name, index=True):
    """Save a DataFrame into outputs/tables as CSV."""
    table.to_csv(TAB_DIR / f"{name}.csv", index=index)


def header(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


# =============================================================================
# PART 1 - DATA LOADING
# WHAT : read the CSV and take a first look (rows, columns, data types)
# UNIT : Unit I (data preprocessing)
# =============================================================================
header("PART 1 - DATA LOADING")
DATA_FILE = next((p for p in [HERE / "StudentsPerformance.csv",
                              HERE / "data" / "StudentsPerformance.csv",
                              HERE.parent / "data" / "StudentsPerformance.csv"] if p.exists()), None)
if DATA_FILE is None:
    raise FileNotFoundError("Put StudentsPerformance.csv next to this script or in a 'data' folder.")

df_raw = pd.read_csv(DATA_FILE)
df = df_raw.copy()            # we always clean a COPY and keep the original untouched
print("Loaded:", DATA_FILE.name)
print("Shape (rows, columns):", df.shape)
print(df.head())
print("\nData types:\n", df.dtypes)

# =============================================================================
# PART 2 - DATA QUALITY ASSESSMENT AND CLEANING
# WHAT : missing values, duplicates, inconsistent text, wrong types,
#        invalid values, outlier investigation, final verification
# UNIT : Unit I (preprocessing)
# NOTE : this dataset is already clean. The checks are still run so you can
#        SHOW they were done; report honestly that nothing needed fixing
#        if the printed counts are zero.
# =============================================================================
header("PART 2 - DATA QUALITY ASSESSMENT AND CLEANING")

# 2.1 Column names -> simple snake_case so they are easy to type
df.columns = (df.columns.str.strip().str.lower()
              .str.replace("/", "_").str.replace(" ", "_"))
df = df.rename(columns={"parental_level_of_education": "parent_education",
                        "test_preparation_course": "test_prep"})
print("Cleaned column names:", list(df.columns))

# 2.2 Missing values
missing = df.isnull().sum()
print("\nMissing values per column:\n", missing)

# 2.3 Duplicates
n_dup = int(df.duplicated().sum())
print("\nDuplicate rows:", n_dup)
if n_dup > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Duplicates removed. New shape:", df.shape)

# 2.4 Inconsistent text (extra spaces, mixed upper/lower case)
text_cols = df.select_dtypes(include=["object", "string"]).columns
for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()
    print(f"\n{col}: {sorted(df[col].unique())}")   # eyeball the categories
unexpected = set(df["parent_education"].unique()) - set(EDU_ORDER)
print("\nEducation labels not in expected list:", unexpected or "none")

# 2.5 Data types: scores must be numeric
for col in SCORE_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")
print("Missing after type conversion:", int(df[SCORE_COLS].isnull().sum().sum()))

# 2.6 Invalid values: a score must lie between 0 and 100
invalid = ((df[SCORE_COLS] < 0) | (df[SCORE_COLS] > 100)).sum()
print("\nScores outside 0-100:\n", invalid)

# 2.7 Outlier investigation with the IQR rule (we INVESTIGATE, then decide)
outlier_rows = []
for col in SCORE_COLS:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_rows.append({"column": col, "lower_fence": low, "upper_fence": high,
                         "n_outliers": int(((df[col] < low) | (df[col] > high)).sum()),
                         "min": df[col].min(), "max": df[col].max()})
outlier_table = pd.DataFrame(outlier_rows)
print("\nOutlier summary (IQR rule):\n", outlier_table)
# DECISION: low scores inside 0-100 are VALID observations (a student really can
# score 0 or 8), and they are exactly the students we want to detect later.
# So outliers are KEPT, not deleted. Say this in your report and viva.

# 2.8 Verification
quality_report = pd.DataFrame({
    "check": ["rows", "columns", "missing_cells", "duplicate_rows", "scores_outside_0_100"],
    "result": [len(df), df.shape[1], int(df.isnull().sum().sum()), int(df.duplicated().sum()),
               int(invalid.sum())]})
print("\nFINAL QUALITY REPORT\n", quality_report)
save_table(quality_report, "01_data_quality_report", index=False)
save_table(outlier_table, "02_outlier_summary", index=False)
SUMMARY["rows"], SUMMARY["columns"] = int(df_raw.shape[0]), int(df_raw.shape[1])
SUMMARY["missing_cells"] = int(df.isnull().sum().sum())
SUMMARY["duplicates_found"] = n_dup
SUMMARY["outliers"] = {r["column"]: r["n_outliers"] for r in outlier_rows}

# =============================================================================
# PART 3 - EXPLORATORY DATA ANALYSIS
# WHAT : charts + tables that answer OBJ1 and OBJ2
# UNIT : Unit I (EDA), Unit II (correlation)
# =============================================================================
header("PART 3 - EXPLORATORY DATA ANALYSIS")

# 3.1 Distribution of each score (histogram)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, SCORE_COLS):
    sns.histplot(df[col], bins=20, kde=True, ax=ax)
    ax.set_title(f"Distribution of {col.replace('_', ' ').title()}")
    ax.set_xlabel("Score")
    ax.set_ylabel("Number of students")
save_fig("01_score_distributions")
score_summary = df[SCORE_COLS].describe().T
save_table(score_summary, "03_score_summary")
print(score_summary)

# 3.2 Count plots of the categorical columns
cat_cols_all = ["gender", "race_ethnicity", "parent_education", "lunch", "test_prep"]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.ravel(), cat_cols_all):
    order = EDU_ORDER if col == "parent_education" else None
    sns.countplot(data=df, x=col, order=order, ax=ax)
    ax.set_title(f"Students by {col.replace('_', ' ').title()}")
    ax.set_xlabel(col.replace("_", " ").title())
    ax.set_ylabel("Number of students")
    ax.tick_params(axis="x", rotation=30)
axes.ravel()[-1].axis("off")
save_fig("02_category_counts")

# 3.3 OBJ1 - mean scores for every category (bar charts)
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
group_tables = {}
for ax, col in zip(axes.ravel(), cat_cols_all):
    means = df.groupby(col)[SCORE_COLS].mean()
    if col == "parent_education":
        means = means.loc[EDU_ORDER]
    group_tables[col] = means.round(2)
    means.plot(kind="bar", ax=ax)
    ax.set_title(f"Average Scores by {col.replace('_', ' ').title()}")
    ax.set_xlabel(col.replace("_", " ").title())
    ax.set_ylabel("Mean score")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Subject", fontsize=8)
axes.ravel()[-1].axis("off")
save_fig("03_mean_scores_by_category")
for col, tbl in group_tables.items():
    print(f"\nMean scores by {col}:\n{tbl}")
    save_table(tbl, f"04_mean_scores_by_{col}")

# 3.4 OBJ1 - box plots: math score vs test prep and lunch
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df, x="test_prep", y="math_score", ax=axes[0])
axes[0].set_title("Math Score by Test Preparation Course")
axes[0].set_xlabel("Test preparation course")
axes[0].set_ylabel("Math score")
sns.boxplot(data=df, x="lunch", y="math_score", ax=axes[1])
axes[1].set_title("Math Score by Lunch Type")
axes[1].set_xlabel("Lunch type")
axes[1].set_ylabel("Math score")
save_fig("04_boxplots_math_by_prep_lunch")

# 3.5 OBJ1 - are the differences statistically significant? (Welch t-test)
test_rows = []
for col, a, b in [("test_prep", "completed", "none"),
                  ("lunch", "standard", "free/reduced"),
                  ("gender", "female", "male")]:
    for score in SCORE_COLS:
        g1, g2 = df.loc[df[col] == a, score], df.loc[df[col] == b, score]
        t, p = stats.ttest_ind(g1, g2, equal_var=False)
        test_rows.append({"factor": col, "group_1": a, "group_2": b, "subject": score,
                          "mean_group_1": g1.mean(), "mean_group_2": g2.mean(),
                          "difference": g1.mean() - g2.mean(), "t_stat": t, "p_value": p})
ttest_table = pd.DataFrame(test_rows)
print("\nWelch t-tests (p < 0.05 means the difference is unlikely to be chance):")
print(ttest_table[["factor", "subject", "difference", "t_stat", "p_value"]].to_string(index=False))
save_table(ttest_table, "05_ttests", index=False)
# CORRELATION IS NOT CAUSATION: a difference by lunch type does not mean lunch
# CAUSES lower scores; it is a proxy for family income and other factors.

# 3.6 OBJ2 - correlation heatmap (temporary numeric encoding, only for this chart)
enc = df[SCORE_COLS].copy()
enc["parent_edu_level"] = df["parent_education"].map({e: i for i, e in enumerate(EDU_ORDER)})
enc["lunch_free_reduced"] = (df["lunch"] == "free/reduced").astype(int)
enc["test_prep_done"] = (df["test_prep"] == "completed").astype(int)
enc["gender_male"] = (df["gender"] == "male").astype(int)
corr = enc.corr(method="spearman")
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("Spearman Correlation: Scores and Encoded Background Factors")
save_fig("05_correlation_heatmap")
save_table(corr.round(3), "06_correlation_matrix")
score_corr = df[SCORE_COLS].corr()
print("\nScore-to-score Pearson correlation:\n", score_corr.round(3))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.scatterplot(data=df, x="reading_score", y="math_score", hue="test_prep", alpha=0.6, ax=axes[0])
axes[0].set_title("Math vs Reading Score")
axes[0].set_xlabel("Reading score")
axes[0].set_ylabel("Math score")
sns.scatterplot(data=df, x="reading_score", y="writing_score", hue="gender", alpha=0.6, ax=axes[1])
axes[1].set_title("Writing vs Reading Score")
axes[1].set_xlabel("Reading score")
axes[1].set_ylabel("Writing score")
save_fig("06_scatter_scores")

# 3.7 Pair plot of the three scores
pp = sns.pairplot(df[SCORE_COLS + ["test_prep"]], hue="test_prep", corner=True, plot_kws={"alpha": 0.5})
pp.fig.suptitle("Pair Plot of Subject Scores (colour = test preparation)", y=1.02)
plt.savefig(FIG_DIR / "07_pairplot.png", dpi=130, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
plt.close()

# =============================================================================
# PART 4 - FEATURE ENGINEERING
# WHAT : create features that models can use
# UNIT : Unit I
# LEAKAGE RULE: math_score is our TARGET, so nothing built from math_score may
#        be used as an input (no 'total score' or 'average of all three').
# =============================================================================
header("PART 4 - FEATURE ENGINEERING")

# 4.1 reading_writing_avg: reading and writing are almost the same signal
#     (see correlation in Part 3), so one average avoids duplicate information
#     and unstable linear-model coefficients (multicollinearity).
df["reading_writing_avg"] = df[["reading_score", "writing_score"]].mean(axis=1)

# 4.2 Ordered education level (0 = some high school ... 5 = master's degree)
df["parent_edu_level"] = df["parent_education"].map({e: i for i, e in enumerate(EDU_ORDER)})

# 4.3 Classification target: 1 = at risk (math below PASS_MARK), 0 = not at risk
df["at_risk_math"] = (df["math_score"] < PASS_MARK).astype(int)
print(f"At-risk students (math < {PASS_MARK}):", int(df["at_risk_math"].sum()),
      f"({df['at_risk_math'].mean():.1%} of students)")
SUMMARY["pass_mark"] = PASS_MARK
SUMMARY["at_risk_count"] = int(df["at_risk_math"].sum())
SUMMARY["at_risk_share"] = float(df["at_risk_math"].mean())

# 4.4 Score bands (low/medium/high thirds). Used ONLY for association rules (PART 9)
for col in SCORE_COLS:
    df[col.replace("_score", "_band")] = pd.qcut(df[col], 3, labels=["low", "medium", "high"])
print("Columns now:", list(df.columns))

# =============================================================================
# PART 5 - PREPROCESSING PIPELINE BUILDER AND MODEL-SELECTION RULE
# WHAT : one function that builds the encoding/scaling steps
# UNIT : Unit I
# LEAKAGE : the pipeline is fitted INSIDE each training fold only, so the test
#           set never influences scaling or encoding.
# =============================================================================
def build_preprocessor(num_cols, poly=False, degree=2):
    """Scale numeric cols, ordinal-encode education, one-hot the nominal cols."""
    parts = []
    if num_cols:
        steps = [("scale", StandardScaler())]
        if poly:
            steps.append(("poly", PolynomialFeatures(degree, include_bias=False)))
        parts.append(("num", Pipeline(steps), num_cols))
    parts.append(("edu", Pipeline([("ord", OrdinalEncoder(categories=[EDU_ORDER])),
                                   ("scale", StandardScaler())]), ["parent_education"]))
    parts.append(("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                                       sparse_output=False), NOMINAL_COLS))
    return ColumnTransformer(parts)


def one_se_best(res, mean_col, std_col):
    """
    Model-selection rule (not just 'highest score'): keep every model whose
    cross-validated score is within one standard deviation of the best, then pick
    the SIMPLEST of them (lowest 'complexity'). This favours models that
    generalise and are easier to explain.
    """
    best = res.loc[res[mean_col].idxmax()]
    ok = res[res[mean_col] >= best[mean_col] - best[std_col]]
    return ok.sort_values(["complexity", mean_col], ascending=[True, False]).iloc[0]["model"]


# =============================================================================
# PART 6 - REGRESSION: PREDICT MATH SCORE
# WHAT : Simple LR, Multiple LR (OLS), Ridge, Polynomial, Random Forest,
#        Gradient Boosting, Bagging, AdaBoost, MLP (feedforward neural network).
#        Two feature sets are compared:
#          A) background factors only          (no other test scores)
#          B) background + reading/writing avg (main model)
# UNIT : Unit II (regression, OLS, MAE/MSE/RMSE/R2), Unit V (MLP),
#        Unit VI (bagging, boosting, random forest)
# OBJ  : OBJ3
# NOTE : OLS = Ordinary Least Squares: choose the line/plane that minimises the
#        sum of squared errors. LinearRegression and linregress both use it.
# =============================================================================
header("PART 6 - REGRESSION (OBJ3)")

# 6.1 Simple linear regression: math ~ reading (one input, one output)
slr = stats.linregress(df["reading_score"], df["math_score"])
print(f"Simple linear regression:  math = {slr.intercept:.2f} + {slr.slope:.3f} * reading")
print(f"Correlation r = {slr.rvalue:.3f}   R-squared = {slr.rvalue ** 2:.3f}   p-value = {slr.pvalue:.2e}")
SUMMARY["slr"] = {"intercept": slr.intercept, "slope": slr.slope, "r": slr.rvalue,
                  "r2": slr.rvalue ** 2}
plt.figure(figsize=(7, 5))
sns.regplot(data=df, x="reading_score", y="math_score", scatter_kws={"alpha": 0.4},
            line_kws={"color": "red"})
plt.title("Simple Linear Regression: Math Score vs Reading Score")
plt.xlabel("Reading score")
plt.ylabel("Math score")
save_fig("08_simple_linear_regression")

y_reg = df["math_score"]
cv_reg = KFold(n_splits=5, shuffle=True, random_state=SEED)


def reg_row(label, name, rank, y_tr, p_tr, y_te, p_te, cv):
    mse = mean_squared_error(y_te, p_te)
    return {"feature_set": label, "model": name, "complexity": rank,
            "MAE": mean_absolute_error(y_te, p_te), "MSE": mse, "RMSE": np.sqrt(mse),
            "R2_test": r2_score(y_te, p_te), "R2_train": r2_score(y_tr, p_tr),
            "R2_gap(train-test)": r2_score(y_tr, p_tr) - r2_score(y_te, p_te),
            "CV_R2_mean": cv.mean(), "CV_R2_std": cv.std()}


def run_regression(num_cols, label):
    feature_cols = num_cols + ["parent_education"] + NOMINAL_COLS
    X = df[feature_cols]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_reg, test_size=0.2, random_state=SEED)

    # name: (estimator, uses polynomial features?, complexity rank for selection rule)
    models = {
        "Multiple Linear (OLS)": (LinearRegression(), False, 1),
        "Ridge Regression": (Ridge(alpha=10), False, 2),
        "Random Forest": (RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                random_state=SEED), False, 4),
        "Bagging (Decision Trees)": (BaggingRegressor(DecisionTreeRegressor(min_samples_leaf=5),
                                                      n_estimators=100, random_state=SEED), False, 4),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=SEED), False, 5),
        "AdaBoost": (AdaBoostRegressor(n_estimators=100, random_state=SEED), False, 5),
        "MLP Neural Network": (MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=3000,
                                            early_stopping=True, random_state=SEED), False, 4),
    }
    if num_cols:                  # polynomial terms need at least one numeric input
        models["Polynomial Regression (deg 2)"] = (Ridge(alpha=10), True, 3)

    rows, fitted, preds = [], {}, {}
    for name, (est, poly, rank) in models.items():
        pipe = Pipeline([("prep", build_preprocessor(num_cols, poly)), ("model", est)])
        cv = cross_val_score(pipe, X_tr, y_tr, cv=cv_reg, scoring="r2")
        pipe.fit(X_tr, y_tr)
        p_te, p_tr = pipe.predict(X_te), pipe.predict(X_tr)
        rows.append(reg_row(label, name, rank, y_tr, p_tr, y_te, p_te, cv))
        fitted[name], preds[name] = pipe, p_te

    if num_cols:                  # simple linear regression on reading score only
        Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(df[["reading_score"]], y_reg,
                                                      test_size=0.2, random_state=SEED)
        cv = cross_val_score(LinearRegression(), Xs_tr, ys_tr, cv=cv_reg, scoring="r2")
        m = LinearRegression().fit(Xs_tr, ys_tr)
        rows.append(reg_row(label, "Simple Linear (reading only)", 0, ys_tr, m.predict(Xs_tr),
                            ys_te, m.predict(Xs_te), cv))
        preds["Simple Linear (reading only)"] = m.predict(Xs_te)
    return pd.DataFrame(rows), fitted, preds, y_te


res_a, _, _, _ = run_regression([], "A: background only")
res_b, fitted_b, preds_b, y_te_b = run_regression(["reading_writing_avg"],
                                                  "B: background + reading/writing")
reg_results = pd.concat([res_a, res_b], ignore_index=True)
show_cols = ["feature_set", "model", "MAE", "MSE", "RMSE", "R2_test", "R2_gap(train-test)",
             "CV_R2_mean", "CV_R2_std"]
print(reg_results[show_cols].round(3).to_string(index=False))
save_table(reg_results.drop(columns="complexity"), "07_regression_results", index=False)

best_reg = one_se_best(res_b, "CV_R2_mean", "CV_R2_std")
top_reg = res_b.loc[res_b["R2_test"].idxmax(), "model"]
print(f"\nBEST REGRESSION MODEL (set B, one-standard-error rule): {best_reg}")
print(f"Highest raw test R2 (set B): {top_reg}  <- not automatically the best-fit model")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.barplot(data=reg_results, x="R2_test", y="model", hue="feature_set", ax=axes[0])
axes[0].set_title("Regression Models: Test R-squared")
axes[0].set_xlabel("R-squared (higher is better)")
axes[0].set_ylabel("Model")
sns.barplot(data=reg_results, x="RMSE", y="model", hue="feature_set", ax=axes[1])
axes[1].set_title("Regression Models: Test RMSE")
axes[1].set_xlabel("RMSE in marks (lower is better)")
axes[1].set_ylabel("")
save_fig("09_regression_model_comparison")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(y_te_b, preds_b[best_reg], alpha=0.6)
axes[0].plot([0, 100], [0, 100], "r--", label="Perfect prediction")
axes[0].set_title(f"Actual vs Predicted Math Score ({best_reg})")
axes[0].set_xlabel("Actual math score")
axes[0].set_ylabel("Predicted math score")
axes[0].legend()
axes[1].scatter(preds_b[best_reg], y_te_b - preds_b[best_reg], alpha=0.6)
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_title("Residual Plot (should look like random scatter)")
axes[1].set_xlabel("Predicted math score")
axes[1].set_ylabel("Residual (actual - predicted)")
save_fig("10_regression_actual_vs_predicted")

lin_pipe = fitted_b["Multiple Linear (OLS)"]
coef = pd.Series(lin_pipe.named_steps["model"].coef_,
                 index=lin_pipe.named_steps["prep"].get_feature_names_out()).sort_values()
print("\nMultiple linear regression coefficients (OLS). Numeric/education inputs are "
      "standardised (effect per 1 std dev); yes/no columns are 0/1 (effect of switching "
      "category, holding the other inputs fixed). These are associations, not causes:")
print(coef.round(3))
save_table(coef.rename("coefficient").to_frame(), "08_linear_coefficients")
plt.figure(figsize=(8, 6))
coef.plot(kind="barh")
plt.title("Multiple Linear Regression Coefficients (Math Score)")
plt.xlabel("Coefficient (effect on math score)")
plt.ylabel("Feature")
save_fig("11_linear_coefficients")

# =============================================================================
# PART 7 - CLASSIFICATION: IDENTIFY STUDENTS AT RISK IN MATH
# WHAT : Logistic Regression, KNN, Naive Bayes, Decision Tree (pruned and
#        unpruned), SVM, Random Forest, Bagging, AdaBoost, Gradient Boosting, MLP.
#        Positive class = at risk (1).
# UNIT : Unit II (logistic regression), Unit III (KNN, NB, DT, SVM + metrics),
#        Unit V (MLP), Unit VI (random forest, bagging, boosting)
# OBJ  : OBJ4
# WHY F1 AND RECALL, NOT ACCURACY: at-risk students are a minority. A model
#      that predicts "nobody is at risk" would still score high accuracy.
# =============================================================================
header("PART 7 - CLASSIFICATION (OBJ4)")
CLS_NUM = ["reading_writing_avg"]
CLS_FEATURES = CLS_NUM + ["parent_education"] + NOMINAL_COLS
X_c, y_c = df[CLS_FEATURES], df["at_risk_math"]
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_c, y_c, test_size=0.2, stratify=y_c,
                                              random_state=SEED)
cv_cls = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print(f"Train: {len(Xc_tr)} rows | Test: {len(Xc_te)} rows | "
      f"at-risk share train={yc_tr.mean():.1%} test={yc_te.mean():.1%}")

cls_models = {   # name: (estimator, complexity rank)
    "Logistic Regression": (LogisticRegression(max_iter=1000, class_weight="balanced"), 1),
    "Naive Bayes": (GaussianNB(), 1),
    "KNN (k=7)": (KNeighborsClassifier(n_neighbors=7), 2),
    "Decision Tree (depth 4)": (DecisionTreeClassifier(max_depth=4, class_weight="balanced",
                                                       random_state=SEED), 2),
    "Decision Tree (unpruned)": (DecisionTreeClassifier(class_weight="balanced",
                                                        random_state=SEED), 3),
    "SVM (RBF)": (SVC(probability=True, class_weight="balanced", random_state=SEED), 3),
    "Random Forest": (RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                             class_weight="balanced", random_state=SEED), 4),
    "Bagging (Decision Trees)": (BaggingClassifier(DecisionTreeClassifier(max_depth=6),
                                                   n_estimators=100, random_state=SEED), 4),
    "AdaBoost": (AdaBoostClassifier(n_estimators=100, random_state=SEED), 4),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=SEED), 5),
    "MLP Neural Network": (MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1500,
                                         random_state=SEED), 4),
}

cls_rows, cls_fitted = [], {}
for name, (est, rank) in cls_models.items():
    pipe = Pipeline([("prep", build_preprocessor(CLS_NUM)), ("model", est)])
    cv_f1 = cross_val_score(pipe, Xc_tr, yc_tr, cv=cv_cls, scoring="f1")
    pipe.fit(Xc_tr, yc_tr)
    pred_te, pred_tr = pipe.predict(Xc_te), pipe.predict(Xc_tr)
    proba_te = pipe.predict_proba(Xc_te)[:, 1]
    cls_rows.append({"model": name, "complexity": rank,
                     "Accuracy": accuracy_score(yc_te, pred_te),
                     "Precision": precision_score(yc_te, pred_te, zero_division=0),
                     "Recall": recall_score(yc_te, pred_te),
                     "F1": f1_score(yc_te, pred_te),
                     "ROC_AUC": roc_auc_score(yc_te, proba_te),
                     "Log_Loss": log_loss(yc_te, proba_te),
                     "F1_train": f1_score(yc_tr, pred_tr),
                     "F1_gap(train-test)": f1_score(yc_tr, pred_tr) - f1_score(yc_te, pred_te),
                     "CV_F1_mean": cv_f1.mean(), "CV_F1_std": cv_f1.std()})
    cls_fitted[name] = pipe

cls_results = pd.DataFrame(cls_rows)
print(cls_results.drop(columns="complexity").round(3).to_string(index=False))
save_table(cls_results.drop(columns="complexity"), "09_classification_results", index=False)

best_cls = one_se_best(cls_results, "CV_F1_mean", "CV_F1_std")
print(f"\nBEST CLASSIFIER (CV F1 within one std of the best, then simplest): {best_cls}")
print("Large F1_gap = overfitting (high variance). Low F1 on both = underfitting (high bias).")

melted = cls_results.melt(id_vars="model", value_vars=["Precision", "Recall", "F1", "ROC_AUC"],
                          var_name="Metric", value_name="Score")
plt.figure(figsize=(13, 5))
sns.barplot(data=melted, x="model", y="Score", hue="Metric")
plt.title("Classification Models: Test-Set Metrics (positive class = at risk)")
plt.xlabel("Model")
plt.ylabel("Score (0-1)")
plt.xticks(rotation=35, ha="right")
save_fig("12_classification_comparison")

fig, ax = plt.subplots(figsize=(9, 7))
for name, pipe in cls_fitted.items():
    RocCurveDisplay.from_estimator(pipe, Xc_te, yc_te, name=name, ax=ax)
ax.plot([0, 1], [0, 1], "k--", label="Random guess")
ax.set_title("ROC Curves of All Classifiers")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.legend(fontsize=7)
save_fig("13_roc_curves")

fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay.from_estimator(cls_fitted[best_cls], Xc_te, yc_te, ax=ax,
                                      display_labels=["Not at risk", "At risk"], cmap="Blues")
ax.set_title(f"Confusion Matrix: {best_cls}")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
save_fig("14_confusion_matrix_best")
tn, fp, fn, tp = confusion_matrix(yc_te, cls_fitted[best_cls].predict(Xc_te)).ravel()
print(f"Confusion matrix of {best_cls}: TN={tn} FP={fp} FN={fn} TP={tp}")

# =============================================================================
# PART 8 - UNSUPERVISED LEARNING: K-MEANS, K-MEDOIDS, HIERARCHICAL CLUSTERING
# WHAT : group students by their three scores (no target used)
# UNIT : Unit IV (K-Means, K-Medoids, hierarchical, single/complete/average/
#        centroid linkage)
# OBJ  : OBJ5
# K-MEDOIDS: like K-Means but each cluster centre is a REAL student (the
#        medoid), and it uses Manhattan distance, so it is less sensitive to
#        outliers. It is written from scratch below (no extra package needed).
# =============================================================================
header("PART 8 - CLUSTERING (OBJ5)")
X_clu = StandardScaler().fit_transform(df[SCORE_COLS])     # scale: all on the same footing

# 8.1 Choose k for K-Means with the elbow (inertia) and silhouette score
ks = range(2, 9)
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_clu)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_clu, km.labels_))
k_table = pd.DataFrame({"k": list(ks), "inertia": inertias, "silhouette": sils})
print(k_table.round(3).to_string(index=False))
save_table(k_table, "10_kmeans_k_selection", index=False)
best_k = int(k_table.loc[k_table["silhouette"].idxmax(), "k"])
print(f"k chosen by highest silhouette score: {best_k}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(k_table["k"], k_table["inertia"], marker="o")
axes[0].set_title("Elbow Method for K-Means")
axes[0].set_xlabel("Number of clusters k")
axes[0].set_ylabel("Inertia (within-cluster sum of squares)")
axes[1].plot(k_table["k"], k_table["silhouette"], marker="o", color="green")
axes[1].set_title("Silhouette Score vs k")
axes[1].set_xlabel("Number of clusters k")
axes[1].set_ylabel("Silhouette score")
save_fig("15_kmeans_elbow_silhouette")

kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(X_clu)
df["cluster_kmeans"] = kmeans.labels_
profile = df.groupby("cluster_kmeans").agg(
    students=("math_score", "size"), math=("math_score", "mean"),
    reading=("reading_score", "mean"), writing=("writing_score", "mean"),
    test_prep_done=("test_prep", lambda s: (s == "completed").mean()),
    free_lunch=("lunch", lambda s: (s == "free/reduced").mean())).round(2)
print("\nK-Means cluster profile (averages):\n", profile)
save_table(profile, "11_kmeans_cluster_profile")
sil_kmeans = silhouette_score(X_clu, kmeans.labels_)


# 8.2 K-Medoids (Voronoi-iteration version of PAM, Manhattan distance)
def kmedoids(X, k, n_init=10, max_iter=100, seed=SEED):
    rng = np.random.default_rng(seed)
    D = squareform(pdist(X, metric="cityblock"))            # distance between every pair
    best = None
    for _ in range(n_init):
        medoids = rng.choice(len(X), k, replace=False)      # start: k random real students
        for _ in range(max_iter):
            labels = np.argmin(D[:, medoids], axis=1)       # assign to nearest medoid
            new = medoids.copy()
            for c in range(k):                              # new medoid = member with the
                members = np.where(labels == c)[0]          # smallest total distance to
                if len(members):                            # the other members
                    new[c] = members[np.argmin(D[np.ix_(members, members)].sum(axis=1))]
            if np.array_equal(np.sort(new), np.sort(medoids)):
                break
            medoids = new
        labels = np.argmin(D[:, medoids], axis=1)
        cost = D[np.arange(len(X)), medoids[labels]].sum()
        if best is None or cost < best[0]:
            best = (cost, medoids.copy(), labels)
    return best


cost, medoid_idx, med_labels = kmedoids(X_clu, best_k)
df["cluster_kmedoids"] = med_labels
sil_kmedoids = silhouette_score(X_clu, med_labels)
medoid_table = df.loc[medoid_idx, SCORE_COLS].reset_index().rename(columns={"index": "student_row"})
print(f"\nK-Medoids (k={best_k}): silhouette = {sil_kmedoids:.3f} | total Manhattan cost = {cost:.1f}")
print("Medoid students (the real students at the centre of each cluster):\n", medoid_table)
save_table(medoid_table, "12_kmedoids_medoids", index=False)

# 8.3 Hierarchical clustering with four linkage methods (+ Ward as a bonus)
hier_rows = []
for method in ["single", "complete", "average", "centroid", "ward"]:
    Z = linkage(X_clu, method=method)
    labels = fcluster(Z, t=best_k, criterion="maxclust")
    coph, _ = cophenet(Z, pdist(X_clu))
    hier_rows.append({"linkage": method, "cophenetic_corr": coph,
                      "silhouette": silhouette_score(X_clu, labels) if len(set(labels)) > 1 else np.nan,
                      "cluster_sizes": sorted(pd.Series(labels).value_counts().tolist(), reverse=True)})
hier_table = pd.DataFrame(hier_rows)
print(f"\nHierarchical clustering (cut at k = {best_k}):\n", hier_table.round(3).to_string(index=False))
save_table(hier_table, "13_hierarchical_linkage_comparison", index=False)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, method in zip(axes, ["complete", "ward"]):
    dendrogram(linkage(X_clu, method=method), truncate_mode="lastp", p=30, no_labels=True, ax=ax)
    ax.set_title(f"Dendrogram ({method} linkage, last 30 merges)")
    ax.set_xlabel("Merged groups of students")
    ax.set_ylabel("Merge distance")
save_fig("16_dendrograms")

clu_compare = pd.DataFrame([
    {"method": "K-Means", "silhouette": sil_kmeans},
    {"method": "K-Medoids", "silhouette": sil_kmedoids}] +
    [{"method": f"Hierarchical ({r['linkage']})", "silhouette": r["silhouette"]} for r in hier_rows])
print("\nClustering comparison (silhouette, higher = better separated):\n", clu_compare.round(3))
save_table(clu_compare, "14_clustering_comparison", index=False)
print("Silhouette near 0 = overlapping clusters. The three scores form one continuous "
      "spread, so do not over-claim 'distinct' groups. Single/centroid linkage that give "
      "one huge cluster are a known 'chaining' problem, not a good segmentation.")

# =============================================================================
# PART 9 - ASSOCIATION RULES (APRIORI) / MARKET-BASKET STYLE ANALYSIS
# WHAT : treat each student as a "basket" of attributes (gender, group, parent
#        education, lunch, test prep, math/reading/writing band) and find items
#        that occur together.
# UNIT : Unit IV (association rules, market basket analysis)
# OBJ  : OBJ5
# HONESTY NOTE: this dataset has NO purchase transactions. This is an
#        ADAPTATION: attributes play the role of items. State that in your report.
# TERMS: support = share of students containing the itemset;
#        confidence = P(consequent | antecedent);
#        lift = confidence / support(consequent). Lift > 1 = occur together more
#        than chance. Apriori is written from scratch below (no extra package).
# =============================================================================
header("PART 9 - ASSOCIATION RULES (OBJ5)")
basket = pd.get_dummies(df[["gender", "race_ethnicity", "parent_education", "lunch", "test_prep",
                            "math_band", "reading_band", "writing_band"]].astype(str),
                        prefix_sep="=").astype(bool)
MIN_SUPPORT, MIN_CONF, MIN_LIFT = 0.10, 0.60, 1.2
print(f"Basket matrix: {basket.shape[0]} students x {basket.shape[1]} items "
      f"| min_support={MIN_SUPPORT}, min_confidence={MIN_CONF}, min_lift={MIN_LIFT}")


def apriori_simple(onehot, min_support, max_len=4):
    """Level-wise Apriori: grow itemsets one item at a time, keep only frequent ones."""
    M, names = onehot.values, list(onehot.columns)
    support = {}
    level = []
    for j in range(M.shape[1]):
        s = M[:, j].mean()
        if s >= min_support:
            support[frozenset([j])] = s
            level.append(frozenset([j]))
    k = 2
    while level and k <= max_len:
        candidates = {a | b for a, b in combinations(level, 2) if len(a | b) == k}
        new_level = []
        for c in candidates:
            if any(frozenset(sub) not in support for sub in combinations(c, k - 1)):
                continue                      # Apriori rule: every subset must be frequent
            s = M[:, list(c)].all(axis=1).mean()
            if s >= min_support:
                support[c] = s
                new_level.append(c)
        level, k = new_level, k + 1
    return support, names


def make_rules(support, names, min_conf, min_lift):
    rows = []
    for itemset, s in support.items():
        if len(itemset) < 2:
            continue
        for cons in itemset:                  # rule form: (other items) -> one item
            ant = itemset - {cons}
            conf = s / support[ant]
            lift = conf / support[frozenset([cons])]
            if conf >= min_conf and lift >= min_lift:
                rows.append({"antecedent": " & ".join(sorted(names[i] for i in ant)),
                             "consequent": names[cons], "support": s,
                             "confidence": conf, "lift": lift})
    return pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)


sup, item_names = apriori_simple(basket, MIN_SUPPORT)
itemsets = pd.DataFrame([{"itemset": " & ".join(sorted(item_names[i] for i in s)), "size": len(s),
                          "support": v} for s, v in sup.items()]).sort_values("support", ascending=False)
rules = make_rules(sup, item_names, MIN_CONF, MIN_LIFT)
math_rules = rules[rules["consequent"].str.startswith("math_band")].reset_index(drop=True)
print(f"Frequent itemsets found: {len(itemsets)} | rules passing filters: {len(rules)} "
      f"| rules predicting a math band: {len(math_rules)}")
print("\nTop rules predicting math band (sorted by lift):")
print(math_rules.head(10).round(3).to_string(index=False))
save_table(itemsets, "15_frequent_itemsets", index=False)
save_table(rules, "16_association_rules_all", index=False)
save_table(math_rules, "17_association_rules_math", index=False)

if len(math_rules):
    top = math_rules.head(10).copy()
    top["rule"] = top["antecedent"] + "  ->  " + top["consequent"]
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top, x="lift", y="rule", color="steelblue")
    plt.axvline(1, color="red", linestyle="--", label="lift = 1 (no association)")
    plt.title("Top Association Rules Predicting Math Band (by lift)")
    plt.xlabel("Lift")
    plt.ylabel("Rule (antecedent -> consequent)")
    plt.legend()
    save_fig("17_association_rules_lift")

# =============================================================================
# PART 10 - PCA (DIMENSIONALITY REDUCTION)
# WHAT : squeeze the three scores into principal components
# UNIT : Unit V (PCA)
# OBJ  : OBJ5 (visualise the clusters in 2D)
# =============================================================================
header("PART 10 - PCA")
pca = PCA(random_state=SEED).fit(X_clu)
evr = pca.explained_variance_ratio_
print("Explained variance ratio per component:", np.round(evr, 3))
print("Cumulative:", np.round(np.cumsum(evr), 3))
loadings = pd.DataFrame(pca.components_.T, index=SCORE_COLS,
                        columns=[f"PC{i + 1}" for i in range(len(evr))]).round(3)
print("\nLoadings (how each score contributes to each component):\n", loadings)
save_table(loadings, "18_pca_loadings")

coords = PCA(n_components=2, random_state=SEED).fit_transform(X_clu)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(range(1, len(evr) + 1), evr)
axes[0].set_title("PCA: Variance Explained by Each Component")
axes[0].set_xlabel("Principal component")
axes[0].set_ylabel("Explained variance ratio")
sc = axes[1].scatter(coords[:, 0], coords[:, 1], c=df["cluster_kmeans"], cmap="viridis", alpha=0.7)
axes[1].set_title("Students in PCA Space (colour = K-Means cluster)")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].legend(*sc.legend_elements(), title="Cluster")
save_fig("18_pca")

# =============================================================================
# PART 11 - BIAS-VARIANCE TRADE-OFF AND ENSEMBLE SUMMARY
# WHAT : show underfitting vs overfitting with two model-complexity curves
# UNIT : Unit VI (bias-variance trade-off, ensembles)
# HOW TO READ: when train score is much higher than validation score, the model
#        has HIGH VARIANCE (overfits). When both are low, it has HIGH BIAS
#        (underfits). Ensembles (bagging, random forest) reduce variance;
#        boosting reduces bias.
# =============================================================================
header("PART 11 - BIAS-VARIANCE TRADE-OFF")
feat_b = ["reading_writing_avg", "parent_education"] + NOMINAL_COLS
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(df[feat_b], y_reg, test_size=0.2, random_state=SEED)

# 11.1 Regression: polynomial degree 1..6 (more degree = more flexible)
bv_rows = []
for d in range(1, 7):
    pipe = Pipeline([("prep", build_preprocessor(["reading_writing_avg"], poly=True, degree=d)),
                     ("model", LinearRegression())])
    cv = cross_val_score(pipe, Xb_tr, yb_tr, cv=cv_reg, scoring="r2")
    pipe.fit(Xb_tr, yb_tr)
    bv_rows.append({"degree": d, "train_R2": r2_score(yb_tr, pipe.predict(Xb_tr)),
                    "cv_R2": cv.mean()})
bv_poly = pd.DataFrame(bv_rows)
print("Polynomial degree vs R2:\n", bv_poly.round(4).to_string(index=False))
save_table(bv_poly, "19_bias_variance_polynomial", index=False)

# 11.2 Classification: decision tree depth 1..12 (deeper = more flexible)
tree_rows = []
for d in list(range(1, 13)):
    pipe = Pipeline([("prep", build_preprocessor(CLS_NUM)),
                     ("model", DecisionTreeClassifier(max_depth=d, class_weight="balanced",
                                                      random_state=SEED))])
    cv = cross_val_score(pipe, Xc_tr, yc_tr, cv=cv_cls, scoring="f1")
    pipe.fit(Xc_tr, yc_tr)
    tree_rows.append({"max_depth": d, "train_F1": f1_score(yc_tr, pipe.predict(Xc_tr)),
                      "cv_F1": cv.mean()})
bv_tree = pd.DataFrame(tree_rows)
print("\nDecision tree depth vs F1:\n", bv_tree.round(4).to_string(index=False))
save_table(bv_tree, "20_bias_variance_tree", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(bv_poly["degree"], bv_poly["train_R2"], marker="o", label="Training R2")
axes[0].plot(bv_poly["degree"], bv_poly["cv_R2"], marker="o", label="Cross-validated R2")
axes[0].set_title("Bias-Variance: Polynomial Degree (Regression)")
axes[0].set_xlabel("Polynomial degree (model complexity)")
axes[0].set_ylabel("R-squared")
axes[0].legend()
axes[1].plot(bv_tree["max_depth"], bv_tree["train_F1"], marker="o", label="Training F1")
axes[1].plot(bv_tree["max_depth"], bv_tree["cv_F1"], marker="o", label="Cross-validated F1")
axes[1].set_title("Bias-Variance: Decision Tree Depth (Classification)")
axes[1].set_xlabel("Maximum tree depth (model complexity)")
axes[1].set_ylabel("F1 score (at-risk class)")
axes[1].legend()
save_fig("19_bias_variance_curves")
tree_best_depth = int(bv_tree.loc[bv_tree["cv_F1"].idxmax(), "max_depth"])
print(f"Tree depth with best cross-validated F1: {tree_best_depth}")

# =============================================================================
# PART 12 - FINDINGS (objective by objective) AND SUMMARY FILE
# WHAT : print sentences built from the computed results, save summary.json
# UNIT : Unit I-VI (interpretation)
# NOTE : every number below is computed, never typed in by hand.
# =============================================================================
header("PART 12 - OBJECTIVE-WISE FINDINGS")
tp_row = ttest_table.set_index(["factor", "subject"])
print("OBJ1 - Background factors (associations only, not causes):")
for factor in ["test_prep", "lunch", "gender"]:
    r = tp_row.loc[(factor, "math_score")]
    print(f"  * {factor}: math mean {r['group_1']}={r['mean_group_1']:.1f} vs {r['group_2']}="
          f"{r['mean_group_2']:.1f} (difference {r['difference']:+.1f}, p={r['p_value']:.2g})")
edu_math = group_tables["parent_education"]["math_score"]
print(f"  * parent education: math mean rises from {edu_math.min():.1f} "
      f"({edu_math.idxmin()}) to {edu_math.max():.1f} ({edu_math.idxmax()})")
race_math = group_tables["race_ethnicity"]["math_score"]
print(f"  * race/ethnicity group means range {race_math.min():.1f} ({race_math.idxmin()}) "
      f"to {race_math.max():.1f} ({race_math.idxmax()}) (groups are anonymised labels)")

print("\nOBJ2 - Correlation between subjects:")
print(f"  * math-reading r={score_corr.loc['math_score', 'reading_score']:.3f}, "
      f"math-writing r={score_corr.loc['math_score', 'writing_score']:.3f}, "
      f"reading-writing r={score_corr.loc['reading_score', 'writing_score']:.3f}")

best_row = res_b.set_index("model").loc[best_reg]
print("\nOBJ3 - Predicting math score:")
print(f"  * background factors only: best test R2 = {res_a['R2_test'].max():.3f}")
print(f"  * with reading/writing average: best-fit model = {best_reg} "
      f"(test R2={best_row['R2_test']:.3f}, RMSE={best_row['RMSE']:.2f}, MAE={best_row['MAE']:.2f})")
print(f"  * simple regression on reading alone: R2 = {slr.rvalue ** 2:.3f}")

cb = cls_results.set_index("model").loc[best_cls]
print("\nOBJ4 - Identifying at-risk students:")
print(f"  * best-fit classifier = {best_cls}: recall={cb['Recall']:.3f}, precision={cb['Precision']:.3f}, "
      f"F1={cb['F1']:.3f}, ROC-AUC={cb['ROC_AUC']:.3f}")
print(f"  * on the test set it caught {tp} of {tp + fn} at-risk students and raised {fp} false alarms")

print("\nOBJ5 - Groups and co-occurring attributes:")
print(f"  * K-Means k={best_k}: silhouette {sil_kmeans:.3f}; K-Medoids silhouette {sil_kmedoids:.3f}")
print(f"  * PCA: PC1 explains {evr[0]:.1%} of variance (the scores move together)")
if len(math_rules):
    r0 = math_rules.iloc[0]
    print(f"  * strongest rule predicting a math band: [{r0['antecedent']}] -> {r0['consequent']} "
          f"(support {r0['support']:.2f}, confidence {r0['confidence']:.2f}, lift {r0['lift']:.2f})")

SUMMARY.update({
    "best_regression_model": best_reg, "highest_r2_regression_model": top_reg,
    "best_classifier": best_cls, "best_k": best_k,
    "kmeans_silhouette": sil_kmeans, "kmedoids_silhouette": sil_kmedoids,
    "pca_explained_variance": [float(x) for x in evr],
    "confusion_best": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    "n_rules": int(len(rules)), "n_math_rules": int(len(math_rules)),
    "n_itemsets": int(len(itemsets)), "tree_best_depth": tree_best_depth,
    "regression_best_set_a_r2": float(res_a["R2_test"].max()),
    "score_corr": {"math_reading": float(score_corr.loc["math_score", "reading_score"]),
                   "math_writing": float(score_corr.loc["math_score", "writing_score"]),
                   "reading_writing": float(score_corr.loc["reading_score", "writing_score"])},
})
(TAB_DIR / "summary.json").write_text(json.dumps(SUMMARY, indent=2, default=float))
print("\nAll figures saved in:", FIG_DIR)
print("All tables  saved in:", TAB_DIR)
print("\nLIMITATIONS to write in your report: 1000 students from a single source, no time "
      "dimension, only 5 background variables, scores are highly inter-correlated, "
      "PASS_MARK is an assumption, association rules use attributes (not real "
      "transactions), and associations here are not causal.")