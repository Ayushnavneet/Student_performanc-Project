import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, auc,
    confusion_matrix, classification_report
)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

print("=" * 80)
print("--- [1] LOADING DATA & INITIAL INSPECTION ---")
print("=" * 80)
df = pd.read_csv('StudentsPerformance.csv')
print(f"Dataset Shape: {df.shape}")
print(df.info())

print("\nMissing Values Count:\n", df.isnull().sum())
print(f"\nDuplicate Rows Count: {df.duplicated().sum()}")

print("\n" + "=" * 80)
print("--- [2] PERFORMING FEATURE ENGINEERING ---")
print("=" * 80)

df['total_score'] = df['math score'] + df['reading score'] + df['writing score']
df['average_score'] = df['total_score'] / 3.0

df['stem_vs_humanities_diff'] = df['math score'] - ((df['reading score'] + df['writing score']) / 2.0)

education_hierarchy = {
    'some high school': 1,
    'high school': 2,
    'some college': 3,
    "associate's degree": 4,
    "bachelor's degree": 5,
    "master's degree": 6
}
df['parental_edu_tier'] = df['parental level of education'].map(education_hierarchy)

df['is_standard_lunch'] = (df['lunch'] == 'standard').astype(int)
df['prep_completed'] = (df['test preparation course'] == 'completed').astype(int)
df['is_female'] = (df['gender'] == 'female').astype(int)

df['support_index'] = (
    (df['parental_edu_tier'] / 6.0) * 0.4 +
    df['is_standard_lunch'] * 0.4 +
    df['prep_completed'] * 0.2
)

df['high_edu_std_lunch'] = (
    (df['parental_edu_tier'] >= 5) & (df['is_standard_lunch'] == 1)
).astype(int)

df['pass_status'] = (df['average_score'] >= 60.0).astype(int)
df['math_pass'] = (df['math score'] >= 60.0).astype(int)
df['reading_pass'] = (df['reading score'] >= 60.0).astype(int)
df['writing_pass'] = (df['writing score'] >= 60.0).astype(int)
df['all_subjects_pass'] = (
    (df['math_pass'] == 1) & (df['reading_pass'] == 1) & (df['writing_pass'] == 1)
).astype(int)

def assign_letter_grade(score):
    if score >= 80: return 'A'
    elif score >= 70: return 'B'
    elif score >= 60: return 'C'
    elif score >= 50: return 'D'
    else: return 'F'

df['letter_grade'] = df['average_score'].apply(assign_letter_grade)

print("Engineered Features Sample:")
print(df[['average_score', 'stem_vs_humanities_diff', 'parental_edu_tier', 'support_index', 'high_edu_std_lunch', 'pass_status', 'letter_grade']].head())

print("\n" + "=" * 80)
print("--- [3] STATISTICAL HYPOTHESIS TESTING FOR THE 5 OBJECTIVES ---")
print("=" * 80)

m_math, f_math = df[df['gender'] == 'male']['math score'].mean(), df[df['gender'] == 'female']['math score'].mean()
m_read, f_read = df[df['gender'] == 'male']['reading score'].mean(), df[df['gender'] == 'female']['reading score'].mean()
m_write, f_write = df[df['gender'] == 'male']['writing score'].mean(), df[df['gender'] == 'female']['writing score'].mean()
t_math = stats.ttest_ind(df[df['gender'] == 'male']['math score'], df[df['gender'] == 'female']['math score'])
t_read = stats.ttest_ind(df[df['gender'] == 'male']['reading score'], df[df['gender'] == 'female']['reading score'])
t_write = stats.ttest_ind(df[df['gender'] == 'male']['writing score'], df[df['gender'] == 'female']['writing score'])

print(f"Objective 1: Gender Disparities Across Academic Subjects")
print(f"  * Math:    Male = {m_math:.2f}, Female = {f_math:.2f} | t = {t_math.statistic:.2f}, p-val = {t_math.pvalue:.4e}")
print(f"  * Reading: Male = {m_read:.2f}, Female = {f_read:.2f} | t = {t_read.statistic:.2f}, p-val = {t_read.pvalue:.4e}")
print(f"  * Writing: Male = {m_write:.2f}, Female = {f_write:.2f} | t = {t_write.statistic:.2f}, p-val = {t_write.pvalue:.4e}")

prep_comp = df[df['test preparation course'] == 'completed']['average_score']
prep_none = df[df['test preparation course'] == 'none']['average_score']
t_prep = stats.ttest_ind(prep_comp, prep_none)
print(f"\nObjective 2: Test Preparation Course Impact")
print(f"  * Mean Score: Completed = {prep_comp.mean():.2f}, None = {prep_none.mean():.2f} (Gain = +{prep_comp.mean()-prep_none.mean():.2f})")
print(f"  * Two-sample t-test: t = {t_prep.statistic:.2f}, p-val = {t_prep.pvalue:.4e}")

lunch_std = df[df['lunch'] == 'standard']['average_score']
lunch_free = df[df['lunch'] == 'free/reduced']['average_score']
t_lunch = stats.ttest_ind(lunch_std, lunch_free)
print(f"\nObjective 3: Socioeconomic Status (Lunch Type)")
print(f"  * Mean Score: Standard = {lunch_std.mean():.2f}, Free/Reduced = {lunch_free.mean():.2f} (Gap = {lunch_std.mean()-lunch_free.mean():.2f})")
print(f"  * Two-sample t-test: t = {t_lunch.statistic:.2f}, p-val = {t_lunch.pvalue:.4e}")

f_edu = stats.f_oneway(*[group['average_score'].values for _, group in df.groupby('parental level of education')])
edu_summary = df.groupby('parental level of education')['average_score'].agg(['count', 'mean', 'std']).round(2)
print(f"\nObjective 4: Parental Educational Attainment")
print(f"  * One-Way ANOVA: F = {f_edu.statistic:.2f}, p-val = {f_edu.pvalue:.4e}")
print(edu_summary.to_string())

corr_rw = stats.pearsonr(df['reading score'], df['writing score'])
corr_mr = stats.pearsonr(df['math score'], df['reading score'])
corr_mw = stats.pearsonr(df['math score'], df['writing score'])
print(f"\nObjective 5: Pearson Linear Correlations")
print(f"  * Reading vs Writing: r = {corr_rw[0]:.4f} (p-val = {corr_rw[1]:.4e})")
print(f"  * Math vs Reading:    r = {corr_mr[0]:.4f} (p-val = {corr_mr[1]:.4e})")
print(f"  * Math vs Writing:    r = {corr_mw[0]:.4f} (p-val = {corr_mw[1]:.4e})")
print("=" * 80)

print("\n--- Displaying Visualizations One by One ---")

# Graph 1: Objective 1 - Subject Performance by Gender
plt.figure(figsize=(8, 6))
melted_df = df.melt(id_vars=['gender'], value_vars=['math score', 'reading score', 'writing score'], 
                    var_name='Subject', value_name='Score')
ax1 = sns.barplot(data=melted_df, x='Subject', y='Score', hue='gender', errorbar=None, palette=['#4A90E2', '#E94E77'])
plt.title("Objective 1: Subject Performance by Gender", fontsize=13, fontweight='bold', pad=12)
plt.ylabel("Average Score (0 - 100)", fontsize=11)
plt.xlabel("Subject", fontsize=11)
plt.ylim(0, 100)
for p in ax1.patches:
    h = p.get_height()
    if h > 0:
        ax1.annotate(f"{h:.1f}", (p.get_x() + p.get_width() / 2., h - 8),
                     ha='center', va='center', color='white', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.show()

# Graph 2: Objective 2 - Test Preparation Impact
plt.figure(figsize=(8, 6))
sns.boxplot(x='test preparation course', y='average_score', data=df, palette=['#7F8C8D', '#27AE60'], width=0.45)
plt.title("Objective 2: Test Preparation Impact on Overall Score", fontsize=13, fontweight='bold', pad=12)
plt.ylabel("Overall Average Score", fontsize=11)
plt.xlabel("Test Preparation Status", fontsize=11)
plt.tight_layout()
plt.show()

# Graph 3: Objective 3 - Density Distribution by Lunch Type
plt.figure(figsize=(8, 6))
sns.kdeplot(data=df, x='average_score', hue='lunch', fill=True, common_norm=False, palette=['#2980B9', '#E67E22'])
plt.title("Objective 3: Density Distribution by Lunch Type (Socioeconomic)", fontsize=13, fontweight='bold', pad=12)
plt.ylabel("Density", fontsize=11)
plt.xlabel("Average Score", fontsize=11)
plt.tight_layout()
plt.show()

# Graph 4: Objective 4 - Academic Performance by Parental Education
plt.figure(figsize=(9, 6))
order_edu = ["some high school", "high school", "some college", 
             "associate's degree", "bachelor's degree", "master's degree"]
edu_df = df.groupby('parental level of education')['average_score'].mean().loc[order_edu].reset_index()
ax4 = sns.barplot(x='average_score', y='parental level of education', data=edu_df, palette="crest")
plt.title("Objective 4: Academic Performance by Parental Education", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Mean Overall Average Score", fontsize=11)
plt.ylabel("Parental Educational Attainment", fontsize=11)
plt.xlim(0, 85)
for p in ax4.patches:
    w = p.get_width()
    ax4.annotate(f"{w:.2f}", (w + 1.2, p.get_y() + p.get_height() / 2.),
                 ha='left', va='center', color='black', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

# Graph 5: Objective 5A - Linear Regression Fit
plt.figure(figsize=(8, 6))
sns.regplot(data=df, x='reading score', y='writing score',
            scatter_kws={'alpha': 0.35, 'color': '#2C3E50'}, line_kws={'color': '#C0392B', 'linewidth': 2})
plt.title("Objective 5: Correlation Regression Fit (Reading vs Writing, r = 0.95)", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Reading Score", fontsize=11)
plt.ylabel("Writing Score", fontsize=11)
plt.tight_layout()
plt.show()

# Graph 6: Objective 5B - Pearson Correlation Heatmap
plt.figure(figsize=(8, 6))
corr = df[['math score', 'reading score', 'writing score', 'average_score']].corr()
sns.heatmap(corr, annot=True, cmap='Blues', fmt=".3f", linewidths=0.5, cbar=False)
plt.title("Objective 5: Pearson Correlation Matrix Across Exam Dimensions", fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.show()

# Graph 7: Engineered Feature 1 - STEM vs Humanities Disparity
plt.figure(figsize=(8, 6))
sns.kdeplot(data=df, x='stem_vs_humanities_diff', hue='gender', fill=True, palette=['#E94E77', '#4A90E2'])
plt.axvline(0, color='black', linestyle='--', alpha=0.7)
plt.title("Engineered Feature 1: STEM vs. Humanities Disparity", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Math Score minus (Reading + Writing Average)", fontsize=11)
plt.ylabel("Density", fontsize=11)
plt.tight_layout()
plt.show()

# Graph 8: Engineered Feature 2 - Support Index vs Average Score
plt.figure(figsize=(8, 6))
sns.regplot(data=df, x='support_index', y='average_score',
            scatter_kws={'alpha': 0.3, 'color': '#2C3E50'}, line_kws={'color': '#E67E22', 'linewidth': 2})
plt.title("Engineered Feature 2: Support Index vs Average Score", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Socioeconomic & Academic Support Index (0 - 1)", fontsize=11)
plt.ylabel("Overall Average Score", fontsize=11)
plt.tight_layout()
plt.show()

# Graph 9: Engineered Feature 3 - Discrete Grade Tiers
plt.figure(figsize=(8, 6))
grade_order = ['A', 'B', 'C', 'D', 'F']
ax9 = sns.countplot(data=df, x='letter_grade', order=grade_order, palette="mako")
plt.title("Engineered Feature 3: Discrete Grade Tiers", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Grade (A: >=80, B: 70-79, C: 60-69, D: 50-59, F: <50)", fontsize=11)
plt.ylabel("Number of Students", fontsize=11)
for p in ax9.patches:
    h = p.get_height()
    ax9.annotate(f"{h}", (p.get_x() + p.get_width() / 2., h + 3),
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

predictors = [
    'is_female', 'parental_edu_tier', 'is_standard_lunch', 
    'prep_completed', 'support_index', 'high_edu_std_lunch', 'race/ethnicity'
]
X_raw = df[predictors]
y = df['pass_status']

X = pd.get_dummies(X_raw, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Gaussian Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=4),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []
roc_curves = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    model_auc = auc(fpr, tpr)
    roc_curves[name] = (fpr, tpr, model_auc)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': model_auc
    })

results_df = pd.DataFrame(results)
print("\n" + "=" * 70)
print("MACHINE LEARNING MODEL BENCHMARK RESULTS")
print("=" * 70)
print(results_df.to_string(index=False))

# Graph 10: Dedicated Algorithm Comparison Graph
plt.figure(figsize=(14, 7))
melted_results = results_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
metric_palette = ['#2b5c8f', '#e67e22', '#27ae60', '#9b59b6', '#e74c3c']

ax10 = sns.barplot(data=melted_results, x='Model', y='Score', hue='Metric', palette=metric_palette)
plt.title("Machine Learning Algorithms Performance Comparison (INT234 Metrics)", 
          fontsize=14, fontweight='bold', pad=15)
plt.ylabel("Metric Score (0.0 to 1.0)", fontsize=11, fontweight='bold')
plt.xlabel("Machine Learning Algorithm", fontsize=11, fontweight='bold')
plt.ylim(0.4, 1.0)
plt.xticks(rotation=15, ha='right', fontsize=10)
plt.legend(title="Evaluation Metric", loc='upper left', frameon=True, fontsize=10)
for p in ax10.patches:
    val = p.get_height()
    if val > 0:
        ax10.annotate(f"{val:.2f}",
                     (p.get_x() + p.get_width() / 2., val + 0.008),
                     ha='center', va='bottom', fontsize=8, rotation=90)
plt.tight_layout()
plt.show()

# Graph 11: Multi-Model ROC Curves Comparison
plt.figure(figsize=(8, 6))
for name, (fpr, tpr, model_auc) in roc_curves.items():
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {model_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=11)
plt.ylabel('True Positive Rate', fontsize=11)
plt.title('ROC Curves Comparison (Unit III AUC)', fontsize=13, fontweight='bold', pad=12)
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.show()

# Graph 12: Random Forest Feature Importance
plt.figure(figsize=(8, 6))
rf_model = models['Random Forest']
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)
importances.plot(kind='barh', color='#3498db')
plt.title("Feature Importance with Engineered Predictors", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Relative Importance", fontsize=11)
plt.tight_layout()
plt.show()

best_model_name = results_df.sort_values(by=['F1 Score', 'ROC AUC'], ascending=False).iloc[0]['Model']
print(f"\n>>> Best Performing Model: {best_model_name}")

best_model = models[best_model_name]
best_pred = best_model.predict(X_test_scaled)

# Graph 13: Best Model Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fail (<60%)', 'Pass (>=60%)'],
            yticklabels=['Fail (<60%)', 'Pass (>=60%)'],
            cbar=False)
plt.title(f"Confusion Matrix: {best_model_name}", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Predicted Label", fontsize=11)
plt.ylabel("True Label", fontsize=11)
plt.tight_layout()
plt.show()

print("\nClassification Report for Best Model:")
print(classification_report(y_test, best_pred, target_names=['Fail', 'Pass']))
