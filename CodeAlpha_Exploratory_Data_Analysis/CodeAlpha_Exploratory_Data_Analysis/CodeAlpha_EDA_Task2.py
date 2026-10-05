# =====================================================================
# PROJECT: Exploratory Data Analysis (EDA) - Titanic Dataset
# INTERNSHIP: CodeAlpha Data Analytics
# TASK: Task 2 - Exploratory Data Analysis (EDA)
# AUTHOR: [Your Name]
# DATE: [Current Date]
# =====================================================================

# ---------------------------------------------------------------------
# SECTION 1: IMPORT LIBRARIES & SETUP
# ---------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style for professional look
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)

print("="*70)
print("TASK 2: EXPLORATORY DATA ANALYSIS (EDA) - TITANIC DATASET")
print("="*70)

# ---------------------------------------------------------------------
# SECTION 2: ASK MEANINGFUL QUESTIONS (Pre-Analysis)
# ---------------------------------------------------------------------
print("\n--- 2. ASK MEANINGFUL QUESTIONS ---")
print("1. Demographics: What was the distribution of age and gender?")
print("2. Survival Factors: Did gender, class, or age impact survival?")
print("3. Data Quality: Are there missing values or outliers?")
print("4. Relationships: Is there a correlation between Fare, Class, and Survival?")

# ---------------------------------------------------------------------
# SECTION 3: LOAD DATA & EXPLORE STRUCTURE
# ---------------------------------------------------------------------
print("\n--- 3. EXPLORE DATA STRUCTURE ---")
# NOTE: Replace 'titanic.csv' with your actual dataset path
try:
    df = pd.read_csv('titanic.csv')
    print("Dataset loaded successfully from 'titanic.csv'!")
except FileNotFoundError:
    print("Warning: 'titanic.csv' not found. Using a built-in dummy dataset for demonstration.")
    # Creating a dummy dataset for demonstration if file not found
    data = {
        'Survived': [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1],
        'Pclass': [3, 1, 3, 1, 3, 3, 1, 3, 3, 2, 3, 1, 2, 3, 1],
        'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'female', 'male', 'female', 'male', 'male', 'female'],
        'Age': [22, 38, 26, 35, 35, np.nan, 54, 2, 27, 14, 4, 58, 20, 39, 55],
        'SibSp': [1, 1, 0, 1, 0, 0, 0, 3, 0, 1, 1, 0, 0, 1, 0],
        'Parch': [0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 0, 0, 5, 0],
        'Fare': [7.25, 71.28, 7.92, 53.10, 8.05, 8.45, 51.86, 21.07, 11.13, 30.07, 16.7, 26.55, 8.05, 31.27, 7.25],
        'Embarked': ['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', 'C', 'S', 'S', 'S', 'S', 'S']
    }
    df = pd.DataFrame(data)

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Statistical Summary ---")
print(df.describe())

# ---------------------------------------------------------------------
# SECTION 4: DATA CLEANING & DETECTING ISSUES
# ---------------------------------------------------------------------
print("\n--- 4. DETECTING DATA ISSUES ---")
print("Missing Values per Column:")
print(df.isnull().sum())

# Handling Missing Values
# Strategy: Fill missing Age with median, drop Cabin (too many missing), fill Embarked with mode
if 'Age' in df.columns:
    df['Age'] = df['Age'].fillna(df['Age'].median())
if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
if 'Cabin' in df.columns:
    df.drop(columns=['Cabin'], inplace=True)

print(f"\nDuplicate Rows: {df.duplicated().sum()}")
print("Data cleaning complete. Remaining missing values:", df.isnull().sum().sum())

# ---------------------------------------------------------------------
# SECTION 5: UNIVARIATE ANALYSIS (Trends & Patterns)
# ---------------------------------------------------------------------
print("\n--- 5. UNIVARIATE ANALYSIS ---")
print("Generating plots... Please close the plot windows to continue.")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot for Survival
sns.countplot(x='Survived', data=df, ax=axes[0], palette='viridis')
axes[0].set_title('Distribution of Survival (0 = Died, 1 = Survived)')
axes[0].set_ylabel('Count')

# Count plot for Gender
sns.countplot(x='Sex', data=df, ax=axes[1], palette='magma')
axes[1].set_title('Distribution of Gender')
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.show()

# Age Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['Age'], bins=30, kde=True, color='teal')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.show()

# Fare Boxplot for Outliers
plt.figure(figsize=(10, 3))
sns.boxplot(x=df['Fare'], color='salmon')
plt.title('Boxplot of Fare (Detecting Outliers)')
plt.show()

# ---------------------------------------------------------------------
# SECTION 6: BIVARIATE ANALYSIS (Testing Hypotheses)
# ---------------------------------------------------------------------
print("\n--- 6. BIVARIATE ANALYSIS (HYPOTHESIS TESTING) ---")
print("Hypothesis 1: 'Women and children first' was a real protocol.")
print("Hypothesis 2: Passengers in higher classes (1st Class) had a better chance of survival.")

# Analysis 1: Survival by Gender
plt.figure(figsize=(8, 5))
sns.barplot(x='Sex', y='Survived', data=df, palette='coolwarm', errorbar=None)
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Probability')
plt.show()

# Analysis 2: Survival by Passenger Class
plt.figure(figsize=(8, 5))
sns.barplot(x='Pclass', y='Survived', data=df, palette='Blues', errorbar=None)
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Probability')
plt.show()

# ---------------------------------------------------------------------
# SECTION 7: MULTIVARIATE ANALYSIS (Correlations)
# ---------------------------------------------------------------------
print("\n--- 7. MULTIVARIATE ANALYSIS (CORRELATION) ---")
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='RdBu', fmt=".2f")
plt.title('Correlation Matrix of Numerical Variables')
plt.show()

# ---------------------------------------------------------------------
# SECTION 8: CONCLUSION & RECOMMENDATIONS
# ---------------------------------------------------------------------
print("\n" + "="*70)
print("CONCLUSION & RECOMMENDATIONS")
print("="*70)
print("1. Key Drivers: Gender and Passenger Class are the strongest predictors of survival.")
print("2. Data Quality: Missing Age values were imputed with median. Cabin column was dropped.")
print("3. Next Steps: Create a 'Family Size' feature (SibSp + Parch) for better predictive modeling.")
print("="*70)