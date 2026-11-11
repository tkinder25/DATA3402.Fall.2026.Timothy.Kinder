#import libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from IPython.display import HTML, display
import tabulate
import sklearn
import sklearn.metrics as metrics
import seaborn as sns

varNames_og = ["Name", "Gender", "Age", "City", "Working Professional or Student", 
            "Profession", "Academic Pressure", "Work Pressure", "CGPA", "Study Satisfaction",	
            "Job Satisfaction",	"Sleep Duration", "Dietary Habits",	"Degree", "Have you ever had suicidal thoughts ?",
            "Work/Study Hours",	"Financial Stress",	"Family History of Mental Illness",	"Depression"]

df = pd.read_csv("final_depression_dataset_1.csv", names=varNames_og, header=0) #real dataset to be used for final predictions
df.map


# helper functions 

#checks missing and duplicate values in the dataframe and prints the results
def check_missing_values(df):

    print("\nMISSING VALUES")
    missing = pd.DataFrame({"missing_count": df.isna().sum(), 
                            "missing_percent": (df.isna().mean() * 100).round(1),})
    
    display(missing.sort_values("missing_count", ascending=False))

    print(f"\nDuplicate rows: {df.duplicated().sum():,}")

    #print num of students and num of working professionals
    print(df["Working Professional or Student"].value_counts())

def pair_plots(df, numeric_cols):
    MAX_PAIRPLOT_FEATURES = 5
    MAX_PAIRPLOT_ROWS = 1500
    TARGET = "Depression"

    pair_cols = numeric_cols[:MAX_PAIRPLOT_FEATURES]
    print("\nPair-plot columns:", pair_cols)

    if len(pair_cols) >= 2:
        plot_cols = pair_cols.copy()
        use_hue = False

        if TARGET in df.columns and df[TARGET].nunique(dropna=True) <= 10:
            plot_cols.append(TARGET)
            use_hue = True

        plot_df = df[plot_cols].dropna()
        if len(plot_df) > MAX_PAIRPLOT_ROWS:
            plot_df = plot_df.sample(MAX_PAIRPLOT_ROWS, random_state=42)

        if use_hue:
            g = sns.pairplot(plot_df, vars=pair_cols, hue=TARGET, corner=True, diag_kind="hist",
                            plot_kws={"alpha": 0.6, "s": 28})
        else:
            g = sns.pairplot(plot_df, vars=pair_cols, corner=True, diag_kind="hist",
                            plot_kws={"alpha": 0.6, "s": 28})

            g.fig.suptitle("Pair plot of selected numerical features", y=1.02)
            plt.show()
    else:
        print("Need at least two numerical features for a pair plot.")

def find_unique_values(df, col):
    unique_values = df[col].unique()
    print(f"Column '{col}' has {len(unique_values)} unique values: {unique_values}")
    return unique_values #return unique values so it can be used to create a dictionary of unique values to clean data.

# find_unique_values(df)