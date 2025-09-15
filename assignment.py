# ============================================
# 📘 Assignment: Data Analysis with Pandas & Matplotlib
# Objective:
# 1. Load and analyze a dataset using pandas.
# 2. Create simple plots and charts using matplotlib (and seaborn for styling).
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

def main():
    # -------------------------------
    # Task 1: Load and Explore Dataset
    # -------------------------------
    print("🔹 Loading dataset...")
    try:
        # Load Iris dataset from sklearn
        iris_data = load_iris(as_frame=True)
        df = iris_data.frame
        df['species'] = df['target'].map(dict(enumerate(iris_data.target_names)))
        print("✅ Dataset loaded successfully!\n")
    except FileNotFoundError:
        print("❌ Error: Dataset file not found.")
        return
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return

    # Display first few rows
    print("📌 First 5 rows of dataset:")
    print(df.head())

    # Dataset info
    print("\n📌 Dataset Info:")
    print(df.info())

    # Check for missing values
    print("\n📌 Missing values per column:")
    print(df.isnull().sum())

    # Clean dataset (if missing values exist → fill with mean)
    if df.isnull().sum().any():
        df.fillna(df.mean(), inplace=True)

    # -------------------------------
    # Task 2: Basic Data Analysis
    # -------------------------------
    print("\n📊 Descriptive statistics:")
    print(df.describe())

    # Grouping: mean petal length by species
    grouped = df.groupby('species')['petal length (cm)'].mean()
    print("\n📊 Average petal length per species:")
    print(grouped)

    # -------------------------------
    # Task 3: Data Visualization
    # -------------------------------
    sns.set_style("whitegrid")

    # 1️⃣ Line Chart – Trend (fake time series using index)
    plt.figure(figsize=(8,5))
    plt.plot(df.index, df['sepal length (cm)'], label="Sepal Length", color="blue")
    plt.title("Trend of Sepal Length Over Samples")
    plt.xlabel("Sample Index")
    plt.ylabel("Sepal Length (cm)")
    plt.legend()
    plt.show()

    # 2️⃣ Bar Chart – Average petal length per species
    plt.figure(figsize=(8,5))
    grouped.plot(kind="bar", color=["#FF9999", "#66B2FF", "#99FF99"])
    plt.title("Average Petal Length per Species")
    plt.xlabel("Species")
    plt.ylabel("Avg Petal Length (cm)")
    plt.show()

    # 3️⃣ Histogram – Distribution of sepal width
    plt.figure(figsize=(8,5))
    plt.hist(df['sepal width (cm)'], bins=15, color="purple", edgecolor="black")
    plt.title("Distribution of Sepal Width")
    plt.xlabel("Sepal Width (cm)")
    plt.ylabel("Frequency")
    plt.show()

    # 4️⃣ Scatter Plot – Sepal length vs Petal length
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x="sepal length (cm)", y="petal length (cm)", hue="species", palette="Set1")
    plt.title("Sepal Length vs Petal Length")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.legend(title="Species")
    plt.show()

    # -------------------------------
    # Findings & Observations
    # -------------------------------
    print("\n📌 Observations:")
    print("- The Iris dataset has no missing values.")
    print("- Different species show distinct average petal lengths.")
    print("- Sepal width follows a roughly normal distribution.")
    print("- There is a clear positive relationship between sepal length and petal length, varying by species.")

if __name__ == "__main__":
    main()
