📊 Week 7 Python Data Analysis Assignment
🎯 Objective

The purpose of this assignment is to:

Load and analyze a dataset using the pandas library in Python.

Perform simple statistical analysis and data exploration.

Create visualizations using matplotlib (and optionally seaborn).

📂 Project Structure

Week 7/
│── assignment.py # Main Python script with code for the assignment
│── dataset.csv # Dataset file used in the project (replace with your dataset)
│── README.md # Project documentation

⚙️ Requirements

Make sure you have the following Python libraries installed:

pandas

matplotlib

seaborn (optional, for better visuals)

scikit-learn (optional, if using Iris dataset)

Install dependencies with:
pip install pandas matplotlib seaborn scikit-learn

▶️ How to Run

Clone or download this project folder.

Open a terminal inside the project directory.

Run the script:
python assignment.py

📌 Tasks Covered

Task 1: Load and Explore the Dataset

Load dataset from CSV using pandas.read_csv().

Display first few rows with .head().

Explore data types, missing values, and clean the dataset.

Task 2: Basic Data Analysis

Generate summary statistics using .describe().

Perform grouping operations and find mean per category.

Identify patterns or insights.

Task 3: Data Visualization

Line chart → Show trends over time.

Bar chart → Compare values across categories.

Histogram → Visualize data distribution.

Scatter plot → Explore relationship between two numerical variables.
Each plot is customized with titles, axis labels, and legends.

📊 Example Outputs

Data preview (via .head())

Summary statistics (via .describe())

4 different plots (Line, Bar, Histogram, Scatter)

🛡 Error Handling

File not found → handled using try/except.

Missing data → cleaned by filling/dropping.
