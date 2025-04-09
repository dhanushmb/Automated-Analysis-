import os
import sys
import pandas as pd
import seab as sns
import matplotlib.pyplot as plt
import openai
import warnings
from scipy.stats import zscore

warnings.filterwarnings("ignore", category=UserWarning)

# Check if dataset is provided
if len(sys.argv) < 2:
    print("Usage: uv run autolysis.py Data/happiness.csv")
    sys.exit(1)

filename = os.path.join(os.getcwd(), sys.argv[1])

# Retrieve OpenAI API Token
API_KEY = os.getenv("AIPROXY_TOKEN")
if not API_KEY:
    print("Error: AIPROXY_TOKEN is not set in environment variables.")
    sys.exit(1)

client = openai.OpenAI(api_key=API_KEY)

def load_data(file):
    """Load dataset with encoding handling."""
    if not os.path.exists(file):
        print(f"Error: File {file} not found.")
        sys.exit(1)
    try:
        df = pd.read_csv(file, encoding="utf-8")
        return df
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file, encoding="ISO-8859-1")
            return df
        except Exception as e:
            print(f"Error loading dataset from {file}: {e}")
            sys.exit(1)

def analyze_data(df):
    """Perform basic data analysis."""
    summary = df.describe(include='all').to_string()
    missing_values = df.isnull().sum().to_string()
    numeric_df = df.select_dtypes(include=['number'])
    correlations = numeric_df.corr().to_string()
    outliers = detect_outliers(numeric_df)
    return summary, missing_values, correlations, outliers

def detect_outliers(df):
    """Detect outliers using Z-score."""
    z_scores = df.apply(zscore)
    outliers = (z_scores.abs() > 3).sum().to_string()
    return outliers

def get_ai_insights(summary, missing_values, correlations, outliers, sample_data):
    """Use GPT-4o-Mini to generate insights."""
    prompt = (
        f"Analyze this dataset and provide key insights.\n\n"
        f"Column Names: {sample_data.columns.tolist()}\n\n"
        f"Summary Stats:\n{summary}\n\n"
        f"Missing Values:\n{missing_values}\n\n"
        f"Correlations:\n{correlations}\n\n"
        f"Outliers Detected:\n{outliers}\n\n"
        f"Sample Data:\n{sample_data.head().to_string()}\n\n"
        f"Provide key insights, trends, and any anomalies detected."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def generate_readme(analysis, dataset_name):
    """Save AI-generated analysis to README.md."""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(f"# {dataset_name} - Data Analysis Report\n\n")
        f.write(analysis)

def save_results(dataset_name):
    """Move README.md to dataset folder."""
    os.makedirs(dataset_name, exist_ok=True)
    if os.path.exists("README.md"):
        os.rename("README.md", f"{dataset_name}/README.md")

def main():
    dataset_name = os.path.splitext(os.path.basename(filename))[0]
    df = load_data(filename)
    summary, missing_values, correlations, outliers = analyze_data(df)
    insights = get_ai_insights(summary, missing_values, correlations, outliers, df)
    generate_readme(insights, dataset_name)
    save_results(dataset_name)
    print(f"Analysis complete. Check {dataset_name}/README.md for insights.")

if __name__ == "__main__":
    main()
