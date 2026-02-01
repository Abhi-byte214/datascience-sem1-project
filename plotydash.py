import plotly.express as px

def interactive_scatter(df):
    fig = px.scatter(
        df,
        x="BMI",
        y="Glucose",
        color="Outcome",
        title="Interactive Scatter Plot: BMI vs Glucose",
        labels={
            "BMI": "Body Mass Index",
            "Glucose": "Glucose Level",
            "Outcome": "Diabetes Outcome"
        }
    )
    fig.show()
