# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("diabetes.csv")

# Display first few rows (optional for checking)
print(df.head())

# -------------------------------
# 1. Line Plot
# -------------------------------
plt.figure()
plt.plot(df["Age"], df["Glucose"])
plt.title("Age vs Glucose")
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.show()


# -------------------------------
# 2. Bar Chart
# -------------------------------
plt.figure()
plt.plot(df["BMI"], df["BloodPressure"])
plt.title("BMI vs Blood Pressure")
plt.xlabel("BMI")
plt.ylabel("Blood Pressure")
plt.show()


# -------------------------------
# 3. Histogram
# -------------------------------
plt.figure()
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("diabetes.csv")
plt.hist(df["BMI"], bins=10)
plt.title("Histogram of BMI")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.show()
