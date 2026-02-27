import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Load the dataset
# Make sure 'spotify dataset CA2.csv' is in the same folder as this Python script
# Using encoding='latin-1' as Spotify datasets often contain special characters in track/artist names
try:
    df = pd.read_csv('spotify dataset CA2.csv', encoding='latin-1')
except FileNotFoundError:
    print("Error: Could not find the file 'spotify dataset CA2.csv'. Please ensure it is in the same folder as this script.")
    exit()

# 2. Data Cleaning
# Ensure 'streams' and 'in_spotify_playlists' are numeric
# errors='coerce' will turn any text (like "1,000" or errors) into NaN (Not a Number)
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
df['in_spotify_playlists'] = pd.to_numeric(df['in_spotify_playlists'], errors='coerce')

# Drop any rows where either streams or playlist data is missing
df = df.dropna(subset=['streams', 'in_spotify_playlists'])

# 3. Define Variables for Linear Regression
# X must be a 2D array (dataframe) for scikit-learn, y is a 1D array (series)
X = df[['in_spotify_playlists']] 
y = df['streams']

# 4. Initialize and Train the Model
model = LinearRegression()
model.fit(X, y)

# 5. Make Predictions (to draw the regression line)
y_pred = model.predict(X)

# 6. Calculate Performance (R-Squared) and Equation
r2 = r2_score(y, y_pred)
slope = model.coef_[0]
intercept = model.intercept_

print("--- Linear Regression Results ---")
print(f"R-Squared Score: {r2:.4f}")
print(f"Equation of the line: y = {slope:.2f}x + {intercept:.2f}")
print("---------------------------------")

# 7. Visualisation
plt.figure(figsize=(10, 6))

# Plot the actual data points
plt.scatter(X, y, color='dodgerblue', alpha=0.5, label='Actual Songs')

# Plot the predicted regression line
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line (Trend)')

# Add labels, title, and legend
plt.title('Spotify Playlists vs. Total Streams', fontsize=14, fontweight='bold')
plt.xlabel('Number of Spotify Playlists', fontsize=12)
plt.ylabel('Total Streams', fontsize=12)

# Format the y-axis to show regular numbers instead of scientific notation (e.g., 1e9)
plt.ticklabel_format(style='plain', axis='y')

plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.tight_layout()
plt.show()