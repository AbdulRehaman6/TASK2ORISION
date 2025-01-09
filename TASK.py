# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create a proper dataset with a linear relationship
data = {
    'Feature': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Target': [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0]  # Perfect linear relationship: Target = 2.5 * Feature
}
df = pd.DataFrame(data)

# Displaying the dataset
print("Dataset:")
print(df)

# Step 2: Splitting the data into training and testing sets
X = df[['Feature']]  # Feature column
y = df['Target']  # Target column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Evaluating the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Step 5: Visualizing the regression line
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.title('Linear Regression')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()

# Step 6: Predicting output based on user input
while True:
    try:
        user_input = input("\nEnter a feature value to predict the target (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting the prediction loop.")
            break
        user_input = float(user_input)
        prediction = model.predict([[user_input]])
        print(f"The predicted target value for feature {user_input} is: {prediction[0]:.2f}")
    except ValueError:
        print("Invalid input. Please enter a numerical value or 'exit' to quit.")
