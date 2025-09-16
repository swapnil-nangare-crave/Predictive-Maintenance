import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Load the dataset
df = pd.read_csv('./data/ai4i2020.csv')

# Drop unnecessary columns
df.drop(['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Type'], axis=1, inplace=True)

# Define features (X) and target (y)
X = df.drop('Machine failure', axis=1)
y = df['Machine failure']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Initialize and fit the scaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')

# Initialize and train the model
model = GradientBoostingClassifier()
model.fit(X_train_scaled, y_train)

# Save the trained model
joblib.dump(model, 'gradient_boosting_model.pkl')

print("Model and scaler have been retrained and saved.")