from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import time # Import time for potential future auto-refresh delay

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Load the trained model and scaler
try:
    model = joblib.load('gradient_boosting_model.pkl')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# Load the original dataset (needed for Random Data Explorer)
try:
    df_original = pd.read_csv('./data/ai4i2020.csv')
except FileNotFoundError:
    print("Original data file 'ai4i2020.csv' not found. Please ensure it's in the 'data' subdirectory.")
    df_original = None


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    prediction_proba = None

    if request.method == 'POST':
        # Extract form data
        air_temp = float(request.form['air_temp'])
        process_temp = float(request.form['process_temp'])
        rotational_speed = int(request.form['rotational_speed'])
        torque = float(request.form['torque'])
        tool_wear = int(request.form['tool_wear'])

        # Create a dataframe from the user inputs
        features = pd.DataFrame({
            'Air temperature [K]': [air_temp],
            'Process temperature [K]': [process_temp],
            'Rotational speed [rpm]': [rotational_speed],
            'Torque [Nm]': [torque],
            'Tool wear [min]': [tool_wear]
        })

        # Scale the input features and make prediction
        if model and scaler:
            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)
            prediction_proba = model.predict_proba(scaled_features)
            prediction_result = "likely to fail" if prediction[0] == 1 else "not likely to fail"
            prediction_proba = f"{prediction_proba[0][1]:.2f}"
        else:
            prediction_result = "Error: Model or scaler not loaded."

    return render_template('index.html',
                           prediction_result=prediction_result,
                           prediction_proba=prediction_proba)

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    required_features = [
        'air_temp', 'process_temp', 'rotational_speed', 'torque', 'tool_wear'
    ]

    for feature in required_features:
        if feature not in data:
            return jsonify({"error": f"Missing feature: {feature}"}), 400

    try:
        air_temp = float(data['air_temp'])
        process_temp = float(data['process_temp'])
        rotational_speed = int(data['rotational_speed'])
        torque = float(data['torque'])
        tool_wear = int(data['tool_wear'])
    except ValueError:
        return jsonify({"error": "Invalid data type for one or more features"}), 400

    features = pd.DataFrame({
        'Air temperature [K]': [air_temp],
        'Process temperature [K]': [process_temp],
        'Rotational speed [rpm]': [rotational_speed],
        'Torque [Nm]': [torque],
        'Tool wear [min]': [tool_wear]
    })

    if model and scaler:
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)

        result = {
            "prediction": int(prediction[0]),
            "prediction_text": "likely to fail" if prediction[0] == 1 else "not likely to fail",
            "probability_of_failure": float(f"{prediction_proba[0][1]:.2f}")
        }
        return jsonify(result), 200
    else:
        return jsonify({"error": "Model or scaler not loaded on the server."}), 500

@app.route('/simulation_page')
def simulation_page():
    return render_template('simulation.html')

@app.route('/simulate', methods=['GET'])
def simulate():
    if df_original is None:
        return jsonify({"error": "Original data not loaded."}), 500
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded."}), 500

    model_features_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    target_col = 'Machine failure'

    # Select a random row from the original dataset
    random_row = df_original.sample(n=1)
    random_row_index = random_row.index[0]

    # Get features for the model
    features_for_model = random_row[model_features_cols]

    # Scale features and make prediction
    scaled_features_for_model = scaler.transform(features_for_model)
    prediction = model.predict(scaled_features_for_model)
    prediction_proba = model.predict_proba(scaled_features_for_model)
    actual_failure = random_row[target_col].iloc[0]

    # Prepare data for JSON response
    simulated_data = random_row.iloc[0].to_dict()
    # Convert numpy types to native Python types for JSON serialization
    for key, value in simulated_data.items():
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            simulated_data[key] = str(value)
        elif hasattr(value, 'item'): # For numpy types
            simulated_data[key] = value.item()

    result = {
        "simulated_data": simulated_data,
        "prediction": int(prediction[0]),
        "prediction_text": "likely to fail" if prediction[0] == 1 else "not likely to fail",
        "probability_of_failure": float(f"{prediction_proba[0][1]:.2f}"),
        "actual_failure": int(actual_failure),
        "actual_failure_text": "Failure occurred" if actual_failure == 1 else "No failure"
    }

    return jsonify(result), 200

if __name__ == '__main__':
    app.run(debug=True)
