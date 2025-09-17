# Predictive Maintenance Application

## Overview
This application provides a web-based interface for predicting machine failures and simulating machine behavior based on historical data. It leverages a pre-trained Gradient Boosting Classifier model to offer insights into potential machine failures, helping in proactive maintenance.

## Features

*   **Predictive Maintenance Interface:** Allows users to input various machine parameters and get an instant prediction on whether the machine is likely to fail.
*   **Simulation View:** Provides a dynamic simulation where the model predicts on randomly selected data points from the original dataset, showcasing the model's performance against actual outcomes.
*   **REST API Endpoint:** Exposes a `/predict` API endpoint, enabling external applications to send machine parameters and receive failure predictions in JSON format.
*   **Modern User Interface:** Designed with a minimal, consistent, and premium look using Bootstrap 5, ensuring a responsive and intuitive user experience.
*   **Two-Page Structure:** Separates the predictive maintenance input and simulation views into distinct pages for better organization and navigation.

## Technologies Used

*   **Backend:** Python (Flask)
*   **Machine Learning:** scikit-learn (Gradient Boosting Classifier, StandardScaler), joblib
*   **Data Handling:** pandas
*   **Frontend:** HTML5, CSS3, JavaScript, Bootstrap 5
*   **CORS:** Flask-CORS

## Setup Instructions

To get this application up and running on your local machine, follow these steps:

1.  **Navigate to the Project Directory:**
    ```bash
    cd  project
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment:**
    *   **Windows:**
        ```bash
        .venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```

4.  **Install Dependencies:**
    Install all required Python packages using `pip`:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure `requirements.txt` contains `Flask`, `pandas`, `scikit-learn`, `joblib`, `flask-cors`)*

5.  **Ensure Model and Data Files are in Place:**
    *   `gradient_boosting_model.pkl`: Pre-trained machine learning model.
    *   `scaler.joblib`: Pre-trained data scaler.
    *   `data/ai4i2020.csv`: The dataset used for training and simulation.
    These files should be in their respective locations as shown in the project structure.

## How to Run

1.  **Start the Flask Application:**
    With your virtual environment activated, run the main application file:
    ```bash
    python flask_app.py
    ```
    The application will typically run on `http://127.0.0.1:5000/`.

2.  **Access the Web UI:**
    Open your web browser and navigate to:
    *   **Predictive Maintenance Page:** `http://127.0.0.1:5000/`
    *   **Simulation View Page:** `http://127.0.0.1:5000/simulation_page`

3.  **Using the REST API Endpoint:**
    You can send `POST` requests to the `/predict` endpoint to get predictions programmatically.

    **Endpoint:** `http://127.0.0.1:5000/predict`
    **Method:** `POST`
    **Content-Type:** `application/json`

    **Example Request Body:**
    ```json
    {
        "air_temp": 300.0,
        "process_temp": 310.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 100
    }
    ```

    **Example `curl` command:**
    ```bash
    curl -X POST -H "Content-Type: application/json" \
         -d "{\"air_temp\": 300.0, \"process_temp\": 310.0, \"rotational_speed\": 1500, \"torque\": 40.0, \"tool_wear\": 100}" \
         http://127.0.0.1:5000/predict
    ```

    **Example Python `requests`:**
    ```python
    import requests

    url = "http://127.0.0.1:5000/predict"
    headers = {"Content-Type": "application/json"}
    data = {
        "air_temp": 300.0,
        "process_temp": 310.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 100
    }

    response = requests.post(url, headers=headers, json=data)
    print(response.json())
    ```

## Project Structure

```
Predictive Maintenance UI/
├── app.py (Original Streamlit app - now deprecated)
├── flask_app.py (Main Flask application file)
├── gradient_boosting_model.pkl (Pre-trained ML model)
├── predictive-maintenance-milling-machine.ipynb (Jupyter Notebook for model development)
├── requirements.txt (Python dependencies)
├── scaler.joblib (Pre-trained data scaler)
├── train.py (Script for training the ML model)
├── README.md (This file)
├── .venv/ (Python virtual environment)
├── data/
│   └── ai4i2020.csv (Dataset)
└── templates/
    ├── index.html (HTML template for Predictive Maintenance page)
    └── simulation.html (HTML template for Simulation View page)
```

## Future Enhancements

*   **Plotly Integration:** Migrate the Plotly visualizations from the original Streamlit app to the Flask application.
*   **Auto-refresh for Simulation:** Implement client-side JavaScript to periodically fetch new simulation data.
*   **More Robust Error Handling:** Enhance error messages and user feedback.
*   **User Authentication:** Add user login/registration for secure access.
*   **Database Integration:** Store historical predictions or user data in a database.
