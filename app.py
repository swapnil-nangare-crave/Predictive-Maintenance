
import streamlit as st
import pandas as pd
import joblib
import time # Needed for the random data explorer
import datetime

st.set_page_config(page_title="Predictive Maintenance App", layout="wide")

st.markdown("""
<style>
div[data-testid="stMetric"] {
    border: 1px solid #ccc;
    border-radius: 5px;
    padding: 10px;
    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
    margin-bottom: 10px; /* Add some space between metrics */
}

/* Target Streamlit secondary buttons */
.stButton button[kind="secondary"] {
    background-color: #ff4b4b;
    color: white;
    border-color: #ff4b4b; /* Optional: to match border with background */
}

/* Increase font size of st.tabs labels */
#  .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {      
#       size: 20px; /* Adjust this value as needed */                                     
#       font-size: 2.8rem; /* Adjust this value as needed */                              
#  }                                                                                     
                                                                                        
#   /* Increase size of tab containers */                                                 
#   .stTabs [data-baseweb="tab-list"] button {                                            
#       padding: 30px 15px; /* Adjust padding as needed (top/bottom, left/right) */       
# }
</style>
""", unsafe_allow_html=True)

# Load the trained model and scaler (common to both sections)
try:
    model = joblib.load('gradient_boosting_model.pkl')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError as e:
    st.error(f"Model or scaler not found: {e}. Please ensure 'gradient_boosting_model.pkl' and 'scaler.joblib' are in the main project directory.")
    st.stop()

# Load the original dataset (needed for Random Data Explorer)
try:
    df_original = pd.read_csv('./data/twice_freq.csv')
except FileNotFoundError:
    st.error("Original data file 'twice_freq.csv' not found. Please ensure it's in the 'data' subdirectory.")
    st.stop()

_, col2, _ = st.columns([1,6,1])
with col2:
    st.markdown('# Predictive Maintenance Application')
    st.markdown('**MILLING MACHINE** failure prediction using a Gradient Boosting Classifier.')

    tab1, tab2 = st.tabs(["Prediction View", "Simulation View"])

    with tab1:
        air_temp = st.slider('Air temperature [K]', min_value=290.0, max_value=310.0, value=300.0, step=0.1)
        process_temp = st.slider('Process temperature [K]', min_value=300.0, max_value=320.0, value=310.0, step=0.1)
        rotational_speed = st.slider('Rotational speed [rpm]', min_value=1000, max_value=3000, value=1500)
        torque = st.slider('Torque [Nm]', min_value=0.0, max_value=100.0, value=40.0, step=0.1)
        tool_wear = st.slider('Tool wear [min]', min_value=0, max_value=300, value=100)
        


        # Predict button
        if st.button('Predict Failure', type="secondary"):
            # Create a dataframe from the user inputs
            features = pd.DataFrame({
                'Air temperature [K]': [air_temp],
                'Process temperature [K]': [process_temp],
                'Rotational speed [rpm]': [rotational_speed],
                'Torque [Nm]': [torque],
                'Tool wear [min]': [tool_wear]
            })

            # Scale the input features
            scaled_features = scaler.transform(features)
            
            # Make prediction
            try:
                prediction = model.predict(scaled_features)
                prediction_proba = model.predict_proba(scaled_features)

                if prediction[0] == 0:
                    st.success('The machine is not likely to fail.')
                else:
                    st.error('The machine is likely to fail.')

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    with tab2:
        col1, col2 = st.columns([2, 1])

        with col1:
            # Prepare the original data for random selection (all columns for display)
            df_for_random_selection = df_original.copy()

            # Define features that go into the model
            model_features_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

            # Initialize session state for random data and prediction if not already present
            if 'random_data_display' not in st.session_state:
                st.session_state.random_data_display = None
                st.session_state.prediction_result = None
                st.session_state.prediction_proba_result = None
                st.session_state.actual_value = None
                st.session_state.failure_reason = None
                st.session_state.failure_log = []

            def generate_and_predict_random_data_callback():
                random_row_index = df_for_random_selection.sample(n=1).index[0]
                
                # Get the features for the model from the selected row
                features_for_model = df_for_random_selection.loc[[random_row_index], model_features_cols]
                
                # Get all features for display
                all_features_for_display = df_for_random_selection.loc[[random_row_index]]

                # Scale the features for the model
                scaled_features_for_model = scaler.transform(features_for_model)
                
                # Make prediction
                prediction = model.predict(scaled_features_for_model)
                prediction_proba = model.predict_proba(scaled_features_for_model)
                real_failure_value = df_original.loc[random_row_index, 'Machine failure']

                # Determine failure reason if applicable
                failure_reason = None
                if real_failure_value == 1:
                    failure_types = {
                        "TWF": "Tool Wear Failure",
                        "HDF": "Heat Dissipation Failure",
                        "PWF": "Power Failure",
                        "OSF": "Overstrain Failure",
                        "RNF": "Random Failure"
                    }
                    for col, reason in failure_types.items():
                        if df_original.loc[random_row_index, col] == 1:
                            failure_reason = reason
                            break  # Assume one reason is enough
                    
                    # New logging logic
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_entry = {
                        "timestamp": timestamp,
                        "reason": failure_reason,
                        "values": all_features_for_display.iloc[0].to_dict()
                    }
                    
                    st.session_state.failure_log.append(log_entry)


                st.session_state.random_data_display = all_features_for_display
                st.session_state.prediction_result = prediction[0]
                st.session_state.prediction_proba_result = prediction_proba[0][1]
                st.session_state.actual_value = real_failure_value
                st.session_state.failure_reason = failure_reason

            def toggle_auto_refresh():
                st.session_state.simulate = not st.session_state.simulate

            # Button to manually generate new data
            if 'simulate' not in st.session_state:
                st.session_state.simulate = False
            st.button("Simulate", on_click=toggle_auto_refresh)

            
            # Generate initial data if not already present
            if st.session_state.random_data_display is None:
                generate_and_predict_random_data_callback()
                st.rerun() # Rerun to display the initial data

            # Display the results from session state
            if st.session_state.random_data_display is not None:
                
                # Display features in a grid resembling "rectangular tabs"
                num_cols = 4 # Number of columns for the grid
                cols = st.columns(num_cols)
                
                all_cols = st.session_state.random_data_display.columns.tolist()
                columns = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF",]
                
                for i, col in enumerate(all_cols):
                    with cols[i % num_cols]:
                        value = st.session_state.random_data_display[col].iloc[0]
                        if col in columns:
                            if value == 1:
                                value = "Yes"
                            else:
                                value = "No"
                        st.metric(label=col, value=value)

                st.subheader("Prediction Result:")
                col_pred, col_real = st.columns(2)

                with col_pred:
                    st.write("**Predicted:**")
                    if st.session_state.prediction_result == 0:
                        st.success('Not likely to fail.')
                    else:
                        st.error('Likely to fail.')
                    # st.write(f"Probability of failure: {st.session_state.prediction_proba_result:.2f}")

                with col_real:
                    st.write("**Actual:**")
                    if st.session_state.actual_value == 0:
                        st.success('No failure.')
                    else:
                        st.error('Failure occurred.')

        with col2:
            # Display failure log
            st.subheader("Failure Log")
            if st.button("Clear Log"):
                st.session_state.failure_log = []
                st.rerun()

            for i, log_entry in enumerate(reversed(st.session_state.failure_log)):
                timestamp = log_entry["timestamp"]
                reason = log_entry["reason"]
                values = log_entry["values"]
                
                # The first log (most recent) is expanded by default
                is_expanded = i == 0

                with st.expander(f"[{timestamp}] {reason}", expanded=is_expanded):
                    st.write("Values:")
                    st.json(values)

        # Logic for auto-refresh
        if st.session_state.simulate:
            time.sleep(1) # Delay before rerunning
            generate_and_predict_random_data_callback() # Generate new data before rerunning
            st.rerun()

