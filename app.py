
import streamlit as st
import pandas as pd
import joblib
import time # Needed for the random data explorer

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
    df_original = pd.read_csv('./data/ai4i2020.csv')
except FileNotFoundError:
    st.error("Original data file 'ai4i2020.csv' not found. Please ensure it's in the 'data' subdirectory.")
    st.stop()


st.title('Predictive Maintenance Application')

tab1, tab2 = st.tabs(["Predictive Maintenance", "Simulation"])

with tab1:
    st.markdown('### Set the Data to Predict Machine Failure')

    # Create columns for a cleaner layout
    # col1, col2 = st.columns(2)

    # with col1:
    #     air_temp = st.slider('Air temperature [K]', min_value=290.0, max_value=310.0, value=300.0, step=0.1)
    #     rotational_speed = st.slider('Rotational speed [rpm]', min_value=1000, max_value=3000, value=1500)
    #     tool_wear = st.slider('Tool wear [min]', min_value=0, max_value=300, value=100)

    # with col2:
    #     process_temp = st.slider('Process temperature [K]', min_value=300.0, max_value=320.0, value=310.0, step=0.1)
    #     torque = st.slider('Torque [Nm]', min_value=0.0, max_value=100.0, value=40.0, step=0.1)

    air_temp = st.slider('Air temperature [K]', min_value=290.0, max_value=310.0, value=300.0, step=0.1)
    rotational_speed = st.slider('Rotational speed [rpm]', min_value=1000, max_value=3000, value=1500)
    tool_wear = st.slider('Tool wear [min]', min_value=0, max_value=300, value=100)
    process_temp = st.slider('Process temperature [K]', min_value=300.0, max_value=320.0, value=310.0, step=0.1)
    torque = st.slider('Torque [Nm]', min_value=0.0, max_value=100.0, value=40.0, step=0.1)


    # st.text('The following inputs are present in data and do not affect the prediction.')

    # col3, col4 = st.columns(2)

    # with col3:
    #     uid = st.text_input('UID', value='1')
    #     product_id = st.text_input('Product ID', value='M14860')
    #     type_val = st.selectbox('Type', options=['L', 'M', 'H'], index=0) # Assuming 'L' is default

    # with col4:
    #     twf = st.checkbox('TWF (Tool Wear Failure)', value=False)
    #     hdf = st.checkbox('HDF (Heat Dissipation Failure)', value=False)
    #     pwf = st.checkbox('PWF (Power Failure)', value=False)
    #     osf = st.checkbox('OSF (Overstrain Failure)', value=False)
    #     rnf = st.checkbox('RNF (Random Failure)', value=False)


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

            st.header('Prediction Result')

            if prediction[0] == 0:
                st.success('The machine is not likely to fail.')
            else:
                st.error('The machine is likely to fail.')
                
                # The notebook mentions different failure types, but the model output is just 0 or 1.
                # To provide more detail, we would need to know how the failure types are encoded.
                # For now, we'll just show the probability of failure.
                st.write(f"Probability of failure: {prediction_proba[0][1]:.2f}")


        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

with tab2:
    st.markdown('### Simulation View')
    st.write("Watch the model predict on randomly selected data points from the original dataset.")

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

        st.session_state.random_data_display = all_features_for_display
        st.session_state.prediction_result = prediction[0]
        st.session_state.prediction_proba_result = prediction_proba[0][1]
        st.session_state.actual_value = real_failure_value

    # Button to manually generate new data
    st.button("Simulate on New Data", type="secondary", on_click=generate_and_predict_random_data_callback)

    # Auto-refresh checkbox
    auto_refresh = st.checkbox("Auto-refresh random data (every 1 seconds)", value=False) # Default to False to avoid immediate loop

    # Generate initial data if not already present
    if st.session_state.random_data_display is None:
        generate_and_predict_random_data_callback()
        st.rerun() # Rerun to display the initial data

    # Display the results from session state
    if st.session_state.random_data_display is not None:
        st.markdown("#### Randomly Selected Data Point:")
        
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

    # Logic for auto-refresh
    if auto_refresh:
        time.sleep(1) # Delay before rerunning
        generate_and_predict_random_data_callback() # Generate new data before rerunning
        st.rerun()

