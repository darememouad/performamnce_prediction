import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv('exams.csv')

# Perform data preprocessing (similar to the previous steps)

# Define the features (X) and target variable (y)
X = df.drop('math score', axis=1)
y = df['math score']

# Perform one-hot encoding on the features
X_encoded = pd.get_dummies(X, columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'])

# Create an instance of the Linear Regression model
model = LinearRegression()

# Train the model on the entire dataset
model.fit(X_encoded, y)

# Streamlit app title and menu bar
st.title("Student Performance Predictor")

# Add markdown with explanation and welcome
st.markdown("Welcome to the Student Performance Predictor app! This app uses machine learning techniques to develop "
            "models that predict student performance based on demographic and other factors. You can choose either to "
            "predict the math score or visualize different aspects of the dataset.")

# Sidebar inputs for prediction
st.sidebar.header("Input Features")

gender = st.sidebar.selectbox("Gender", df['gender'].unique())
race = st.sidebar.selectbox("Race/Ethnicity", df['race/ethnicity'].unique())
parent_education = st.sidebar.selectbox("Parental Level of Education", df['parental level of education'].unique())
lunch = st.sidebar.selectbox("Lunch", df['lunch'].unique())
test_prep = st.sidebar.selectbox("Test Preparation Course", df['test preparation course'].unique())
reading_score = st.sidebar.slider("Reading Score", min_value=0, max_value=100, step=1, value=50)
writing_score = st.sidebar.slider("Writing Score", min_value=0, max_value=100, step=1, value=50)

# Process the prediction
if st.sidebar.button('Predict'):
    if not (gender and race and parent_education and lunch and test_prep):
        st.sidebar.warning("Please enter all features.")
    else:
        # Create a DataFrame from the input data
        input_data = {
            'gender': [gender],
            'race/ethnicity': [race],
            'parental level of education': [parent_education],
            'lunch': [lunch],
            'test preparation course': [test_prep],
            'reading score': [reading_score],
            'writing score': [writing_score]
        }
        input_df = pd.DataFrame(input_data)

        # Perform one-hot encoding on the input DataFrame
        input_df_encoded = pd.get_dummies(input_df, columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'])

        # Ensure input DataFrame has the same columns as the training data
        input_df_encoded = input_df_encoded.reindex(columns=X_encoded.columns, fill_value=0)

        # Make predictions using the pre-trained model
        predicted_math_score = model.predict(input_df_encoded)

        # Calculate the accuracy of the model's predictions (R-squared)
        y_pred_train = model.predict(X_encoded)
        accuracy = r2_score(y, y_pred_train)

        # Display the predicted math score
        st.subheader("Predicted Math Score")
        st.write(f"{predicted_math_score[0]:.2f}%")

        # Display the accuracy of the prediction
        st.subheader("Model Accuracy")
        st.write(f"{accuracy*100:.2f}%")

# Sidebar inputs for visualization
st.sidebar.header("Visualizations")

# Create a form for visualization
with st.sidebar.form(key='visualization_form'):
    selected_visualization = st.sidebar.selectbox("Select Visualization", ["Distribution of Math Scores", "Math Score vs. Writing Score"])

    # Visualization button
    visualization_button = st.form_submit_button('Visualize')

