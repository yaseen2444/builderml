# Save this as `ml_model_builder_app.py`

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, accuracy_score
import pickle
from scipy import stats

# Initialize session state for data and model
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None

def load_data(file):
    extension = file.name.split('.')[-1]
    if extension == 'csv':
        return pd.read_csv(file)
    elif extension == 'xlsx':
        return pd.read_excel(file)
    else:
        st.error("Unsupported file type!")
        return None

def handle_categorical_data(df):
    categorical_cols = df.select_dtypes(['object']).columns
    if categorical_cols.empty:
        st.write("No categorical columns found.")
        return df
    encoding_method = st.selectbox("Choose Encoding Method", ["One-Hot Encoding", "Label Encoding"])
    if encoding_method == "One-Hot Encoding":
        df = pd.get_dummies(df, columns=categorical_cols)
    elif encoding_method == "Label Encoding":
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col])
    return df

def handle_null_values(df):
    st.write("Null values are:", df.isnull().sum())
    handling_method = st.selectbox("Choose Method to Handle Null Values", ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode"])
    if handling_method == "Drop rows":
        df.dropna(inplace=True)
    elif handling_method == "Fill with mean":
        df.fillna(df.mean(), inplace=True)
    elif handling_method == "Fill with median":
        df.fillna(df.median(), inplace=True)
    elif handling_method == "Fill with mode":
        df.fillna(df.mode().iloc[0], inplace=True)
    return df

def handle_duplicates(df):
    if st.button("Drop duplicates"):
        original_len = len(df)
        df.drop_duplicates(inplace=True)
        st.write(f"Dropped {original_len - len(df)} duplicates. Remaining rows: {len(df)}")
    return df

def handle_outliers(df):
    numerical_df = df.select_dtypes(include=[np.number])
    if numerical_df.empty:
        st.write("No numerical columns to check for outliers.")
        return df
    
    z_scores = np.abs(stats.zscore(numerical_df))
    threshold = 3
    outliers = (z_scores > threshold).any(axis=1)
    outliers_count = np.sum(outliers)
    st.write(f"Detected {outliers_count} rows as outliers.")
    
    if outliers_count > 0:
        if st.button("Remove Outliers"):
            df = df[~outliers]
            st.write(f"Removed {outliers_count} outliers.")
    return df

def feature_scaling(df):
    scaling_method = st.selectbox("Choose Scaling Method", ["Standard Scaling", "Min-Max Scaling", "Log Scaling"])
    st.write(f"Selected Scaling Method: {scaling_method}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if numeric_cols.empty:
        st.write("No numeric columns found for scaling.")
        return df
    
    try:
        if scaling_method == "Standard Scaling":
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        elif scaling_method == "Min-Max Scaling":
            scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        elif scaling_method == "Log Scaling":
            df[numeric_cols] = np.log1p(df[numeric_cols])
    except ValueError as e:
        st.error(f"An error occurred during scaling: {e}")
    return df

def select_variables(df):
    st.write("Selected variables:")
    y_column = st.selectbox("Choose the dependent variable", df.columns)
    X_columns = [column for column in df.columns if column != y_column]
    st.write(f"Independent variables: {X_columns}")
    X = df[X_columns]
    y = df[y_column]
    return X, y

def select_model():
    model_type = st.selectbox("Choose Model Type", ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "Support Vector Regressor", "Logistic Regression"])
    st.write(f"Selected Model: {model_type}")
    if model_type == "Linear Regression":
        return LinearRegression()
    elif model_type == "Decision Tree Regressor":
        return DecisionTreeRegressor()
    elif model_type == "Random Forest Regressor":
        return RandomForestRegressor()
    elif model_type == "Support Vector Regressor":
        return SVR()
    elif model_type == "Logistic Regression":
        return LogisticRegression(max_iter=1000)

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_type):
    y_pred = model.predict(X_test)
    if model_type in ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "Support Vector Regressor"]:
        mse = mean_squared_error(y_test, y_pred)
        st.write("Mean Squared Error (MSE):", mse)
        r2 = model.score(X_test, y_test)
        st.write("R^2 Score:", r2)
        if r2 >= 0.8:
            st.write("Model performance: Excellent")
        elif r2 >= 0.6:
            st.write("Model performance: Good")
        else:
            st.write("Model performance: Bad")
    elif model_type == "Logistic Regression":
        acc = accuracy_score(y_test, y_pred)
        st.write("Accuracy:", acc)
        if acc >= 0.8:
            st.write("Model performance: Excellent")
        elif acc >= 0.6:
            st.write("Model performance: Good")
        else:
            st.write("Model performance: Bad")
    st.write("Model evaluation complete.")

def save_model(model):
    save_path = st.text_input("Enter the file path to save the model", "model.pkl")
    if st.button("Save Model"):
        with open(save_path, 'wb') as f:
            pickle.dump(model,f)
        st.write(f"Model saved to {save_path}")

def load_model():
    load_path = st.text_input("Enter the file path to load the model", "model.pkl")
    if st.button("Load Model"):
        with open(load_path, 'rb') as f:
            model = pickle.load(f)
        st.session_state.model = model
        st.write(f"Model loaded from {load_path}")

# Streamlit App Interface
st.title("ML MODEL BUILDER....")

task = st.sidebar.selectbox("Choose Task", ["Data Loading", "Data Exploration", "Data Preprocessing", "Feature Scaling", "Variable Selection", "Model Selection", "Model Training", "Model Evaluation", "Save Model", "Load and Use Model"])

if task == "Data Loading":
    uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state.df = df
            st.write("Dataset Loaded:")
            st.write(df.head())

elif task == "Data Exploration":
    if st.session_state.df is not None:
        df = st.session_state.df
        option = st.selectbox("Choose an option", ["Describe a Specific column", "Show Properties of each column", "Show the data set", "Remove unwanted columns"])
        if option == "Describe a Specific column":
            col_name = st.selectbox("Choose column", df.columns)
            st.write(df[col_name].describe())
        elif option == "Show Properties of each column":
            st.write("General Information of DataFrame:")
            st.write(df.info())
            st.write("Detailed Column Properties:")
            for col in df.columns:
                st.subheader(f"Properties of column: {col}")
                if df[col].dtype == 'object':
                    st.write(df[col].value_counts())
                else:
                    st.write(df[col].describe())
        elif option == "Show the data set":
            st.write(df)
        elif option == "Remove unwanted columns":
            columns_to_remove = st.multiselect("Select columns to remove", df.columns)
            df.drop(columns=columns_to_remove, inplace=True)
            st.session_state.df = df
            st.write("Updated DataFrame:")
            st.write(df.head())
            st.write(f"Removed columns: {columns_to_remove}")
    else:
        st.error("Please load data first.")

elif task == "Data Preprocessing":
    if st.session_state.df is not None:
        df = st.session_state.df
        option = st.selectbox("Choose an option", ["Handle Categorical data", "Handle Null Values", "Handle duplicates", "Handle Outliers"])
        if option == "Handle Categorical data":
            df = handle_categorical_data(df)
        elif option == "Handle Null Values":
            df = handle_null_values(df)
        elif option == "Handle duplicates":
            df = handle_duplicates(df)
        elif option == "Handle Outliers":
            df = handle_outliers(df)
        st.session_state.df = df
    else:
        st.error("Please load data first.")

elif task == "Feature Scaling":
    if st.session_state.df is not None:
        df = st.session_state.df
        df = feature_scaling(df)
        st.session_state.df = df
    else:
        st.error("Please load data first.")

elif task == "Variable Selection":
    if st.session_state.df is not None:
        df = st.session_state.df
        X, y = select_variables(df)
        st.session_state.X = X
        st.session_state.y = y
    else:
        st.error("Please load data first.")

elif task == "Model Selection":
    st.session_state.model = select_model()

elif task == "Model Training":
    if 'X' in st.session_state and 'y' in st.session_state and st.session_state.model is not None:
        X_train, X_test, y_train, y_test = train_test_split(st.session_state.X, st.session_state.y, test_size=0.2, random_state=42)
        st.session_state.model = train_model(st.session_state.model, X_train, y_train)
        st.write("Model trained successfully.")
    else:
        st.error("Please complete variable selection and model selection first.")

elif task == "Model Evaluation":
    if 'X' in st.session_state and 'y' in st.session_state and st.session_state.model is not None:
        X_train, X_test, y_train, y_test = train_test_split(st.session_state.X, st.session_state.y, test_size=0.2, random_state=42)
        model_type = st.selectbox("Choose Model Type", ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "Support Vector Regressor", "Logistic Regression"])
        evaluate_model(st.session_state.model, X_test, y_test, model_type)
    else:
        st.error("Please complete model training first.")

elif task == "Save Model":
    if st.session_state.model is not None:
        save_model(st.session_state.model)
    else:
        st.error("Please train a model first.")

elif task == "Load and Use Model":
    load_model()
    if st.session_state.model is not None:
        if 'X' in st.session_state and 'y' in st.session_state:
            X_train, X_test, y_train, y_test = train_test_split(st.session_state.X, st.session_state.y, test_size=0.2, random_state=42)
            model_type = st.session_state.model.__class__.__name__
            evaluate_model(st.session_state.model, X_test, y_test, model_type)
            if model_type == "LogisticRegression":
                prediction = st.session_state.model.predict(X_test)
                st.write("Accuracy is: ", accuracy_score(y_test, prediction))
        else:
            st.error("Please complete variable selection first.")
    else:
        st.error("Please load a model first.")
