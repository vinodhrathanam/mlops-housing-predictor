import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import dvc.api

# Pull the latest data from DVC
try:
    path = dvc.api.get_url('data/Housing.csv')
    df = pd.read_csv(path)
except Exception as e:
    print(f"Error getting data from DVC: {e}")
    exit()

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Sets the current active experiment to the "Housing_Models" experiment and
# returns the Experiment metadata
apple_experiment = mlflow.set_experiment("Housing_Models")

# Define a run name for this iteration of training.
# If this is not set, a unique name will be auto-generated for your run.
run_name = "house_price_rf_test"

# Define an artifact path that the model will be saved to.
artifact_path = "rf_housing"


# Simple feature selection
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start an MLflow run
with mlflow.start_run() as run:
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions and evaluate
    predictions = model.predict(X_test)
    
    # Calculate metrics
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Log parameters and metrics to MLflow
    mlflow.log_param("model_name", "LinearRegression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    # Log the model artifact
    mlflow.sklearn.log_model(model, "housing_model")
    
    # Print results inside the run block where variables are accessible
    print(f"MLflow run_id: {run.info.run_id}")
    print(f"Model trained. RMSE: {rmse:.2f}, R2: {r2:.2f}")