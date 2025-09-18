from mlflow import MlflowClient
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor

client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

all_experiments = client.search_experiments()

# # print('hi')
print(all_experiments)

default_experiment = [
    {"name": experiment.name, "lifecycle_stage": experiment.lifecycle_stage}
    for experiment in all_experiments
    if experiment.name == "Default"
][0]

pprint(default_experiment)


# Provide an Experiment description that will appear in the UI
experiment_description = (
    "This is the house price forecasting project. "
    "This experiment contains the produce models for house pricing."
)

# Provide searchable tags that define characteristics of the Runs that
# will be in this Experiment
experiment_tags = {
    "project_name": "housing-price-forecasting",
    "store_dept": "realestate",
    "team": "stores-ml",
    "project_quarter": "Q3-2025",
    "mlflow.note.content": experiment_description,
}

# Create the Experiment, providing a unique name
housing_price_experiment = client.create_experiment(
    name="Housing_Models", tags=experiment_tags
)
