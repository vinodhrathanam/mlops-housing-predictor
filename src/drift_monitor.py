import pandas as pd
import dvc.api
import pandera as pa

# Pull the original training data from DVC
path = dvc.api.get_url('data/Housing.csv')
reference_data = pd.read_csv(path)

# In a real-world scenario, you would pull this from a live database
# For this project, let's create a simulated "live" dataset with drift
current_data = reference_data.copy()
current_data['area'] = current_data['area'] * 1.1

# Define a schema with a check for drift
# We use a custom check to compare the mean of the current data to the reference data
schema = pa.DataFrameSchema(
    columns={
        "area": pa.Column(
            pa.Float, 
            checks=pa.Check(
                lambda s: s.mean() < reference_data['area'].mean() * 1.05, # Check for more than 5% drift
                element_wise=False,
                error="Area mean has drifted significantly."
            )
        )
        # You would add other columns and checks here
    }
)

# Run the validation and handle the exception
try:
    schema.validate(current_data, lazy=True)
    drift_detected = False
    print("No significant data drift detected. The model is stable.")
except pa.errors.SchemaErrors as err:
    drift_detected = True
    print("Data drift detected. The model may be degrading. A retraining event will be triggered.")
    print("Validation errors:")
    print(err.failure_cases)