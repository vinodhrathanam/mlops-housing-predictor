import pandas as pd
import dvc.api
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


# Pull the original training data from DVC
path = dvc.api.get_url('data/Housing.csv')
reference_data = pd.read_csv(path)

# In a real-world scenario, you would pull this from a live database
# For this project, let's create a simulated "live" dataset with drift
current_data = reference_data.copy()
current_data['area'] = current_data['area'] * 1.1 # Simulate 10% drift in square footage

# Generate the data drift report
data_drift_report = Report(metrics=[DataDriftPreset()])
data_drift_report.run(
    reference_data=reference_data,
    current_data=current_data
)

# Save the report
data_drift_report.save_html('data_drift_report.html')

# You can also get a JSON summary and check for drift
report_json = data_drift_report.as_dict()
drift_detected = report_json['metrics'][0]['result']['dataset_drift']

if drift_detected:
    print("Data drift detected. The model may be degrading. A retraining event will be triggered.")
    # Trigger your model retraining pipeline here!
    # This could be a curl command to a webhook or another GitHub Action
else:
    print("No significant data drift detected. The model is stable.")